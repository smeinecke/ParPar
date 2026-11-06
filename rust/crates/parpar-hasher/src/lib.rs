//! Native Rust implementation of ParPar's hashing layer, ported from
//! hasher/hasher_input_base.h + md5mb (multi-buffer MD5).
//!
//! `NativeHasherInput` computes a whole-file MD5 plus per-slice
//! MD5+CRC32 records (20 bytes each) written into the caller's `block_out`
//! buffer, matching `IHasherInput` as driven by the pps_hin_* shim wrapper.
//! `NativeHasherOutput` is the MD5Multi equivalent: N independent MD5
//! streams over equal-length regions.

mod crc32;
mod md5;
#[cfg(target_arch = "aarch64")]
mod md5_neon;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod md5_aarch64;
mod md5mb;

use crc32::{crc_zero_pad, Crc32};
use md5::{digest_bytes, md5_final_block, Md5Ctx, Md5X2, MD5_BLOCKSIZE};

/// Re-exported for the read side's external file-MD5 lane (lane-split
/// hashing): the reader streams chunk bytes through it and deposits the
/// finished digest on the HasherInput.
pub use md5::Md5Ctx as FileMd5Ctx;

/// Producer-side lane-split hasher: runs BOTH MD5 lanes on the read
/// thread through the x2 kernel (~same cost as one lane via ILP). Per
/// slice it emits the lane0 chain state at the boundary (before the
/// slice's <64B tail is finalised); at file end it emits the whole-file
/// digest. The consumer then only runs CRC + the final-block step per
/// slice instead of hashing all input bytes twice.
///
/// Only usable when slice_size % 64 == 0 so every slice boundary lands
/// on a block edge (callers fall back to the file-lane-only split).
pub struct LaneProducer {
    x2: Md5X2,
    tail: [u8; MD5_BLOCKSIZE],
    tail_len: usize,
    slice_size: u64,
    slice_off: u64,
    file_len: u64,
}

impl LaneProducer {
    pub fn new(slice_size: u64) -> Self {
        debug_assert!(slice_size % MD5_BLOCKSIZE as u64 == 0);
        LaneProducer {
            x2: Md5X2::new(),
            tail: [0; MD5_BLOCKSIZE],
            tail_len: 0,
            slice_size,
            slice_off: 0,
            file_len: 0,
        }
    }

    /// Feed a chunk; pushes a lane0 state to `states` at each completed
    /// slice boundary (in stream order).
    pub fn update(&mut self, mut data: &[u8], states: &mut Vec<[u32; 4]>) {
        self.file_len = self.file_len.wrapping_add(data.len() as u64);
        if self.tail_len > 0 {
            let take = (MD5_BLOCKSIZE - self.tail_len).min(data.len());
            self.tail[self.tail_len..self.tail_len + take].copy_from_slice(&data[..take]);
            self.tail_len += take;
            data = &data[take..];
            if self.tail_len == MD5_BLOCKSIZE {
                let blk = self.tail;
                self.x2.update_block(&blk, &blk);
                self.tail_len = 0;
                self.slice_off += MD5_BLOCKSIZE as u64;
                self.check_boundary(states);
            }
            if data.is_empty() {
                return;
            }
        }
        while data.len() >= MD5_BLOCKSIZE {
            let blk = &data[..MD5_BLOCKSIZE];
            self.x2.update_block(blk, blk);
            self.slice_off += MD5_BLOCKSIZE as u64;
            self.check_boundary(states);
            data = &data[MD5_BLOCKSIZE..];
        }
        self.tail[..data.len()].copy_from_slice(data);
        self.tail_len = data.len();
    }

    fn check_boundary(&mut self, states: &mut Vec<[u32; 4]>) {
        if self.slice_off == self.slice_size {
            states.push(self.x2.extract(HASH2X_BLOCK));
            self.x2.init_lane(HASH2X_BLOCK);
            self.slice_off = 0;
        }
    }

    /// At EOF: emit the partial last slice's lane0 state (if any), then
    /// finalise lane1 into the 16-byte file MD5. A partial slice exists
    /// iff file_len % slice_size != 0 - slice_off alone undercounts: it
    /// only tracks full 64-byte blocks, so a <64-byte tail would ship no
    /// state and starve the consumer's get_block queue.
    pub fn finish(&mut self, states: &mut Vec<[u32; 4]>) -> [u8; 16] {
        if self.file_len % self.slice_size != 0 {
            states.push(self.x2.extract(HASH2X_BLOCK));
        }
        let mut st = self.x2.extract(HASH2X_FILE);
        md5_final_block(&mut st, &self.tail[..self.tail_len], self.file_len, 0);
        digest_bytes(&st)
    }
}
use parpar_backend::{BackendError, HasherInput, HasherOutput, Result};

const HASH2X_BLOCK: usize = 0;
const HASH2X_FILE: usize = 1;

/// Method description for the info display, matching the C++ naming style
/// (e.g. "BMI1+PCLMUL"). Reports the kernels actually in use.
pub fn inhash_method_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 md5x2 is always used on x86_64; the CRC half is PCLMUL only
        // when the CPU supports it
        if std::arch::is_x86_feature_detected!("bmi1") && crc32::clmul::available() {
            return "BMI1+PCLMUL";
        }
        return if crc32::clmul::available() {
            "SSE2+PCLMUL"
        } else {
            "SSE2+Generic"
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // scalar MD5 on aarch64 (C++ parity: arm64 GPR asm beats NEON on
        // the serial chain); CRC half uses ARMv8 CRC instructions
        return if crc32::arm_crc_available() {
            "Scalar+ARMCRC"
        } else {
            "Scalar+Generic"
        };
    }
    #[allow(unreachable_code)]
    "Scalar+Generic"
}

/// Recovery-output hasher (MD5Multi) method name.
pub fn outhash_method_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        // Md5Mb<4> uses SSE2 lanes on x86_64
        return "SSE2";
    }
    // Md5Mb<4> uses NEON lanes on aarch64
    #[cfg(target_arch = "aarch64")]
    return "NEON";
    #[allow(unreachable_code)]
    "Scalar"
}

/// Core two-lane hasher (HasherInput in hasher_input_base.h).
struct InputCore {
    md5: Md5X2,
    tmp: [u8; MD5_BLOCKSIZE * 2],
    tmp_len: usize,
    pos_offset: usize,
    crc: Crc32,
    data_len: [u64; 2],
    /// file MD5 is produced externally (read thread) and deposited on the
    /// HasherInput before end() - update() then only runs lane0 (block
    /// MD5 + CRC), skipping the file-lane work entirely
    file_external: bool,
    /// lane0 chain states are also produced externally (one per slice,
    /// emitted at slice boundaries by LaneProducer) - update() then only
    /// keeps CRC + the <64B tail; get_block pops the next shipped state
    block_external: bool,
    states: std::collections::VecDeque<[u32; 4]>,
}

impl InputCore {
    fn new() -> Self {
        InputCore {
            md5: Md5X2::new(),
            tmp: [0; MD5_BLOCKSIZE * 2],
            tmp_len: 0,
            pos_offset: 0,
            crc: Crc32::new(),
            data_len: [0, 0],
            file_external: false,
            block_external: false,
            states: std::collections::VecDeque::new(),
        }
    }

    fn reset(&mut self) {
        self.md5 = Md5X2::new();
        self.tmp_len = 0;
        self.states.clear();
        self.pos_offset = 0;
        self.crc = Crc32::new();
        self.data_len = [0, 0];
    }

    fn update(&mut self, data: &[u8]) {
        if self.file_external {
            self.update_lane0_only(data);
            return;
        }
        self.data_len[HASH2X_BLOCK] = self.data_len[HASH2X_BLOCK].wrapping_add(data.len() as u64);
        self.data_len[HASH2X_FILE] = self.data_len[HASH2X_FILE].wrapping_add(data.len() as u64);
        let mut data = data;

        if self.tmp_len > 0 {
            loop {
                debug_assert!(self.tmp_len >= self.pos_offset);
                let mut wanted = MD5_BLOCKSIZE + self.pos_offset - self.tmp_len;
                if data.len() < wanted {
                    self.tmp[self.tmp_len..self.tmp_len + data.len()].copy_from_slice(data);
                    self.tmp_len += data.len();
                    return;
                }
                let mut block_in_tmp = true;
                if self.tmp_len <= self.pos_offset {
                    // block hash comes from source data instead of tmp
                    wanted = MD5_BLOCKSIZE - self.tmp_len;
                    block_in_tmp = false;
                }
                self.tmp[self.tmp_len..self.tmp_len + wanted].copy_from_slice(&data[..wanted]);
                if block_in_tmp {
                    let blk: &[u8] = &self.tmp[self.pos_offset..self.pos_offset + MD5_BLOCKSIZE];
                    self.crc.update(blk);
                    self.md5.update_block(blk, &self.tmp[..MD5_BLOCKSIZE]);
                } else {
                    let blk: &[u8] = &data[..MD5_BLOCKSIZE];
                    self.crc.update(blk);
                    self.md5.update_block(blk, &self.tmp[..MD5_BLOCKSIZE]);
                }
                data = &data[wanted..];
                if self.tmp_len > self.pos_offset {
                    // both hashes came from tmp: shift file hash window across
                    self.tmp
                        .copy_within(MD5_BLOCKSIZE..MD5_BLOCKSIZE + self.pos_offset, 0);
                    self.tmp_len = self.pos_offset;
                    continue;
                }
                break;
            }
        }

        // batched CRC over the contiguous shifted-window run (same bytes,
        // one call) - per-window PCLMUL calls cost ~85ns setup each
        let blocks = data.len().saturating_sub(self.pos_offset) / MD5_BLOCKSIZE;
        if blocks > 0 {
            self.crc
                .update(&data[self.pos_offset..self.pos_offset + blocks * MD5_BLOCKSIZE]);
            for _ in 0..blocks {
                self.md5.update_block(
                    &data[self.pos_offset..self.pos_offset + MD5_BLOCKSIZE],
                    &data[..MD5_BLOCKSIZE],
                );
                data = &data[MD5_BLOCKSIZE..];
            }
        }

        self.tmp[..data.len()].copy_from_slice(data);
        self.tmp_len = data.len();
    }

    /// Lane0-only streaming: block MD5 + CRC over the raw byte stream.
    /// The file lane is supplied externally (read thread computes it) so the
    /// tmp/pos_offset double-window machinery is unnecessary - tmp is just a
    /// 64-byte tail buffer here and pos_offset stays 0.
    fn update_lane0_only(&mut self, mut data: &[u8]) {
        self.data_len[HASH2X_BLOCK] = self.data_len[HASH2X_BLOCK].wrapping_add(data.len() as u64);
        if self.tmp_len > 0 {
            let take = (MD5_BLOCKSIZE - self.tmp_len).min(data.len());
            self.tmp[self.tmp_len..self.tmp_len + take].copy_from_slice(&data[..take]);
            self.tmp_len += take;
            data = &data[take..];
            if self.tmp_len == MD5_BLOCKSIZE {
                let mut blk = [0u8; MD5_BLOCKSIZE];
                blk.copy_from_slice(&self.tmp[..MD5_BLOCKSIZE]);
                self.crc.update(&blk);
                if !self.block_external {
                    self.md5.update_lane0(&blk);
                }
                self.tmp_len = 0;
            }
            if data.is_empty() {
                return;
            }
        }
        // CRC is streamed in one call per contiguous run - per-64B calls
        // cost ~85ns each in PCLMUL setup (~3s/GiB), batched it's ~0.1s
        let full = data.len() / MD5_BLOCKSIZE * MD5_BLOCKSIZE;
        if full > 0 {
            self.crc.update(&data[..full]);
            if !self.block_external {
                for blk in data[..full].chunks_exact(MD5_BLOCKSIZE) {
                    self.md5.update_lane0(blk);
                }
            }
            data = &data[full..];
        }
        self.tmp[..data.len()].copy_from_slice(data);
        self.tmp_len = data.len();
    }

    fn reset_block(&mut self) {
        if self.tmp_len >= MD5_BLOCKSIZE {
            // push one block through the file hash
            self.md5
                .update_block(&self.tmp[..MD5_BLOCKSIZE], &self.tmp[..MD5_BLOCKSIZE]);
            self.tmp.copy_within(MD5_BLOCKSIZE..self.tmp_len, 0);
            self.tmp_len -= MD5_BLOCKSIZE;
        }
        self.md5.init_lane(HASH2X_BLOCK);
        self.crc = Crc32::new();
        // lane0-only mode keeps the tail anchored at tmp[0] (pos_offset is
        // only for aligning the file lane's window in dual mode)
        self.pos_offset = if self.file_external { 0 } else { self.tmp_len };
        self.data_len[HASH2X_BLOCK] = 0;
    }

    /// getBlock(md5crc, zeroPad): write 16-byte block MD5 + 4-byte CRC32.
    fn get_block(&mut self, out: &mut [u8], zero_pad: u64) {
        let mut st = if self.block_external {
            self.states
                .pop_front()
                .expect("lane0 state missing - reader not shipping slice states")
        } else {
            self.md5.extract(HASH2X_BLOCK)
        };
        md5_final_block(
            &mut st,
            &self.tmp[self.pos_offset..],
            self.data_len[HASH2X_BLOCK],
            zero_pad,
        );
        out[..16].copy_from_slice(&digest_bytes(&st));
        let tail_len = (self.data_len[HASH2X_BLOCK] & (MD5_BLOCKSIZE as u64 - 1)) as usize;
        let crc = self
            .crc
            .finish(&self.tmp[self.pos_offset..self.pos_offset + tail_len]);
        out[16..20].copy_from_slice(&crc_zero_pad(crc, zero_pad).to_le_bytes());
        self.reset_block();
    }

    /// end(md5): finalise the file-lane MD5 into `out`.
    fn end(&mut self, out: &mut [u8]) {
        if self.tmp_len >= MD5_BLOCKSIZE {
            // defensive: the update loop keeps tmp_len < 64, but if this is
            // ever reached the consumed block must be shifted out before the
            // tail is finalised (same fix as C++ extractFileMD5)
            self.md5
                .update_block(&self.tmp[..MD5_BLOCKSIZE], &self.tmp[..MD5_BLOCKSIZE]);
            self.tmp.copy_within(MD5_BLOCKSIZE..self.tmp_len, 0);
            self.tmp_len -= MD5_BLOCKSIZE;
        }
        let mut st = self.md5.extract(HASH2X_FILE);
        md5_final_block(&mut st, &self.tmp, self.data_len[HASH2X_FILE], 0);
        out[..16].copy_from_slice(&digest_bytes(&st));
    }
}

/// `HasherInput` with the shim's slice-boundary record emission
/// (pps_hin_update / pps_hin_end).
pub struct NativeHasherInput {
    core: InputCore,
    block_size: u64,
    pos: u64,
    block_out: Option<Box<[u8]>>,
    ptr: usize,
    count: i64,
    ended: bool,
    /// file MD5 supplied by the read thread (external lane1) - deposited via
    /// set_file_md5 before end() writes it out
    ext_md5: Option<[u8; 16]>,
}

impl NativeHasherInput {
    pub fn new(slice_size: u64, block_out: Box<[u8]>) -> Self {
        // slice_size 0 is a caller bug - update()/end() reject it with an
        // error rather than corrupting pos accounting

        let count = (block_out.len() / 20) as i64;
        NativeHasherInput {
            core: InputCore::new(),
            block_size: slice_size,
            pos: 0,
            block_out: Some(block_out),
            ptr: 0,
            count,
            ended: false,
            ext_md5: None,
        }
    }

    /// Lane-split variant: the whole-file MD5 is computed by the read
    /// thread and handed in via set_file_md5; this hasher only runs the
    /// per-slice block MD5 + CRC lane. When `ship_states` is set the read
    /// thread also ships per-slice lane0 chain states (slice_size % 64
    /// == 0 required), so this hasher only runs CRC + finalisation.
    pub fn new_ext(slice_size: u64, block_out: Box<[u8]>, ship_states: bool) -> Self {
        let mut h = Self::new(slice_size, block_out);
        h.core.file_external = true;
        h.core.block_external = ship_states;
        h
    }
}

impl HasherInput for NativeHasherInput {
    fn update(&mut self, buf: &[u8]) -> Result<()> {
        if self.ended {
            return Err(BackendError::new("Process already ended"));
        }
        if self.block_size == 0 {
            // new() debug_asserts slice_size > 0; keep release callers safe
            return Err(BackendError::new("Slice size must be positive"));
        }
        let mut src = buf;
        let mut block_left = self.block_size - self.pos;
        while block_left > 0 && src.len() as u64 >= block_left {
            self.core.update(&src[..block_left as usize]);
            src = &src[block_left as usize..];
            block_left = self.block_size;
            self.pos = 0;
            if self.count > 0 {
                let mut rec = [0u8; 20];
                self.core.get_block(&mut rec, 0);
                if let Some(bo) = self.block_out.as_mut() {
                    if self.ptr + 20 <= bo.len() {
                        bo[self.ptr..self.ptr + 20].copy_from_slice(&rec);
                    }
                }
                self.ptr += 20;
                self.count -= 1;
            } // else drop extra records, matching JS/shim behaviour
        }
        if !src.is_empty() {
            self.core.update(src);
        }
        self.pos += src.len() as u64;
        Ok(())
    }

    fn end(&mut self, out: &mut [u8]) -> Result<Box<[u8]>> {
        if self.ended {
            return Err(BackendError::new("Process already ended"));
        }
        if self.block_size == 0 {
            return Err(BackendError::new("Slice size must be positive"));
        }
        if out.len() < 16 {
            return Err(BackendError::new("Buffer must be at least 16 bytes long"));
        }
        if self.count > 0 {
            let mut rec = [0u8; 20];
            self.core.get_block(&mut rec, self.block_size - self.pos);
            if let Some(bo) = self.block_out.as_mut() {
                if self.ptr + 20 <= bo.len() {
                    bo[self.ptr..self.ptr + 20].copy_from_slice(&rec);
                }
            }
        }
        if self.core.file_external {
            let d = self
                .ext_md5
                .take()
                .ok_or_else(|| BackendError::new("file MD5 not supplied by reader"))?;
            out[..16].copy_from_slice(&d);
        } else {
            self.core.end(out);
        }
        self.ended = true;
        Ok(self.block_out.take().unwrap_or_default())
    }

    fn set_file_md5(&mut self, md5: &[u8; 16]) {
        self.ext_md5 = Some(*md5);
    }

    fn set_block_states(&mut self, states: &[[u32; 4]]) {
        self.core.states.extend(states.iter().copied());
    }

    fn reset(&mut self) -> Result<()> {
        if self.ended {
            return Err(BackendError::new("Process already ended"));
        }
        self.core.reset();
        Ok(())
    }
}

/// MD5Multi equivalent: N independent MD5 streams over equal-length regions.
/// Regions are hashed in lanes of up to `unit` (8 = AVX2, 4 = SSE2) through
/// Md5Mb; leftover regions through scalar Md5Ctx.
enum MbGroup {
    Mb8(md5mb::Md5Mb<8>),
    Mb4(md5mb::Md5Mb<4>),
}

pub struct NativeHasherOutput {
    unit: usize,
    groups: Vec<MbGroup>,
    rest: Vec<Md5Ctx>,
    started: Vec<bool>,
    regions: usize,
}

impl NativeHasherOutput {
    pub fn new(regions: u32) -> Self {
        let n = regions as usize;
        // unit is the multi-buffer lane count when an SIMD hasher exists
        // (4 or 8); on other arches everything hashes through scalar `rest`
        let unit = md5mb::max_lanes().min(8);
        let unit = if unit >= 4 { unit } else { 0 };
        let (unit, ngroups) = if unit == 0 { (1, 0) } else { (unit, n / unit) };
        let groups = (0..ngroups)
            .map(|_| {
                if unit >= 8 {
                    MbGroup::Mb8(md5mb::Md5Mb::new())
                } else {
                    MbGroup::Mb4(md5mb::Md5Mb::new())
                }
            })
            .collect();
        NativeHasherOutput {
            unit,
            groups,
            rest: (0..n - ngroups * unit).map(|_| Md5Ctx::new()).collect(),
            started: vec![false; n],
            regions: n,
        }
    }
}

impl HasherOutput for NativeHasherOutput {
    fn update(&mut self, bufs: &[&[u8]]) -> Result<()> {
        if bufs.len() != self.regions {
            return Err(BackendError::new("Region count mismatch"));
        }
        let len = bufs.first().map(|b| b.len()).unwrap_or(0);
        if bufs.iter().any(|b| b.len() != len) {
            return Err(BackendError::new("Regions must be equal length"));
        }
        if len == 0 {
            return Ok(());
        }
        for s in self.started.iter_mut() {
            *s = true;
        }
        let unit = self.unit;
        for (g, grp) in self.groups.iter_mut().enumerate() {
            let base = g * unit;
            match grp {
                MbGroup::Mb8(m) => m.update(std::array::from_fn(|i| bufs[base + i])),
                MbGroup::Mb4(m) => m.update(std::array::from_fn(|i| bufs[base + i])),
            }
        }
        for (i, r) in self.rest.iter_mut().enumerate() {
            r.update(bufs[self.groups.len() * unit + i]);
        }
        Ok(())
    }

    fn get(&mut self, out: &mut [u8]) -> Result<()> {
        if out.len() < self.regions * 16 {
            return Err(BackendError::new("Output buffer too small"));
        }
        let unit = self.unit;
        let empty = Md5Ctx::new().finish();
        for (g, grp) in self.groups.iter_mut().enumerate() {
            let base = g * unit;
            match grp {
                MbGroup::Mb8(m) => {
                    let mut dg = [[0u8; 16]; 8];
                    m.finish(&mut dg);
                    for (l, d) in dg.iter().enumerate() {
                        let i = base + l;
                        let d = if self.started[i] { *d } else { empty };
                        out[i * 16..i * 16 + 16].copy_from_slice(&d);
                    }
                }
                MbGroup::Mb4(m) => {
                    let mut dg = [[0u8; 16]; 4];
                    m.finish(&mut dg);
                    for (l, d) in dg.iter().enumerate() {
                        let i = base + l;
                        let d = if self.started[i] { *d } else { empty };
                        out[i * 16..i * 16 + 16].copy_from_slice(&d);
                    }
                }
            }
        }
        for (i, r) in self.rest.iter_mut().enumerate() {
            let idx = self.groups.len() * unit + i;
            let d = if self.started[idx] { r.finish() } else { empty };
            out[idx * 16..idx * 16 + 16].copy_from_slice(&d);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        *self = Self::new(self.regions as u32);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// file MD5 through the slice-boundary path must equal plain MD5 even
    /// when end() sees tmp_len >= 64 (file tail spans past a slice edge)
    #[test]
    fn end_file_md5_with_pending_tail() {
        for &(slice, len) in &[
            (64u64, 100usize),
            (128, 300),
            (96, 250),
            (64, 5000),
            (68, 400),
            (100, 700),
            (65, 200),
            (192, 900),
            (124, 500),
            (124, 624),
            (124, 130),
            (60, 300),
        ] {
            let mut h = NativeHasherInput::new(slice, vec![0u8; 1024].into_boxed_slice());
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            h.update(&data).unwrap();
            let mut got = [0u8; 16];
            h.end(&mut got).unwrap();
            let mut ctx = crate::md5::Md5Ctx::new();
            ctx.update(&data);
            let want = ctx.finish();
            assert_eq!(got, want, "slice_size={slice} len={len}");
        }
    }

    /// slice_size=0 must fail fast rather than corrupting pos accounting
    /// (block_size - pos underflows after the first update)
    #[test]
    fn zero_slice_size_rejected() {
        let mut h = NativeHasherInput::new(0, vec![0u8; 64].into_boxed_slice());
        assert!(h.update(&[1, 2, 3]).is_err());
        let mut out = [0u8; 16];
        assert!(h.end(&mut out).is_err());
    }
}

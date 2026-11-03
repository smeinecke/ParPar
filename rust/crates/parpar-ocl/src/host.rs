//! Native OpenCL GF16 backend - port of `PAR2ProcOCL`
//! (gf16/controller_ocl.cpp + controller_ocl_init.cpp).
//!
//! One `OclBackend` handles a byte range of every slice, driven by the
//! composite proc. Input slices land in per-staging-area device buffers
//! (map -> copy+checksum -> unmap); each full batch writes its coefficient
//! table and enqueues one NDRange kernel. Staging areas are released by
//! kernel-completion events.

use crate::oclsource as src;
use crate::*;
use parpar_backend::composite::{CanAdd, RangeBackend};
use parpar_backend::{BackendError, GfInfo, GfOclInfo, Result};
use parpar_gf16::gf;

const GF16_POLY: u32 = 0x1100B;

/// GF16 methods usable with the embedded sources (Galois16OCLMethods order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OclMethod {
    Auto = 0,
    Lookup = 1,
    LookupHalf = 2,
    LookupNoCache = 3,
    LookupHalfNoCache = 4,
    LookupGrp2 = 5,
    LookupGrp2NoCache = 6,
    Shuffle = 7,
    Log = 8,
    LogSmall = 9,
    LogSmall2 = 10,
    LogTiny = 11,
    LogSmallLmem = 12,
    LogTinyLmem = 13,
    By2 = 14,
}

impl OclMethod {
    pub fn from_i32(v: i32) -> Option<OclMethod> {
        use OclMethod::*;
        Some(match v {
            0 => Auto,
            1 => Lookup,
            2 => LookupHalf,
            3 => LookupNoCache,
            4 => LookupHalfNoCache,
            5 => LookupGrp2,
            6 => LookupGrp2NoCache,
            7 => Shuffle,
            8 => Log,
            9 => LogSmall,
            10 => LogSmall2,
            11 => LogTiny,
            12 => LogSmallLmem,
            13 => LogTinyLmem,
            14 => By2,
            _ => return None,
        })
    }
    fn name(self) -> &'static str {
        use OclMethod::*;
        match self {
            Auto => "Auto",
            Lookup => "Lookup",
            LookupHalf => "Lookup Half",
            LookupNoCache => "Lookup (NoCache)",
            LookupHalfNoCache => "Lookup Half (NoCache)",
            LookupGrp2 => "Lookup Group2",
            LookupGrp2NoCache => "Lookup Group2 (NoCache)",
            Shuffle => "Shuffle",
            Log => "Log",
            LogSmall => "Log-SmallExp",
            LogSmall2 => "Log-Small",
            LogTiny => "Log-TinyExp",
            LogSmallLmem => "Log-SmallExp (Local)",
            LogTinyLmem => "Log-TinyExp (Local)",
            By2 => "ByTwo",
        }
    }
    /// `info()` table: (idealInBatch, idealIters, usesOutGrouping)
    fn params(self) -> (u32, u32, bool) {
        use OclMethod::*;
        match self {
            By2 => (16, 1, true),
            Shuffle => (4, 1, false),
            Log | LogSmall | LogSmall2 | LogTiny | LogSmallLmem | LogTinyLmem => (8, 1, true),
            LookupGrp2 => (4, 2, false),
            Lookup | LookupHalf => (4, 4, false),
            LookupNoCache | LookupHalfNoCache => (4, 8, true),
            LookupGrp2NoCache => (6, 2, true),
            Auto => (8, 1, true),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
#[allow(dead_code)] // Log/LogSeq kept for parity with the C++ coeffType enum
enum CoeffType {
    Normal,
    Log,
    LogSeq,
}

struct StagingArea {
    input: Option<Buffer>,
    coeffs: Option<Buffer>,
    proc_coeffs: Vec<u16>,
    event: Option<ClEvent>,
    active: bool,
    /// inputs actually submitted in this batch (for Progress reporting -
    /// C++ reports req->numInputs, not the batch capacity)
    submitted: usize,
}

/// "OpenCL x.y ..." -> major*1000 + minor
fn parse_ocl_version(ver: &str) -> i32 {
    let p = ver.as_bytes();
    let mut major = 0i32;
    let mut minor = 0i32;
    let mut cur = &mut major;
    for &c in p.iter().skip(7) {
        if c == b'.' {
            cur = &mut minor;
        } else if c.is_ascii_digit() {
            *cur = *cur * 10 + (c - b'0') as i32;
        } else {
            break;
        }
    }
    major * 1000 + minor
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn round_down_pow2(v: usize) -> usize {
    if v == 0 {
        return 0;
    }
    let bits = usize::BITS - v.leading_zeros();
    1usize << (bits - 1)
}

// ---------- gf16 checksum helpers (gf16_cksum_copy/_check ports) ----------

/// mul2 per u16 lane over a u64 word (gf16_multi_mul2 on a lane).
#[inline]
fn mul2x4(v: u64) -> u64 {
    const MASK: u64 = 0x0001_0001_0001_0001;
    (v.wrapping_mul(2) & !MASK) ^ ((v >> 15) & MASK).wrapping_mul(GF16_POLY as u64 & 0xffff)
}

/// gf16_checksum_exp_generic: multiply each u16 lane by coeff (Horner).
#[inline]
fn cksum_exp(c: u64, exp: u16) -> u64 {
    let mut coeff = exp;
    let mut res = (0u64).wrapping_sub((coeff >> 15) as u64) & c;
    for _ in 0..15 {
        res = mul2x4(res);
        coeff <<= 1;
        res ^= (0u64).wrapping_sub((coeff >> 15) as u64) & c;
    }
    res
}

fn cksum_block(ck: &mut [u64], data: &[u8]) {
    for (i, c) in ck.iter_mut().enumerate() {
        let mut w = [0u8; 8];
        let s = i * 8;
        let n = data.len().saturating_sub(s).min(8);
        if n > 0 {
            w[..n].copy_from_slice(&data[s..s + n]);
        }
        *c = mul2x4(*c) ^ u64::from_le_bytes(w);
    }
}

/// gf16_cksum_copy: dst[..slice] = src (zero-padded), dst[slice..+ck] = cksum.
fn cksum_copy(dst: &mut [u8], src: &[u8], slice_len: usize, ck_size: usize) {
    debug_assert!(src.len() <= slice_len);
    let lanes = ck_size / 8;
    let mut ck = vec![0u64; lanes];
    let mut pos = 0usize;
    while pos + ck_size <= src.len() {
        cksum_block(&mut ck, &src[pos..pos + ck_size]);
        dst[pos..pos + ck_size].copy_from_slice(&src[pos..pos + ck_size]);
        pos += ck_size;
    }
    let remaining = src.len() - pos;
    if remaining > 0 {
        cksum_block(&mut ck, &src[pos..]);
        dst[pos..src.len()].copy_from_slice(&src[pos..]);
    }
    if src.len() < slice_len {
        dst[src.len()..slice_len].fill(0);
        let done = src.len().div_ceil(ck_size) * ck_size;
        if done < slice_len {
            let blocks = ((slice_len - done).div_ceil(ck_size)) % 65535;
            let e = gf::gf16_exp(blocks as u16);
            for c in ck.iter_mut() {
                *c = cksum_exp(*c, e);
            }
        }
    }
    for (i, c) in ck.iter().enumerate() {
        dst[slice_len + i * 8..slice_len + i * 8 + 8].copy_from_slice(&c.to_le_bytes());
    }
}

/// gf16_grp2_finish: de-interleave paired outputs (WRITE_GRP2 methods).
/// `src` holds `2*len` interleaved data bytes then `ck_size` checksum bytes;
/// dst u16[k] = src u16[2k + (grp&1)]. Verifies the checksum over dst.
fn grp2_finish(dst: &mut [u8], src: &[u8], len: usize, grp: usize, ck_size: usize) -> bool {
    let lanes = ck_size / 8;
    let odd = (grp & 1) * 2;
    // checksum block is interleaved the same way as the data: de-interleave
    // it into a contiguous block first (ungrp2a/b_block in the C++)
    let mut ckblock = vec![0u8; ck_size];
    for j in 0..ck_size / 2 {
        let so = len * 2 + j * 4 + odd;
        ckblock[j * 2] = src[so];
        ckblock[j * 2 + 1] = src[so + 1];
    }
    let mut ck: Vec<u64> = ckblock
        .chunks_exact(8)
        .take(lanes)
        .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let blocks = (len.div_ceil(ck_size)) % 65535;
    let e = gf::gf16_exp((65535u32.wrapping_sub(blocks as u32)) as u16);
    for c in ck.iter_mut() {
        *c = cksum_exp(*c, e);
    }
    let mut pos = 0usize;
    while pos < len {
        let n = (len - pos).min(ck_size);
        let mut block = vec![0u8; n];
        for j in (0..n).step_by(2) {
            // dst u16 (pos+j)/2 <- src u16 index (pos+j) + grp&1
            let so = (pos + j) * 2 + odd;
            block[j] = src[so];
            if j + 1 < n {
                block[j + 1] = src[so + 1];
            }
        }
        dst[pos..pos + n].copy_from_slice(&block);
        cksum_block(&mut ck, &block);
        pos += n;
    }
    ck.iter().all(|&c| c == 0)
}

/// gf16_cksum_copy_check: dst[..len] = src[..len], verifies trailing cksum.
fn cksum_copy_check(dst: &mut [u8], src: &[u8], len: usize, ck_size: usize) -> bool {
    let lanes = ck_size / 8;
    let mut ck = vec![0u64; lanes];
    for (i, c) in ck.iter_mut().enumerate() {
        let s = len + i * 8;
        *c = u64::from_le_bytes(src[s..s + 8].try_into().unwrap());
    }
    let blocks = (len.div_ceil(ck_size)) % 65535;
    let e = gf::gf16_exp((65535u32.wrapping_sub(blocks as u32)) as u16);
    for c in ck.iter_mut() {
        *c = cksum_exp(*c, e);
    }
    let mut pos = 0usize;
    while pos + ck_size <= len {
        cksum_block(&mut ck, &src[pos..pos + ck_size]);
        dst[pos..pos + ck_size].copy_from_slice(&src[pos..pos + ck_size]);
        pos += ck_size;
    }
    if pos < len {
        cksum_block(&mut ck, &src[pos..len]);
        dst[pos..len].copy_from_slice(&src[pos..len]);
    }
    ck.iter().all(|&c| c == 0)
}

// ------------------------------------------------------------------------

pub struct OclBackend {
    #[allow(dead_code)]
    platform: ClPlatformId,
    device: Device,
    ctx: Context,
    queue: Queue,
    ocl_plat_version: i32,
    #[allow(dead_code)]
    ocl_dev_version: i32,
    device_info: OclDeviceInfo,
    #[allow(dead_code)]
    device_id: i32,

    // slice geometry
    slice_size: usize,
    slice_size_cksum: usize,
    cksum_size: usize,
    slice_size_aligned: usize,
    allocated_slice_size: usize,
    bytes_per_group: usize,

    // kernel parameters
    setup_method: OclMethod,
    target_batch: u32,
    target_min_batch: i32,
    target_iters: u32,
    target_grouping: u32,
    input_batch_size: usize,
    min_in_batch: usize,
    outputs_per_group: u32,
    outputs_per_thread: u32,
    wg_size: usize,
    process_range: [usize; 2],
    workgroup_range: [usize; 2],
    outputs_interleaved: u32,
    coeff_type: CoeffType,

    // kernels and buffers
    kernel_mul: Option<Kernel>,
    kernel_mul_add: Option<Kernel>,
    kernel_mul_last: Option<Kernel>,
    kernel_mul_add_last: Option<Kernel>,
    program: Option<Program>,
    staging: Vec<StagingArea>,
    buffer_output: Option<Buffer>,
    buffer_out_exp: Option<Buffer>,
    extra_buffers: Vec<Buffer>,
    output_exps: Vec<u16>,

    // runtime state
    cur_area: usize,
    cur_inputs: usize,
    active_count: usize,
    processing_add: bool,
    has_added: bool,
    /// latched on a mid-submit run_kernel failure: partial kernel
    /// launches may be in flight with no event recorded, so the area
    /// can't be safely recycled - fail all further work and let
    /// deinit/reset (which queue.finish()es first) clean up
    dead: bool,
    dev_name: String,
}

struct OclDeviceInfo {
    short_vec_size: u32,
    char_vec_size: u32,
    vendor_id: u32,
    max_wg: usize,
    local_size: usize,
    #[allow(dead_code)]
    const_size: usize,
    #[allow(dead_code)]
    local_is_global: bool,
    endian_little: bool,
}

impl OclBackend {
    /// platform/device are indices into enumerate() results.
    pub fn new(platform_id: i32, device_id: i32, staging_areas: usize) -> Result<OclBackend> {
        let plats = platforms();
        let plat = *plats
            .get(platform_id.max(0) as usize)
            .ok_or_else(|| BackendError::new("OpenCL platform not found"))?;
        let devs = plat.devices(CL_DEVICE_TYPE_ALL);
        let dev = *devs
            .get(device_id.max(0) as usize)
            .ok_or_else(|| BackendError::new("OpenCL device not found"))?;

        let info = unsafe {
            OclDeviceInfo {
                short_vec_size: dev_info_scalar(dev.0, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT),
                char_vec_size: dev_info_scalar(dev.0, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR),
                vendor_id: dev_info_scalar(dev.0, CL_DEVICE_VENDOR_ID),
                max_wg: dev_info_scalar(dev.0, CL_DEVICE_MAX_WORK_GROUP_SIZE),
                local_size: dev_info_scalar(dev.0, CL_DEVICE_LOCAL_MEM_SIZE),
                const_size: dev_info_scalar(dev.0, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE),
                local_is_global: dev_info_scalar::<ClDeviceLocalMemType>(
                    dev.0,
                    CL_DEVICE_LOCAL_MEM_TYPE,
                ) == CL_GLOBAL,
                endian_little: dev_info_scalar::<ClBool>(dev.0, CL_DEVICE_ENDIAN_LITTLE) != 0,
            }
        };
        if !info.endian_little {
            return Err(BackendError::new("OpenCL device is not little-endian"));
        }

        let ctx = Context::create(dev).map_err(|e| BackendError::new(e.to_string()))?;
        let queue = Queue::create(&ctx, dev).map_err(|e| BackendError::new(e.to_string()))?;
        let dev_name = dev.name();
        let plat_ver = parse_ocl_version(&unsafe { plat_info_string(plat.0, CL_PLATFORM_VERSION) });
        let dev_ver = parse_ocl_version(&unsafe { dev_info_string(dev.0, CL_DEVICE_VERSION) });

        Ok(OclBackend {
            platform: plat.0,
            device: dev,
            ctx,
            queue,
            ocl_plat_version: plat_ver,
            ocl_dev_version: dev_ver,
            device_info: info,
            device_id: device_id,
            slice_size: 0,
            slice_size_cksum: 0,
            cksum_size: 32,
            slice_size_aligned: 0,
            allocated_slice_size: 0,
            bytes_per_group: 0,
            setup_method: OclMethod::Auto,
            target_batch: 0,
            target_min_batch: 0,
            target_iters: 0,
            target_grouping: 0,
            input_batch_size: 0,
            min_in_batch: 0,
            outputs_per_group: 8,
            outputs_per_thread: 0,
            wg_size: 0,
            process_range: [0, 0],
            workgroup_range: [0, 0],
            outputs_interleaved: 1,
            coeff_type: CoeffType::Normal,
            kernel_mul: None,
            kernel_mul_add: None,
            kernel_mul_last: None,
            kernel_mul_add_last: None,
            program: None,
            staging: (0..staging_areas.max(1))
                .map(|_| StagingArea {
                    input: None,
                    coeffs: None,
                    proc_coeffs: Vec::new(),
                    event: None,
                    active: false,
                    submitted: 0,
                })
                .collect(),
            buffer_output: None,
            buffer_out_exp: None,
            extra_buffers: Vec::new(),
            output_exps: Vec::new(),
            cur_area: 0,
            cur_inputs: 0,
            active_count: 0,
            processing_add: false,
            dead: false,
            has_added: false,
            dev_name,
        })
    }

    /// `init`: resolve method + build kernels (PAR2ProcOCL::init +
    /// setRecoverySlices-driven first setup_kernels happens lazily here).
    /// `min_batch` mirrors setMinInputBatchSize (0 = use batch size).
    pub fn init(
        &mut self,
        method: i32,
        target_batch: u32,
        min_batch: i32,
        target_iters: u32,
        target_grouping: u32,
        cksum_size: usize,
    ) -> Result<()> {
        let m = OclMethod::from_i32(method).unwrap_or(OclMethod::Auto);
        self.setup_method = if m == OclMethod::Auto {
            OclMethod::Lookup
        } else {
            m
        };
        self.target_batch = target_batch;
        self.target_min_batch = min_batch;
        self.target_iters = target_iters;
        self.target_grouping = target_grouping;
        self.cksum_size = cksum_size.max(8);
        self.slice_size_cksum = self.slice_size + self.cksum_size;
        self.outputs_interleaved = 1;
        self.reset_state();
        Ok(())
    }

    fn reset_state(&mut self) {
        self.cur_inputs = 0;
        self.cur_area = 0;
        self.active_count = 0;
        for area in &mut self.staging {
            if let Some(ev) = area.event.take() {
                release_event(ev);
            }
            area.active = false;
        }
        self.processing_add = false;
        self.has_added = false;
        self.dead = false;
    }

    /// Port of setup_kernels for the Lookup family + By2. Log/Shuffle
    /// methods are not offered upstream (availableMethods excludes them).
    fn setup_kernels(&mut self, output_sequential: bool) -> Result<()> {
        let method = self.setup_method;
        let dev = &self.device_info;
        if self.output_exps.is_empty() {
            return Err(BackendError::new("Recovery slices not set"));
        }

        let mut info_short_vec = dev.short_vec_size;
        if info_short_vec < 4 {
            info_short_vec = 2;
        }
        if info_short_vec >= 8 && method == OclMethod::By2 {
            info_short_vec = 8;
        } else if info_short_vec >= 4 {
            info_short_vec = 4;
        }

        let mut wg_size = dev.max_wg;
        if dev.char_vec_size == 1 && dev.vendor_id == 0x10de && wg_size > 32 {
            wg_size /= info_short_vec as usize * 2;
        }
        if wg_size > 8192 {
            wg_size = 8192;
        }
        // a driver reporting 0 for CL_DEVICE_MAX_WORK_GROUP_SIZE would
        // divide-by-zero below
        if wg_size == 0 {
            wg_size = 1;
        }

        let device_local = dev.local_size.saturating_sub(128);
        let num_outputs = self.output_exps.len();
        let mut outputs_per_thread = num_outputs;
        let mut source = String::new();

        let (ideal_batch, ideal_iters, _uses_group) = method.params();
        let mut input_batch = if self.target_batch != 0 {
            self.target_batch as usize
        } else {
            ideal_batch as usize
        };
        let mut group_iters = ideal_iters as usize;
        let mut thread_word_size = info_short_vec as usize * 2;
        let mut size_per_wg = thread_word_size * wg_size;
        self.outputs_per_group = 8;
        self.coeff_type = CoeffType::Normal;

        let kernel_code: &str;
        let kernel_funcs: Option<&str>;
        let method_code: &str;
        let ocl_ver_arg: &str;

        use OclMethod::*;
        match method {
            By2 => {
                loop {
                    if input_batch * size_per_wg > device_local {
                        input_batch = device_local / size_per_wg;
                    }
                    if input_batch < 1 && device_local >= 2048 && wg_size > 256 {
                        wg_size = 256;
                        size_per_wg = thread_word_size * wg_size;
                        input_batch = 16;
                        continue;
                    }
                    break;
                }
                kernel_code = src::_OCL_KERNEL_NOLUT;
                kernel_funcs = Some(src::_OCL_KERNEL_NOLUT_FUNCS);
                method_code = src::_OCL_METHOD_BY2;
                ocl_ver_arg = "";
            }
            Log | LogSmall | LogSmall2 | LogTiny | LogSmallLmem | LogTinyLmem | Shuffle => {
                return Err(BackendError::new(format!(
                    "OpenCL method {} is not supported",
                    method.name()
                )));
            }
            _ => {
                // Lookup family
                if method == LookupGrp2 || method == LookupGrp2NoCache {
                    if method == LookupGrp2 {
                        outputs_per_thread += outputs_per_thread & 1;
                        self.outputs_interleaved = 2;
                    } else {
                        self.outputs_interleaved = 2;
                    }
                    source.push_str("#define WRITE_GRP2 1\n");
                    info_short_vec = 2;
                    thread_word_size = info_short_vec as usize * 2;
                    size_per_wg = thread_word_size * wg_size;
                }

                if matches!(
                    method,
                    LookupNoCache | LookupHalfNoCache | LookupGrp2NoCache
                ) {
                    let table_size = if method == LookupHalfNoCache {
                        512
                    } else {
                        1024
                    };
                    let tables = device_local / table_size;
                    input_batch = if tables >= 96 {
                        8
                    } else if tables >= 60 {
                        6
                    } else if tables >= 32 {
                        4
                    } else {
                        2
                    };
                    self.outputs_per_group = (tables / input_batch.max(1)) as u32;
                    if self.outputs_per_group > 16 {
                        self.outputs_per_group = 16;
                    }
                    if method == LookupGrp2NoCache && self.outputs_per_group > 1 {
                        self.outputs_per_group &= !1;
                        self.target_grouping &= !1;
                    }
                    if self.target_batch != 0 {
                        input_batch = self.target_batch as usize;
                    }
                    kernel_code = src::_OCL_KERNEL_LUT;
                    kernel_funcs = Some(src::_OCL_KERNEL_LUT_FUNCS);
                } else {
                    if self.target_batch != 0 {
                        input_batch = self.target_batch as usize;
                    }
                    let mut table_size = if method == LookupHalf { 512 } else { 1024 };
                    if method == LookupGrp2 {
                        table_size *= 2;
                    }
                    loop {
                        let avail = device_local.saturating_sub(input_batch * table_size);
                        group_iters = round_down_pow2(avail / (size_per_wg * input_batch));
                        if group_iters < 1 && device_local >= 8192 {
                            if wg_size > 256 {
                                wg_size = 256;
                                size_per_wg = thread_word_size * wg_size;
                                continue;
                            }
                            if input_batch > 2 {
                                input_batch = 2;
                                continue;
                            }
                        }
                        break;
                    }
                    kernel_code = src::_OCL_KERNEL_CACHELUT;
                    kernel_funcs = None;
                }
                method_code = if matches!(method, LookupHalf | LookupHalfNoCache) {
                    src::_OCL_METHOD_LL
                } else {
                    src::_OCL_METHOD_LH
                };
                ocl_ver_arg = "";

                // multiply-by-256 reduction table
                source.push_str("__constant ushort mul256poly[256] = {0");
                for i in 1..256u32 {
                    let mut n = (i << 8) as i32;
                    for _ in 0..8 {
                        n = (n << 1) ^ (((n >> 15) as i32).wrapping_neg() & GF16_POLY as i32);
                    }
                    source.push_str(&format!(",{}", n & 0xffff));
                }
                source.push_str("};\n");
            }
        }

        if input_batch < 1 {
            return Err(BackendError::new("OpenCL: failed to size input batch"));
        }
        self.input_batch_size = input_batch;
        self.min_in_batch = if self.target_min_batch > 0 {
            self.target_min_batch as usize
        } else {
            input_batch
        };

        if self.target_iters != 0 {
            group_iters = self.target_iters as usize;
        }
        if self.target_grouping != 0 {
            self.outputs_per_group = self.target_grouping;
        }
        if group_iters < 1 || self.outputs_per_group < 1 {
            return Err(BackendError::new(
                "OpenCL: failed to size kernel parameters",
            ));
        }

        // scale down for small slices
        if size_per_wg * group_iters > self.slice_size_cksum {
            if group_iters > 1 {
                group_iters = ceil_div(self.slice_size_cksum, size_per_wg).max(1);
            }
            let min_wg = if kernel_code == src::_OCL_KERNEL_LUT
                || kernel_code == src::_OCL_KERNEL_CACHELUT
            {
                128
            } else {
                1
            };
            if group_iters == 1 && size_per_wg > self.slice_size_cksum && wg_size > min_wg {
                wg_size = ceil_div(self.slice_size_cksum, thread_word_size);
                if wg_size < min_wg {
                    wg_size = min_wg;
                }
            }
        }
        wg_size = round_down_pow2(wg_size);
        let _ = device_local;
        let _ = output_sequential;
        size_per_wg = thread_word_size * wg_size;

        self.bytes_per_group = size_per_wg * group_iters;
        let slice_groups = ceil_div(self.slice_size_cksum, self.bytes_per_group);
        self.process_range = [
            slice_groups * wg_size,
            ceil_div(num_outputs, outputs_per_thread),
        ];
        self.slice_size_aligned = slice_groups * self.bytes_per_group;
        self.allocated_slice_size = self.slice_size_aligned;
        self.outputs_per_thread = outputs_per_thread as u32;
        self.wg_size = wg_size;
        self.workgroup_range = [wg_size, 1];
        let _ = source; // assembled below

        // ---- assemble program source ----
        let mut prog = source.clone();
        prog.push_str(src::_OCL_DEFINES);
        prog.push('\n');
        prog.push_str(method_code);
        prog.push('\n');

        let params = format!(
            "{} -DMAX_SLICE_SIZE={} -DNUM_OUTPUTS={} -DOUTPUTS_PER_THREAD={} -DOUTPUT_THREADS={} -DVECT_WIDTH={} -DCOL_GROUP_SIZE={} -DCOL_GROUP_ITERS={} -DOUTPUT_GROUPING={} -DSUBMIT_INPUTS={}",
            ocl_ver_arg,
            self.slice_size_aligned,
            num_outputs,
            outputs_per_thread,
            ceil_div(num_outputs, outputs_per_thread),
            info_short_vec,
            wg_size,
            group_iters,
            self.outputs_per_group,
            input_batch
        );

        // four kernel instantiations (controller_ocl_init.cpp emission order:
        // last -> [numInputs define] kernel -> [MUL_ONLY] first -> only)
        prog.push_str("#define KERNFN(f) gf16_kernel_last_##f\n");
        if let Some(f) = kernel_funcs {
            prog.push_str(f);
        }
        prog.push_str("void kernel __attribute__((reqd_work_group_size(COL_GROUP_SIZE, 1, 1))) \ngf16_ocl_kernel_last(__global val_t* restrict dst, __global const val_t* restrict src, __global const ushort* restrict coeff, const ushort numInputs  EX_TABLE_KARGS_DECL)\n");
        prog.push_str(kernel_code);
        prog.push('\n');

        prog.push_str("#undef KERNFN\n#define KERNFN(f) gf16_kernel_##f\n");
        prog.push_str(&format!("#define numInputs {}\n", input_batch));
        if let Some(f) = kernel_funcs {
            prog.push_str(f);
        }
        prog.push_str("void kernel __attribute__((reqd_work_group_size(COL_GROUP_SIZE, 1, 1))) \ngf16_ocl_kernel(__global val_t* restrict dst, __global const val_t* restrict src, __global const ushort* restrict coeff  EX_TABLE_KARGS_DECL)\n");
        prog.push_str(kernel_code);
        prog.push('\n');

        prog.push_str("#undef KERNFN\n#define KERNFN(f) gf16_kernel_first_##f\n#define MUL_ONLY\n");
        if let Some(f) = kernel_funcs {
            prog.push_str(f);
        }
        prog.push_str("void kernel __attribute__((reqd_work_group_size(COL_GROUP_SIZE, 1, 1))) \ngf16_ocl_kernel_first(__global val_t* restrict dst, __global const val_t* restrict src, __global const ushort* restrict coeff  EX_TABLE_KARGS_DECL)\n");
        prog.push_str(kernel_code);
        prog.push('\n');

        prog.push_str("#undef numInputs\n");
        prog.push_str("#undef KERNFN\n#define KERNFN(f) gf16_kernel_only_##f\n");
        if let Some(f) = kernel_funcs {
            prog.push_str(f);
        }
        prog.push_str("void kernel __attribute__((reqd_work_group_size(COL_GROUP_SIZE, 1, 1))) \ngf16_ocl_kernel_only(__global val_t* restrict dst, __global const val_t* restrict src, __global const ushort* restrict coeff, const ushort numInputs  EX_TABLE_KARGS_DECL)\n");
        prog.push_str(kernel_code);

        // compile
        let program = Program::build_opts(&self.ctx, self.device, &prog, &params)
            .map_err(|e| BackendError::new(e.to_string()))?;

        let kmul = Kernel::create(&program, "gf16_ocl_kernel_first")
            .map_err(|e| BackendError::new(e.to_string()))?;
        let kmul_last = Kernel::create(&program, "gf16_ocl_kernel_only")
            .map_err(|e| BackendError::new(e.to_string()))?;
        let kmadd_last = Kernel::create(&program, "gf16_ocl_kernel_last")
            .map_err(|e| BackendError::new(e.to_string()))?;
        let kmadd = Kernel::create(&program, "gf16_ocl_kernel")
            .map_err(|e| BackendError::new(e.to_string()))?;

        // buffers
        let mut buf_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        if self.ocl_plat_version >= 1002 {
            buf_flags |= CL_MEM_HOST_WRITE_ONLY;
        }
        for area in &mut self.staging {
            area.input = Some(
                Buffer::create(&self.ctx, buf_flags, input_batch * self.slice_size_aligned)
                    .map_err(|e| BackendError::new(e.to_string()))?,
            );
            let coeff_len = input_batch * num_outputs;
            area.coeffs = Some(
                Buffer::create(&self.ctx, buf_flags, coeff_len * 2)
                    .map_err(|e| BackendError::new(e.to_string()))?,
            );
            area.proc_coeffs = vec![0u16; coeff_len];
        }
        let out_flags = CL_MEM_READ_WRITE
            | if self.ocl_plat_version >= 1002 {
                CL_MEM_HOST_READ_ONLY
            } else {
                0
            };
        self.buffer_output = Some(
            Buffer::create(&self.ctx, out_flags, num_outputs * self.slice_size_aligned)
                .map_err(|e| BackendError::new(e.to_string()))?,
        );

        // arg 0 = output on all four kernels
        let out = self.buffer_output.as_ref().unwrap();
        kmul.set_mem(0, out).map_err(err_map)?;
        kmul_last.set_mem(0, out).map_err(err_map)?;
        kmadd_last.set_mem(0, out).map_err(err_map)?;
        kmadd.set_mem(0, out).map_err(err_map)?;

        self.buffer_out_exp = None;
        self.extra_buffers.clear();

        // dummy run to surface execute-time errors early
        kmul.set_mem(1, self.staging[0].input.as_ref().unwrap())
            .map_err(err_map)?;
        kmul.set_mem(2, self.staging[0].coeffs.as_ref().unwrap())
            .map_err(err_map)?;
        self.queue
            .ndrange(&kmul, &self.workgroup_range, &self.workgroup_range)
            .map_err(|e| BackendError::new(e.to_string()))?;

        self.kernel_mul = Some(kmul);
        self.kernel_mul_add = Some(kmadd);
        self.kernel_mul_last = Some(kmul_last);
        self.kernel_mul_add_last = Some(kmadd_last);
        self.program = Some(program);
        Ok(())
    }

    fn set_coeffs_num(&mut self, area: usize, idx: usize, input_num: u16) {
        let input_log = gf::gfmat_input_log(input_num);
        let batch = self.input_batch_size;
        let coeffs = &mut self.staging[area].proc_coeffs;
        for (i, &e) in self.output_exps.iter().enumerate() {
            coeffs[idx + i * batch] = gf::gfmat_coeff_from_log(input_log, e);
        }
    }

    fn run_kernel(&mut self, buf: usize, num_inputs: usize) -> Result<()> {
        // write coeffs - blocking write required: coeff_bytes is a
        // temporary Vec; a non-blocking enqueue would let the driver read
        // freed host memory (C++ writes from persistent procCoeffs)
        let coeff_bytes: Vec<u8> = self.staging[buf]
            .proc_coeffs
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let coeffs_buf = self.staging[buf]
            .coeffs
            .as_ref()
            .ok_or_else(|| BackendError::new("OpenCL backend not initialised"))?;
        self.queue
            .write(coeffs_buf, 0, &coeff_bytes)
            .map_err(|e| BackendError::new(e.to_string()))?;

        let partial = num_inputs != self.input_batch_size;
        let kernel_idx = match (self.processing_add, partial) {
            (false, false) => 0, // mul
            (true, false) => 1,  // mulAdd
            (false, true) => 2,  // mulLast
            (true, true) => 3,   // mulAddLast
        };
        {
            let k = [
                &self.kernel_mul,
                &self.kernel_mul_add,
                &self.kernel_mul_last,
                &self.kernel_mul_add_last,
            ][kernel_idx]
                .as_ref()
                .ok_or_else(|| BackendError::new("OpenCL backend not initialised"))?;
            if partial {
                k.set_u16(3, num_inputs as u16).map_err(err_map)?;
            }
            let in_buf = self.staging[buf]
                .input
                .as_ref()
                .ok_or_else(|| BackendError::new("OpenCL backend not initialised"))?;
            k.set_mem(1, in_buf).map_err(err_map)?;
            k.set_mem(2, coeffs_buf).map_err(err_map)?;
            let ev = self
                .queue
                .ndrange_ev(k, &self.process_range, &self.workgroup_range, &[])
                .map_err(|e| BackendError::new(e.to_string()))?;
            // the active flag guarantees the old event was already drained;
            // release defensively rather than overwrite-leak
            if let Some(old) = self.staging[buf].event.replace(ev) {
                release_event(old);
            }
        }
        self.processing_add = true;
        self.queue
            .flush()
            .map_err(|e| BackendError::new(e.to_string()))?;
        Ok(())
    }

    /// Poll kernel events; release completed staging areas.
    fn drain_areas(&mut self) -> usize {
        if self.dead {
            // an orphaned area may have untracked work in flight -
            // hold everything until reset_state (after queue.finish)
            return 0;
        }
        let mut freed = 0usize;
        for area in &mut self.staging {
            if !area.active {
                continue;
            }
            let done = match area.event {
                // negative status = command errored (still "done"); a failed
                // query (None) means we don't know - keep the area active
                Some(ev) => matches!(event_status(ev), Some(s) if s <= CL_COMPLETE),
                None => true,
            };
            if done {
                if let Some(ev) = area.event.take() {
                    release_event(ev);
                }
                area.active = false;
                self.active_count = self.active_count.saturating_sub(1);
                // C++ reports the actual submitted count, not batch size
                freed += area.submitted;
            }
        }
        freed
    }

    /// Whether the selected device supports the backend's methods.
    pub fn device_name(&self) -> &str {
        &self.dev_name
    }
    pub fn method_name(&self) -> &'static str {
        self.setup_method.name()
    }
}

fn err_map(e: Error) -> BackendError {
    BackendError::new(e.to_string())
}

// OpenCL object handles are movable between threads (the C++ backend uses
// them across async staging threads); all calls are serialized through the
// owning proc's &mut access.
unsafe impl Send for OclBackend {}

impl RangeBackend for OclBackend {
    fn set_slice_size(&mut self, size: usize) {
        self.slice_size = size;
        if self.slice_size_cksum != 0 {
            self.slice_size_cksum = size + self.cksum_size;
        }
    }

    fn set_current_slice_size(&mut self, size: usize) -> Result<()> {
        self.slice_size = size;
        self.slice_size_cksum = size + self.cksum_size;
        if self.slice_size_cksum > self.allocated_slice_size && self.allocated_slice_size > 0 {
            let exps = self.output_exps.clone();
            self.setup_kernels(false)?;
            if !exps.is_empty() {
                self.output_exps = exps;
            }
            return Ok(());
        }
        if self.bytes_per_group > 0 {
            let groups = self.slice_size_cksum.div_ceil(self.bytes_per_group);
            self.process_range[0] = groups * self.wg_size;
            self.slice_size_aligned = groups * self.bytes_per_group;
        }
        Ok(())
    }

    fn set_recovery_slices(&mut self, exps: Option<&[u16]>, count: usize) -> Result<()> {
        if count == 0 {
            self.output_exps.clear();
            return Ok(());
        }
        let need_reinit = count != self.output_exps.len();
        self.output_exps = match exps {
            Some(e) => {
                let mut v = e.to_vec();
                // C++ assumes >= count exps; the CPU proc zero-pads the
                // shortfall - do the same so index bounds agree
                v.resize(count, 1);
                v
            }
            None => vec![1u16; count],
        };
        if need_reinit || self.program.is_none() {
            let seq = exps.map(|e| e.windows(2).all(|w| w[1] == w[0].wrapping_add(1)));
            self.setup_kernels(seq.unwrap_or(false))?;
        }
        Ok(())
    }

    fn can_add(&mut self) -> CanAdd {
        self.drain_areas();
        if self.staging[self.cur_area].active {
            return CanAdd::Full;
        }
        if self.active_count < self.staging.len().saturating_sub(1) {
            CanAdd::Ok
        } else {
            CanAdd::OkBusy
        }
    }

    fn add_input(&mut self, data: &[u8], input_num: u16, flush: bool) -> Result<()> {
        if self.staging.is_empty() {
            return Err(BackendError::new("Recovery slices not set"));
        }
        if data.len() > self.slice_size {
            return Err(BackendError::new("Input larger than slice size"));
        }
        // a failed setup_kernels leaves staging uninitialised - the
        // proc_coeffs write + buffer unwraps below would panic
        if self.staging[self.cur_area.min(self.staging.len() - 1)]
            .input
            .is_none()
        {
            return Err(BackendError::new("OpenCL backend not initialised"));
        }
        if self.cur_area >= self.staging.len() {
            self.cur_area = 0;
        }
        if self.staging[self.cur_area].active {
            return Err(BackendError::new("OpenCL staging area busy"));
        }
        if self.dead {
            return Err(BackendError::new("OpenCL backend failed"));
        }
        self.set_coeffs_num(self.cur_area, self.cur_inputs, input_num);

        // map -> cksum_copy -> unmap into the staging input slot
        let off = self.cur_inputs * self.slice_size_aligned;
        let flags = if self.ocl_plat_version >= 1002 {
            CL_MAP_WRITE_INVALIDATE_REGION
        } else {
            CL_MAP_WRITE
        };
        let buf = self.staging[self.cur_area].input.as_ref().unwrap();
        let p = self
            .queue
            .map(buf, flags, off, self.slice_size_aligned)
            .map_err(|e| BackendError::new(e.to_string()))?;
        unsafe {
            let dst = std::slice::from_raw_parts_mut(p as *mut u8, self.slice_size_aligned);
            cksum_copy(dst, data, self.slice_size, self.cksum_size);
        }
        self.queue
            .unmap(buf, p)
            .map_err(|e| BackendError::new(e.to_string()))?;

        self.cur_inputs += 1;
        self.has_added = true;
        let submit = flush
            || self.cur_inputs == self.input_batch_size
            || (self.active_count == 0
                && self.staging.len() > 1
                && self.cur_inputs >= self.min_in_batch);
        if submit {
            let n = self.cur_inputs;
            self.staging[self.cur_area].active = true;
            self.staging[self.cur_area].submitted = n;
            self.active_count += 1;
            let area = self.cur_area;
            self.cur_inputs = 0;
            self.cur_area = (self.cur_area + 1) % self.staging.len();
            if let Err(e) = self.run_kernel(area, n) {
                // partial launches may be in flight with no event to
                // track - latch dead so the area is never recycled
                self.dead = true;
                return Err(e);
            }
        }
        Ok(())
    }

    fn end_input(&mut self) -> Result<()> {
        if self.dead {
            return Err(BackendError::new("OpenCL backend failed"));
        }
        // flush partial batch
        if self.cur_inputs > 0 {
            let n = self.cur_inputs;
            self.staging[self.cur_area].active = true;
            self.staging[self.cur_area].submitted = n;
            self.active_count += 1;
            let area = self.cur_area;
            self.cur_inputs = 0;
            self.cur_area = (self.cur_area + 1) % self.staging.len();
            if let Err(e) = self.run_kernel(area, n) {
                self.dead = true;
                return Err(e);
            }
        }
        // wait for all staging events; release every handle even if a wait
        // fails, then surface the first error
        let mut wait_err: Result<()> = Ok(());
        for area in &mut self.staging {
            if let Some(ev) = area.event.take() {
                if let Err(e) = Queue::wait_events(&[ev]) {
                    if wait_err.is_ok() {
                        wait_err = Err(BackendError::new(e.to_string()));
                    }
                }
                release_event(ev);
            }
            area.active = false;
        }
        self.active_count = 0;
        wait_err
    }

    fn get_output(&mut self, index: usize, out: &mut [u8]) -> Result<bool> {
        if index >= self.output_exps.len() {
            return Err(BackendError::new("Invalid output index"));
        }
        if out.len() < self.slice_size {
            return Err(BackendError::new("Output buffer too small"));
        }
        let buf = self
            .buffer_output
            .as_ref()
            .ok_or_else(|| BackendError::new("Output not initialised"))?;
        let mut off = index * self.slice_size_aligned;
        let mut grp = 1usize;
        if self.outputs_interleaved > 1
            && index < (self.output_exps.len() & !(self.outputs_interleaved as usize - 1))
        {
            grp = self.outputs_interleaved as usize;
            off = (index & !(self.outputs_interleaved as usize - 1)) * self.slice_size_aligned;
        }
        let p = self
            .queue
            .map(buf, CL_MAP_READ, off, self.slice_size_aligned * grp)
            .map_err(|e| BackendError::new(e.to_string()))?;
        let valid = unsafe {
            let src = std::slice::from_raw_parts(p as *const u8, self.slice_size_aligned * grp);
            if grp == 2 {
                grp2_finish(
                    out,
                    src,
                    self.slice_size,
                    index & (grp - 1),
                    self.cksum_size,
                )
            } else {
                cksum_copy_check(out, src, self.slice_size, self.cksum_size)
            }
        };
        self.queue
            .unmap(buf, p)
            .map_err(|e| BackendError::new(e.to_string()))?;
        Ok(valid)
    }

    fn has_added(&self) -> bool {
        self.has_added
    }

    fn discard_output(&mut self) {
        self.processing_add = false;
        self.has_added = false;
    }

    fn deinit(&mut self) -> Result<()> {
        let _ = self.queue.finish();
        self.reset_state();
        Ok(())
    }

    fn drain_progress(&mut self) -> Result<i64> {
        Ok(self.drain_areas() as i64)
    }

    fn info(&mut self) -> GfInfo {
        GfInfo {
            cpu: None,
            ocl: vec![GfOclInfo {
                device_name: self.dev_name.clone(),
                method_desc: self.setup_method.name().to_string(),
                staging_count: self.staging.len() as u32,
                staging_size: self.input_batch_size as u64,
                chunk_size: self.bytes_per_group as u64,
                output_chunks: self.outputs_per_group,
                slice_mem: self.slice_size_aligned as u64,
                num_output_slices: self.output_exps.len() as i32,
            }],
        }
    }
}

// GF coefficient math comes from parpar_gf16::gf (gfmat_input_log /
// gfmat_coeff_from_log / gf16_exp) - identical tables to the C++ code.

impl Drop for OclBackend {
    fn drop(&mut self) {
        // any staging event left at drop still needs releasing
        for a in &mut self.staging {
            if let Some(ev) = a.event.take() {
                release_event(ev);
            }
        }
    }
}

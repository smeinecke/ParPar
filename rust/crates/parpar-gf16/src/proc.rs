//! Native GfProc: CPU implementation of the GF(2^16) multiply-accumulate
//! pipeline with a background worker thread, mirroring PAR2ProcCPU's
//! processQueue - hashing/IO on the pipeline thread overlaps with the GF
//! multiply on the worker.
//!
//! Semantics mirror PAR2ProcCPU + the shim's proc wrapper:
//! - `add_input` packs the input on the caller thread (like the C++
//!   prepare_packed in addInput) then hands it to the worker channel;
//!   it returns `AddResult::Full` when `staging_limit*batchsize` inputs
//!   are in flight (the C++ staging-area bound)
//! - the worker accumulates inputs into batches of `input_batchsize`; a
//!   batch flushes when full or on `end_input`; `InputDone` fires per
//!   input when its batch flush completes (staging freed)
//! - each input is zero-padded to `current_slice_size` and, for every
//!   recovery output j, XOR-accumulated with coefficient
//!   `gfmat_coeff(input_index, exponents[j])`
//! - the first flush after `set_recovery_slices` overwrites outputs
//!   (processing_add=false); later flushes accumulate
//! - `get_output` returns `current_slice_size` bytes plus a corruption
//!   self-check flag, processed in channel order after `end_input`

use crate::gf::{gfmat_coeff_from_log, gfmat_input_log};
use crate::kernel::{packed_len, pick_explicit, GfKernel};
use parpar_backend::{
    AddResult, BackendError, CpuSpec, GfCpuInfo, GfInfo, GfProc, ProcEvent, Result, SliceBuf,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub type EventQueues = Arc<Mutex<HashMap<usize, VecDeque<ProcEvent>>>>;

struct StagedInput {
    buf: SliceBuf,
    input_index: i64,
    token: u64,
    coeffs: Vec<u16>, // one coefficient per recovery output
    packed: Vec<u8>,  // kernel-packed form, filled by the caller thread
}

/// GF(2^16) lane checksum (gf16_checksum_generic equivalent): each u16 lane
/// of a u64 word evolves as c = c*2 ^ data.
#[inline(always)]
fn mul2x4(v: u64) -> u64 {
    const MASK: u64 = 0x0001_0001_0001_0001;
    (v.wrapping_mul(2) & !MASK) ^ ((v >> 15) & MASK).wrapping_mul(0x100B)
}

fn cksum_update(mut c: u64, data: &[u8]) -> u64 {
    let mut chunks = data.chunks_exact(8);
    for w in &mut chunks {
        c = mul2x4(c) ^ u64::from_le_bytes(w.try_into().unwrap());
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rem.len()].copy_from_slice(rem);
        c = mul2x4(c) ^ u64::from_le_bytes(buf);
    }
    c
}

/// Caller/worker shared configuration: caller methods validate against
/// atomics while the worker owns the compute state. Config mutations go
/// through the command channel so the worker sees them in FIFO order.
struct Shared {
    exponents_set: AtomicBool,
    outputs_count: AtomicUsize,
    current_slice_size: AtomicU64,
    /// inputs handed to the worker but not yet flushed (staging bound)
    pending: AtomicUsize,
    num_threads: AtomicI32,
    progress_on: AtomicBool,
    closed: AtomicBool,
    /// worker exited - sends fail fast instead of queueing forever
    worker_dead: AtomicBool,
    /// recycled per-slice packed staging buffers: the worker returns them
    /// when a batch flushes so add_input avoids a 4MB alloc+memset per
    /// slice (C++ keeps fixed staging slots instead)
    packed_pool: Mutex<Vec<Vec<u8>>>,
}

enum Cmd {
    Input(StagedInput),
    End {
        token: u64,
    },
    Output {
        index: usize,
        out: SliceBuf,
        token: u64,
    },
    SyncRecovery {
        exponents: Vec<u16>,
        count: usize,
    },
    SliceSize {
        alloc: u64,
        current: u64,
    },
    FreeMem,
    Close,
}

/// Worker-owned compute state (single-threaded access inside the worker).
struct Worker {
    exponents: Vec<u16>,
    /// accumulated outputs in the C++ tile-interleaved layout
    /// `[round][output][tile]` - every round's output tiles sit
    /// contiguous so a tile-major multiply sweeps sequential addresses
    /// instead of jumping between per-output regions
    outputs_flat: Vec<u8>,
    /// tile granularity + packed length the flat layout was built for
    out_tile: usize,
    layout_len: usize,
    output_cksum: Vec<u64>,
    processing_add: bool,
    batch: Vec<StagedInput>,
    /// block-interleaved staging rebuilt per flush for packed kernels
    /// (C++ prepare_packed INTERLEAVE_INPUTS layout), split into one
    /// allocation per worker thread's round range - separate address
    /// regions spread L3 home nodes so multiply reads don't funnel
    /// through a single buffer's homes (measured ~40% aggregate win on
    /// shared-line-heavy cores); reused across batches
    packed_stages: Vec<Vec<u8>>,
    /// round -> packed_stages index, and the round's local index inside
    /// that buffer (rounds are contiguous per buffer)
    packed_round_buf: Vec<usize>,
    packed_round_local: Vec<usize>,
    needs_discard: bool,
    stat_batches: u32,
    slice_size: u64,
    current_slice_size: u64,
    input_batchsize: u32,
    chunk_size: u32,
}

/// Multiply-loop tile resolution (also `GfInfo.chunk_size`): user
/// loopTileSize wins, else the kernel's idealChunkSize floored at 64KiB -
/// per-task dispatch amortises over tile bytes and a 64KiB tile keeps the
/// dst tile plus the batch's input window L2-resident.
fn resolve_tile_len(kernel: &dyn GfKernel, chunk_size: u32, len: usize) -> usize {
    let block = kernel.block_len().max(1);
    let t = if chunk_size > 0 {
        chunk_size as usize
    } else {
        (crate::kernel::method_info(kernel.id()).target_chunk as usize).max(64 * 1024)
    };
    t.max(block).min(len.max(1)).div_ceil(block) * block
}

impl Worker {
    fn aligned_size(&self, kernel: &dyn GfKernel) -> usize {
        packed_len(kernel, self.current_slice_size as usize)
    }

    /// Contiguous copy of output `idx`'s accumulated packed data
    /// (`layout_len` bytes) out of the interleaved layout - used by the
    /// Output command and the output checksum.
    fn gather_output(&self, idx: usize) -> Vec<u8> {
        let len = self.layout_len;
        let tile = self.out_tile.max(1);
        let nouts = self.exponents.len().max(1);
        let mut tmp = vec![0u8; len];
        for r in 0..len.div_ceil(tile) {
            let tb = (len - r * tile).min(tile);
            let off = (r * nouts + idx) * tile;
            tmp[r * tile..r * tile + tb].copy_from_slice(&self.outputs_flat[off..off + tb]);
        }
        tmp
    }

    /// Rebuild `outputs_flat` for `nouts` outputs over `len` packed bytes,
    /// preserving existing contributions when only the packed length grew
    /// (same tile granularity => same in-row offsets, so a prefix copy
    /// keeps every stored round).
    fn rebuild_outputs(&mut self, kernel: &dyn GfKernel, len: usize, nouts: usize) {
        let tile = resolve_tile_len(kernel, self.chunk_size, len);
        let nrounds = len.div_ceil(tile);
        let need = nrounds * nouts * tile;
        if tile == self.out_tile && len >= self.layout_len {
            let mut new_flat = vec![0u8; need];
            let keep = self.outputs_flat.len().min(new_flat.len());
            new_flat[..keep].copy_from_slice(&self.outputs_flat[..keep]);
            self.outputs_flat = new_flat;
        } else {
            let old = std::mem::take(&mut self.outputs_flat);
            let mut new_flat = vec![0u8; need];
            if !old.is_empty() && self.out_tile > 0 && self.layout_len > 0 {
                // migrate output by output: byte b of output o sits at
                // (b/tile*nouts + o)*tile + b%tile in each layout, so the
                // byte prefix copies through any tile/len/nouts change
                let old_tile = self.out_tile;
                let old_len = self.layout_len;
                let old_nouts = self.exponents.len().max(1);
                let keep = len.min(old_len);
                for o in 0..old_nouts.min(nouts) {
                    let mut b = 0usize;
                    while b < keep {
                        let n = (tile - b % tile)
                            .min(old_tile - b % old_tile)
                            .min(keep - b);
                        let so = (b / old_tile * old_nouts + o) * old_tile + b % old_tile;
                        let d = (b / tile * nouts + o) * tile + b % tile;
                        new_flat[d..d + n].copy_from_slice(&old[so..so + n]);
                        b += n;
                    }
                }
            }
            self.outputs_flat = new_flat;
        }
        self.out_tile = tile;
        self.layout_len = len;
    }

    /// discardOutput(): after an Ended event, the next add/end resets
    /// output accumulation (matches the shim's pendingDiscardOutput logic -
    /// chunk passes produce independent output chunks, not accumulation).
    fn maybe_discard(&mut self) {
        if self.needs_discard {
            self.processing_add = false;
            self.needs_discard = false;
        }
    }

    /// Inputs per flush for a given output count - shared by the worker's
    /// flush trigger and the caller-side staging cap. `aligned` bounds the
    /// staged memory footprint: the group never grows past ~256MiB of
    /// staging per configured staging area.
    fn effective_batch_for(nouts: usize, base: usize, aligned: usize) -> usize {
        let base = base.max(1);
        if nouts <= base {
            return base;
        }
        let optimal = ((2 * nouts) as f64).sqrt();
        let mut group = (optimal / base as f64).round().clamp(1.0, 4.0) as usize;
        // bound staged bytes: staging_limit areas are added by the caller
        let mem_cap = (128 * 1024 * 1024 / aligned.max(1) / base.max(1)).max(1);
        group = group.min(mem_cap);
        base * group
    }

    /// Inputs per flush. The staged batch size is a pure scheduling choice
    /// (output is identical either way): each flush sweeps every output
    /// region once, so the multiply's dst traffic scales as
    /// nouts*aligned/batch while src work stays fixed - the minimum of
    /// batch + 2*nouts/batch sits near batch = sqrt(2*nouts). Capped at 4x
    /// the configured size so staging memory stays bounded and tail
    /// flushes don't wait on oversized batches.
    fn effective_batch(&self, kernel: &dyn GfKernel) -> usize {
        Self::effective_batch_for(
            self.exponents.len(),
            self.input_batchsize as usize,
            self.aligned_size(kernel),
        )
    }

    fn flush_batch(
        &mut self,
        kernel: &Arc<dyn GfKernel>,
        shared: &Shared,
        emit: &dyn Fn(ProcEvent),
        tail: bool,
    ) {
        if self.batch.is_empty() {
            return;
        }
        self.stat_batches += 1;
        let len = self.aligned_size(&**kernel);
        let mut staged = std::mem::take(&mut self.batch);
        // zero-length slice region: nothing to multiply - release staging
        if len == 0 {
            let n = staged.len();
            for inp in staged {
                emit(ProcEvent::InputDone {
                    token: inp.token,
                    input_index: inp.input_index,
                    buf: inp.buf,
                });
            }
            shared.pending.fetch_sub(n, Ordering::Release);
            return;
        }
        // the flat layout must cover (rounds x outputs x tile) for the
        // current packed length; SyncRecovery/SliceSize keep it in step,
        // but FreeMem or an exponents bump can leave it short
        let need = if self.out_tile > 0 {
            len.div_ceil(self.out_tile) * self.exponents.len() * self.out_tile
        } else {
            usize::MAX
        };
        if self.layout_len != len || need > self.outputs_flat.len() {
            self.rebuild_outputs(&**kernel, len, self.exponents.len());
        }
        let mut coeffs: Vec<Vec<u16>> = Vec::with_capacity(staged.len());
        for inp in staged.iter_mut() {
            coeffs.push(std::mem::take(&mut inp.coeffs));
            let packed = std::mem::take(&mut inp.packed);
            // keep the packed region alive inside the staging slot until the
            // multiply completes - store it back for the work loop
            inp.packed = packed;
            let buf = std::mem::replace(
                &mut inp.buf,
                SliceBuf::new(Vec::new().into_boxed_slice(), 0),
            );
            // InputDone is deferred until after the multiply: the staging
            // slot (and the input buffer with it) frees when the batch's
            // compute finishes, not when it is queued
            inp.buf = buf;
        }

        let nthreads = (shared.num_threads.load(Ordering::Relaxed) as usize).clamp(1, 32);
        let nouts = self.exponents.len();
        // scale the fan-out to the demand: each worker sustains ~8GB/s of
        // muladd while the input feed supplies ~1GB/s per output column,
        // so ceil(nouts/8) threads keep pace - spawning the full hardware
        // count only oversubscribes cores against the serial feed thread
        // (reader/hasher). The tail flush runs once the feed is done and
        // gets everything.
        let nthreads = if tail {
            nthreads
        } else {
            nthreads.min((nouts / 8 + 1).max(1))
        };
        // tile_len comes from the layout built by rebuild_outputs so the
        // flat index (round*nouts + out)*tile stays consistent
        let tile_len = self.out_tile;
        let first_mul = !self.processing_add;
        let kernel_ref: &dyn GfKernel = &**kernel;

        // tasks: (tile offset, tile len, output index, dst slice) -
        // tile-major order so threads sweep the same tile range together.
        // The flat region is chopped per round then per output: each
        // round's output tiles are adjacent, so a thread's contiguous
        // task group writes a contiguous dst span (C++ memProcessing
        // layout) instead of jumping between per-output regions
        let nrounds = len.div_ceil(tile_len.max(1));
        let mut tasks: Vec<(usize, usize, usize, &mut [u8])> = Vec::with_capacity(nrounds * nouts);
        // packpf bound: prefetch targets must stay inside the flat buffer
        // (captured before the mutable chunks_mut borrow below)
        let flat_end =
            self.outputs_flat.as_ptr().wrapping_add(self.outputs_flat.len()) as usize;
        if nouts > 0 {
            let mut round_iters: Vec<std::slice::ChunksMut<u8>> = self
                .outputs_flat
                .chunks_mut(nouts * tile_len)
                .map(|r| r.chunks_mut(tile_len))
                .collect();
            for (t, ch) in round_iters.iter_mut().enumerate() {
                let toff = t * tile_len;
                let tl = (len - toff).min(tile_len);
                for (o, dst) in ch.by_ref().enumerate() {
                    tasks.push((toff, tl, o, &mut dst[..tl]));
                }
            }
        }
        let ntasks = tasks.len();
        let coeffs = &coeffs;
        let ninputs = staged.len();
        // PARPAR_NO_PACKED: escape hatch to A/B the interleaved-staging
        // path against plain per-input runs on production workloads
        static NO_PACKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_packed = *NO_PACKED.get_or_init(|| std::env::var_os("PARPAR_NO_PACKED").is_some());
        let use_packed = kernel_ref.supports_packed() && ninputs > 1 && !no_packed;
        // output-major coeff rows for the packed path (C++
        // `vals = coeffs + out*inputGrouping`): one contiguous u16 row
        // per output - no per-task pointer chasing
        let mut coeffs_flat: Vec<u16> = Vec::new();
        if use_packed {
            coeffs_flat.reserve(nouts * ninputs);
            for o in 0..nouts {
                for c in coeffs.iter() {
                    coeffs_flat.push(c.get(o).copied().unwrap_or(0));
                }
            }
        }
        let coeffs_flat = &coeffs_flat;
        // packed path (C++ INTERLEAVE_INPUTS): for kernels that can read
        // block-interleaved staging, input k of each 8-input group gets
        // its block j at stage[g*8*len + ng*32*j + 32k] - the fused
        // muladd then walks one sequential src stream per group instead
        // of 8 separate ones. Short inputs leave zeroed slots (c*0 = 0).
        let ngroups = ninputs.div_ceil(8);
        if use_packed {
            let blk = kernel_ref.block_len();
            // chunk-major layout (C++ srcPtr = input + round*chunkSize*
            // inputGrouping): [round][group][8-interleaved blocks] so a
            // task's whole src window per round is one contiguous span.
            // Rounds partition contiguously across nthreads buffers -
            // separate allocations spread L3 home-node traffic
            let blocks_per_tile = tile_len / blk;
            let nrounds_p = len.div_ceil(tile_len);
            self.packed_round_buf.clear();
            self.packed_round_buf.resize(nrounds_p, 0);
            self.packed_round_local.clear();
            self.packed_round_local.resize(nrounds_p, 0);
            self.packed_stages.truncate(nthreads.max(1));
            while self.packed_stages.len() < nthreads.max(1) {
                self.packed_stages.push(Vec::new());
            }
            for b in 0..self.packed_stages.len() {
                let r_lo = b * nrounds_p / self.packed_stages.len();
                let r_hi = (b + 1) * nrounds_p / self.packed_stages.len();
                let need = (r_hi - r_lo) * ninputs * tile_len;
                if self.packed_stages[b].len() < need {
                    self.packed_stages[b].resize(need, 0);
                }
                for r in r_lo..r_hi {
                    self.packed_round_buf[r] = b;
                    self.packed_round_local[r] = r - r_lo;
                }
            }
            for (i, inp) in staged.iter().enumerate() {
                let g = i / 8;
                let ng = (ninputs - g * 8).min(8);
                let plen = inp.packed.len().min(len);
                let kbase = g * 8 * tile_len + blk * (i % 8);
                let mut j = 0;
                for chunk in inp.packed[..plen].chunks(blk) {
                    let r = j / blocks_per_tile;
                    let b = self.packed_round_buf[r];
                    let off = self.packed_round_local[r] * ninputs * tile_len
                        + kbase
                        + ng * blk * (j % blocks_per_tile);
                    self.packed_stages[b][off..off + chunk.len()].copy_from_slice(chunk);
                    self.packed_stages[b][off + chunk.len()..off + blk].fill(0);
                    j += 1;
                }
                // a staged input may be shorter than the slice (grew after
                // staging or a partial tail slice) - zero its interleaved
                // tail so it contributes c*0 = 0 instead of stale data
                for j in (plen + blk - 1) / blk..len / blk {
                    let r = j / blocks_per_tile;
                    let b = self.packed_round_buf[r];
                    let off = self.packed_round_local[r] * ninputs * tile_len
                        + kbase
                        + ng * blk * (j % blocks_per_tile);
                    self.packed_stages[b][off..off + blk].fill(0);
                }
            }
        }
        let stages = &self.packed_stages;
        let round_buf = &self.packed_round_buf;
        let round_local = &self.packed_round_local;
        let packed_inputs: Vec<&[u8]> = if use_packed {
            Vec::new()
        } else {
            staged.iter().map(|i| &i.packed[..]).collect()
        };
        let work = |group: Vec<(usize, usize, usize, &mut [u8])>| {
            for (toff, tl, o, dst) in group {
                // general-coefficient inputs are accumulated in runs of up
                // to 8 same-length sources (C++ procRegions /
                // CLMUL_NUM_REGIONS): a fused muladd_multi does one dst
                // sweep - and for clmul one reduction - per run. XOR
                // accumulation commutes, so runs only break on a length
                // change or the first-mul overwrite; coeff 0/1 keep their
                // cheap dedicated paths without flushing
                let mut run: Vec<(&[u8], u16)> = Vec::with_capacity(8);
                // packpf: the first run of each task prefetches the next
                // output tile - tiles are tile_len-strided in the flat
                // layout (dst[..tl] may be a short tail), so the next
                // output's tile is always at dst+tile_len; kernels
                // without prefetch ignore it
                let mut pf = dst.as_ptr().wrapping_add(tile_len);
                if (pf as usize) >= flat_end {
                    pf = std::ptr::null();
                }
                let mut flush = |run: &mut Vec<(&[u8], u16)>, dst: &mut [u8]| {
                    kernel_ref.muladd_multi_pf(dst, run, pf);
                    pf = std::ptr::null();
                    run.clear();
                };
                if use_packed {
                    // interleaved staging already encodes the groups;
                    // coeff-0/1 members multiply like the rest (C++ pays
                    // the same) and coeff gaps read as 0
                    if first_mul {
                        dst.fill(0);
                    }
                    let r = toff / tile_len;
                    let buf = &stages[round_buf[r]];
                    let round_base = round_local[r] * ninputs * tile_len;
                    let vals = &coeffs_flat[o * ninputs..o * ninputs + ninputs];
                    for g in 0..ngroups {
                        let ng = (ninputs - g * 8).min(8);
                        let base = round_base + g * 8 * tile_len;
                        kernel_ref.muladd_multi_packed_pf(
                            dst,
                            &buf[base..base + ng * tl],
                            ng,
                            &vals[g * 8..g * 8 + ng],
                            pf,
                        );
                        pf = std::ptr::null();
                    }
                    continue;
                }
                for (i, packed) in packed_inputs.iter().enumerate() {
                    // a staged input may be shorter than tl if the slice
                    // size grew after staging - treat the missing region as
                    // zeros (it contributes c*0 = 0 to every output)
                    let send = (toff + tl).min(packed.len());
                    let src = &packed[toff.min(send)..send];
                    // coeffs were captured at add_input time against the
                    // then-current output count; a later SyncRecovery can
                    // grow outputs - C++ zero-fills resized procCoeffs, so
                    // missing entries act as coefficient 0 (skip)
                    let coeff = coeffs[i].get(o).copied().unwrap_or(0);
                    // C++ routes coefficient-0 outputs through add_multi and
                    // skips them entirely; coefficient-1 is a plain XOR add
                    // (exponent 0 -> GF element 1), no GF multiply needed
                    match coeff {
                        0 if i == 0 && first_mul => dst.fill(0),
                        0 => {}
                        1 if i == 0 && first_mul => {
                            dst[..src.len()].copy_from_slice(src);
                            dst[src.len()..].fill(0);
                        }
                        1 => {
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d ^= s;
                            }
                        }
                        c if i == 0 && first_mul => {
                            kernel_ref.mul(dst, src, c);
                            dst[src.len()..].fill(0);
                        }
                        c => {
                            if run.len() == 8
                                || run.first().is_some_and(|(s, _)| s.len() != src.len())
                            {
                                flush(&mut run, &mut *dst);
                            }
                            run.push((src, c));
                        }
                    }
                }
                if !run.is_empty() {
                    flush(&mut run, dst);
                }
            }
        };
        if nthreads > 1 && ntasks > 1 {
            let mut it = tasks.into_iter();
            let work = &work;
            std::thread::scope(|sc| {
                for t in 0..nthreads {
                    let lo = t * ntasks / nthreads;
                    let hi = (t + 1) * ntasks / nthreads;
                    let group: Vec<_> = it.by_ref().take(hi - lo).collect();
                    if group.is_empty() {
                        continue;
                    }
                    sc.spawn(move || {
                        // the multiply is elastic background work: workers
                        // run nice(19) so they yield proportionally to the
                        // pipeline's serial feed (reader/hasher/consumer)
                        // under contention. SCHED_IDLE was tried and is
                        // worse: it also defers to unrelated host load.
                        #[cfg(unix)]
                        unsafe {
                            libc::setpriority(libc::PRIO_PROCESS, 0, 19);
                        }
                        work(group)
                    });
                }
            });
        } else {
            work(tasks);
        }
        if !self.processing_add {
            self.processing_add = true;
        }
        // staging freed: buffers return to the pool and the pending count
        // drops - this is what lets a Full add_input succeed on retry
        let n = staged.len();
        {
            let mut pool = shared.packed_pool.lock().unwrap();
            for inp in staged.iter_mut() {
                let packed = std::mem::take(&mut inp.packed);
                if pool.len() < 64 && packed.capacity() > 0 {
                    pool.push(packed);
                }
            }
        }
        for inp in staged {
            emit(ProcEvent::InputDone {
                token: inp.token,
                input_index: inp.input_index,
                buf: inp.buf,
            });
        }
        shared.pending.fetch_sub(n, Ordering::Release);
        if shared.progress_on.load(Ordering::Relaxed) {
            emit(ProcEvent::Progress {
                num_inputs: n as i64,
            });
        }
    }

    fn finalize_cksums(&mut self, kernel: &dyn GfKernel) {
        let len = self.aligned_size(kernel);
        self.output_cksum = (0..self.exponents.len())
            .map(|o| {
                let g = self.gather_output(o);
                cksum_update(0, &g[..len.min(g.len())])
            })
            .collect();
    }
}

pub struct NativeGfProc {
    proc_id: usize,
    queues: EventQueues,
    shared: Arc<Shared>,
    tx: Option<Sender<Cmd>>,
    worker: Option<JoinHandle<()>>,
    kernel: Arc<dyn GfKernel>,
    input_batchsize: u32,
    #[allow(dead_code)]
    input_minbatchsize: u32,
    #[allow(dead_code)]
    chunk_size: u32,
    staging_limit: usize,
    #[allow(dead_code)]
    slice_offset: u64,
    slice_size: u64,
    /// caller-side copy of the configured exponents (for coeff packing);
    /// the worker receives its own via Cmd::SyncRecovery
    exponents: Vec<u16>,
    /// tracks end_input issuance (C++ endSignalled); informational only -
    /// deferred adds may legitimately arrive afterwards while the
    /// post-end event pump drains staging
    #[allow(dead_code)]
    input_ended: bool,
}

impl NativeGfProc {
    pub fn new(
        proc_id: usize,
        queues: EventQueues,
        slice_size: u64,
        cpu: Option<&CpuSpec>,
        staging_areas: i32,
    ) -> Result<Self> {
        let (batchsize, minbatch, chunk, off, sz) = match cpu {
            Some(c) => (
                c.input_batchsize.max(1),
                c.input_minbatchsize,
                c.chunk_size,
                c.slice_offset,
                if c.slice_size != 0 {
                    c.slice_size
                } else {
                    slice_size
                },
            ),
            None => (1, 1, 0, 0, slice_size),
        };
        // pick_explicit: unsupported/unimplemented methods error like the
        // C++ proc init rather than silently falling back
        let kernel: Arc<dyn GfKernel> = pick_explicit(cpu.map(|c| c.method).unwrap_or(0))
            .map(Arc::from)
            .ok_or_else(|| BackendError::new("GF16 method not supported on this CPU"))?;
        let staging_limit = staging_areas.max(1) as usize;
        let shared = Arc::new(Shared {
            exponents_set: AtomicBool::new(false),
            outputs_count: AtomicUsize::new(0),
            current_slice_size: AtomicU64::new(sz),
            pending: AtomicUsize::new(0),
            num_threads: AtomicI32::new(
                // C++ GfProc defaults to hardware concurrency; JS only calls
                // setNumThreads when the user passes a count explicitly
                std::thread::available_parallelism()
                    .map(|n| n.get() as i32)
                    .unwrap_or(1),
            ),
            progress_on: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            worker_dead: AtomicBool::new(false),
            packed_pool: Mutex::new(Vec::new()),
        });

        let (tx, rx) = channel::<Cmd>();
        let w_shared = shared.clone();
        let w_queues = queues.clone();
        let w_kernel = kernel.clone();
        let mut w = Worker {
            exponents: Vec::new(),
            outputs_flat: Vec::new(),
            out_tile: 0,
            layout_len: 0,
            output_cksum: Vec::new(),
            processing_add: false,
            batch: Vec::new(),
            packed_stages: Vec::new(),
            packed_round_buf: Vec::new(),
            packed_round_local: Vec::new(),
            needs_discard: false,
            stat_batches: 0,
            slice_size: sz,
            current_slice_size: sz,
            input_batchsize: batchsize,
            chunk_size: chunk,
        };
        let worker = std::thread::spawn(move || {
            // a panic inside the loop must still flag the worker dead -
            // otherwise senders keep succeeding into the dead channel and
            // `pending` never drains (has_staging_room stuck false). Also
            // emit an Error event so waiters stop polling a dead queue.
            struct DeadGuard(Arc<Shared>, EventQueues, usize);
            impl Drop for DeadGuard {
                fn drop(&mut self) {
                    self.0.worker_dead.store(true, Ordering::Release);
                    if std::thread::panicking() {
                        // recover a poisoned lock - the panic may have
                        // happened while holding it (emit/drain), and
                        // losing this event leaves waiters spinning
                        let mut q = self.1.lock().unwrap_or_else(|e| e.into_inner());
                        q.entry(self.2).or_default().push_back(ProcEvent::Error {
                            token: 0,
                            message: "proc worker died".into(),
                        });
                    }
                }
            }
            let _guard = DeadGuard(w_shared.clone(), w_queues.clone(), proc_id);
            let emit = |ev: ProcEvent| {
                w_queues
                    .lock()
                    .unwrap()
                    .entry(proc_id)
                    .or_default()
                    .push_back(ev);
            };
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    Cmd::Input(si) => {
                        w.maybe_discard();
                        w.batch.push(si);
                        if w.batch.len() >= w.effective_batch(&*w_kernel) {
                            w.flush_batch(&w_kernel, &w_shared, &emit, false);
                        }
                    }
                    Cmd::End { token } => {
                        w.maybe_discard();
                        w.flush_batch(&w_kernel, &w_shared, &emit, true);
                        w.finalize_cksums(&*w_kernel);
                        emit(ProcEvent::Ended { token });
                        w.needs_discard = true;
                    }
                    Cmd::Output {
                        index,
                        mut out,
                        token,
                    } => {
                        // outputs can be empty after FreeMem even though the
                        // caller-side index bound (outputs_count) still passes
                        let configured =
                            index < w.exponents.len() && !w.outputs_flat.is_empty();
                        // a round with no input flush leaves stale data in
                        // outputs_flat (chunk passes only feed slices with
                        // real bytes) - its recovery data is definitionally
                        // zero, matching the composite's !has_added fill; the
                        // proc is still valid so the checksum flag stays true
                        let live = configured && w.processing_add;
                        let valid = if live && index < w.output_cksum.len() {
                            let plen = w.aligned_size(&*w_kernel);
                            let g = w.gather_output(index);
                            cksum_update(0, &g[..plen.min(g.len())]) == w.output_cksum[index]
                        } else {
                            configured
                        };
                        let n = w.current_slice_size as usize;
                        let avail = n.min(out.data.len());
                        if live {
                            let src = w.gather_output(index);
                            w_kernel.finish(&mut out.data[..avail], &src);
                            out.len = avail;
                        } else {
                            out.data[..avail].fill(0);
                            out.len = avail;
                        }
                        emit(ProcEvent::OutputDone {
                            token,
                            cksum_valid: valid,
                            buf: out,
                        });
                    }
                    Cmd::SyncRecovery { exponents, count } => {
                        w.exponents = exponents;
                        if w.exponents.len() != count {
                            w.exponents.resize(count, 1);
                        }
                        let len = w.aligned_size(&*w_kernel);
                        let tile = resolve_tile_len(&*w_kernel, w.chunk_size, len);
                        w.out_tile = tile;
                        w.layout_len = len;
                        w.outputs_flat = vec![0u8; len.div_ceil(tile.max(1)) * count * tile];
                        w.output_cksum.clear();
                        w.processing_add = false;
                    }
                    Cmd::SliceSize { alloc, current } => {
                        w.slice_size = w.slice_size.max(alloc);
                        w.current_slice_size = current;
                        // C++ setCurrentSliceSize reallocs memProcessing when
                        // the aligned size grows; outputs must stay >=
                        // aligned(current) or the next flush indexes past the
                        // packed inputs
                        let need = w.aligned_size(&*w_kernel);
                        if need != w.layout_len && !w.outputs_flat.is_empty() {
                            let nouts = w.exponents.len();
                            w.rebuild_outputs(&*w_kernel, need, nouts);
                        }
                    }
                    Cmd::FreeMem => {
                        // C++ freeProcessingMem frees processing buffers but
                        // keeps the staging queue: staged inputs must still
                        // flush (dropping them would lose contributions AND
                        // wedge pending, since no InputDone would ever fire)
                        w.outputs_flat.clear();
                        w.layout_len = 0;
                        w.out_tile = 0;
                        w.output_cksum.clear();
                        w.packed_stages = Vec::new();
                        w.packed_round_buf = Vec::new();
                        w.packed_round_local = Vec::new();
                    }
                    Cmd::Close => break,
                }
            }
        });

        Ok(NativeGfProc {
            proc_id,
            queues,
            shared,
            tx: Some(tx),
            worker: Some(worker),
            kernel,
            input_batchsize: batchsize,
            input_minbatchsize: minbatch.max(1),
            chunk_size: chunk,
            staging_limit,
            slice_offset: off,
            slice_size: sz,
            exponents: Vec::new(),
            input_ended: false,
        })
    }

    /// Capacity for new inputs (used by the composite's can_add probing).
    pub fn has_staging_room(&self) -> bool {
        let batch = Worker::effective_batch_for(
            self.shared.outputs_count.load(Ordering::Acquire),
            self.input_batchsize as usize,
            packed_len(
                &*self.kernel,
                self.shared.current_slice_size.load(Ordering::Acquire) as usize,
            ),
        );
        let cap = self.staging_limit * batch;
        self.shared.pending.load(Ordering::Acquire) < cap
    }

    /// Blocking pop of this proc's own event queue (used by range adapters
    /// that need synchronous semantics). `timeout` of zero waits forever.
    /// A dead worker with an empty queue surfaces as an Error event rather
    /// than spinning forever (covers the case where DeadGuard couldn't
    /// emit, e.g. a poisoned queue lock).
    pub fn wait_own_event(&self, timeout: Duration) -> Option<ProcEvent> {
        let deadline = (!timeout.is_zero()).then(|| Instant::now() + timeout);
        loop {
            if let Some(ev) = self
                .queues
                .lock()
                .unwrap()
                .get_mut(&self.proc_id)
                .and_then(|q| q.pop_front())
            {
                return Some(ev);
            }
            if self.shared.worker_dead.load(Ordering::Acquire) {
                return Some(ProcEvent::Error {
                    token: 0,
                    message: "proc worker stopped".into(),
                });
            }
            match deadline {
                Some(d) if Instant::now() >= d => return None,
                _ => std::thread::sleep(Duration::from_micros(200)),
            }
        }
    }

    fn send(&self, cmd: Cmd) -> Result<()> {
        if self.shared.worker_dead.load(Ordering::Acquire) {
            return Err(BackendError::new("proc worker stopped"));
        }
        self.tx
            .as_ref()
            .ok_or_else(|| BackendError::new("proc closed"))?
            .send(cmd)
            .map_err(|_| BackendError::new("proc worker stopped"))
    }

    fn aligned_size(&self) -> usize {
        packed_len(
            &*self.kernel,
            self.shared.current_slice_size.load(Ordering::Relaxed) as usize,
        )
    }
}

impl Drop for NativeGfProc {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Cmd::Close);
            drop(tx);
        }
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

impl GfProc for NativeGfProc {
    fn proc_id(&self) -> usize {
        self.proc_id
    }

    fn set_num_threads(&mut self, threads: i32) -> Result<i32> {
        // C++ setNumThreads(-1) means auto = hardware_concurrency
        let threads = if threads < 0 {
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(1)
        } else {
            threads
        };
        self.shared.num_threads.store(threads, Ordering::Relaxed);
        Ok(threads)
    }

    fn set_current_slice_size(&mut self, slice_size: u64) -> Result<()> {
        if slice_size > self.slice_size {
            self.slice_size = slice_size;
        }
        self.shared
            .current_slice_size
            .store(slice_size, Ordering::Release);
        self.send(Cmd::SliceSize {
            alloc: slice_size,
            current: slice_size,
        })
    }

    fn set_recovery_slices(&mut self, exponents: Option<&[u16]>, count: usize) -> Result<()> {
        let mut exps = match exponents {
            Some(e) => e.to_vec(),
            // C++ default: all exponents = 1
            None => vec![1u16; count],
        };
        if exps.len() != count {
            exps.resize(count, 1);
        }
        self.shared.exponents_set.store(true, Ordering::Release);
        self.shared.outputs_count.store(count, Ordering::Release);
        self.exponents = exps.clone();
        // a new recovery set marks a new processing round (multi-pass
        // reuse) - clear the end-input latch
        self.input_ended = false;
        self.send(Cmd::SyncRecovery {
            exponents: exps,
            count,
        })
    }

    fn free_mem(&mut self) -> Result<()> {
        self.send(Cmd::FreeMem)
    }

    fn add_input(&mut self, buf: SliceBuf, input_index: i32, token: u64) -> Result<AddResult> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(BackendError::new("Cannot add input after proc closed"));
        }
        if !self.shared.exponents_set.load(Ordering::Acquire) {
            return Err(BackendError::new("Recovery slices not set"));
        }
        if buf.len > buf.data.len() {
            return Err(BackendError::new("Input buffer length exceeds data"));
        }
        if buf.len as u64 > self.shared.current_slice_size.load(Ordering::Relaxed) {
            return Err(BackendError::new("Input buffer too large"));
        }
        // a zero-size input is accepted but never completes - the C++
        // backend refcount never reaches zero so no InputDone event fires
        if buf.len == 0 {
            return Ok(AddResult::Accepted);
        }
        // report a dead worker as an error rather than Full forever -
        // pending never drains after a panic so staging never frees
        if self.shared.worker_dead.load(Ordering::Acquire) {
            return Err(BackendError::new("proc worker stopped"));
        }
        if !self.has_staging_room() {
            return Ok(AddResult::Full(buf));
        }
        // input_diff table covers blocks 0..32768 (C++ accepts u16 but the
        // table lookup requires < 32768)
        if !(0..32768).contains(&input_index) {
            return Err(BackendError::new("Input index out of range"));
        }
        let input_log = gfmat_input_log(input_index as u16);
        let coeffs = self
            .exponents
            .iter()
            .map(|&e| gfmat_coeff_from_log(input_log, e))
            .collect();
        // pack on the caller thread (mirrors prepare_packed inside the C++
        // addInput) so the worker only does the multiply. Stage at the max
        // allocated slice size like C++ staging; the work loop additionally
        // clamps reads to packed.len() for grows after staging.
        let plen = packed_len(&*self.kernel, self.slice_size as usize);
        let mut packed = self
            .shared
            .packed_pool
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_default();
        packed.clear();
        packed.reserve(plen);
        // every kernel prepare() fully writes dst[..plen] (packed data or
        // explicit zero-pad) so initialising from the pool is unobservable
        unsafe {
            packed.set_len(plen);
        }
        self.kernel.prepare(&mut packed, &buf.data[..buf.len]);
        // the raw input is packed now - the storage isn't needed by the
        // worker; return it to the origin pool (e.g. reader buffers) here
        // instead of at InputDone so the pool can't starve on the way
        let mut buf = buf;
        buf.release_storage();
        self.shared.pending.fetch_add(1, Ordering::AcqRel);
        if let Err(e) = self.send(Cmd::Input(StagedInput {
            buf,
            input_index: input_index as i64,
            token,
            coeffs,
            packed,
        })) {
            self.shared.pending.fetch_sub(1, Ordering::AcqRel);
            return Err(e);
        }
        Ok(AddResult::Accepted)
    }

    fn end_input(&mut self, token: u64) -> Result<()> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(BackendError::new("Cannot end input after proc closed"));
        }
        self.send(Cmd::End { token })?;
        self.input_ended = true;
        Ok(())
    }

    fn get_output(&mut self, output_index: i32, out: SliceBuf, token: u64) -> Result<()> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(BackendError::new("Cannot get output after proc closed"));
        }
        let idx = output_index as usize;
        if idx >= self.shared.outputs_count.load(Ordering::Acquire) {
            return Err(BackendError::new("Invalid output index"));
        }
        self.send(Cmd::Output {
            index: idx,
            out,
            token,
        })
    }

    fn set_progress(&mut self, on: bool) -> Result<()> {
        self.shared.progress_on.store(on, Ordering::Relaxed);
        Ok(())
    }

    fn info(&mut self) -> Result<GfInfo> {
        // report the loop tiling like C++ getChunkLen(): the configured
        // loopTileSize, or the kernel's idealChunkSize when auto (0) -
        // not the whole aligned region
        let ideal = crate::kernel::method_info(self.kernel.id()).target_chunk;
        let tile = if self.chunk_size > 0 {
            self.chunk_size as u64
        } else {
            ideal
        };
        Ok(GfInfo {
            cpu: Some(GfCpuInfo {
                threads: self.shared.num_threads.load(Ordering::Relaxed),
                method_desc: self.kernel.name().to_string(),
                chunk_size: tile,
                staging_count: self.staging_limit as u32,
                staging_size: self.input_batchsize as u64,
                alignment: self.kernel.alignment() as u32,
                stride: self.kernel.stride() as u64,
                slice_mem: self.aligned_size() as u64,
                num_output_slices: self.shared.outputs_count.load(Ordering::Relaxed) as i32,
            }),
            ocl: Vec::new(),
        })
    }

    fn close(&mut self) -> Result<()> {
        self.shared.closed.store(true, Ordering::Release);
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Cmd::Close);
            drop(tx);
        }
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
        // sentinel for any stale waiter - a wait on a closed proc must
        // terminate rather than spin on an empty queue forever
        if let Ok(mut q) = self.queues.lock() {
            q.entry(self.proc_id)
                .or_default()
                .push_back(ProcEvent::Error {
                    token: 0,
                    message: "proc closed".into(),
                });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        let q: EventQueues = Default::default();
        let cpu = CpuSpec {
            method: 0,
            input_batchsize: 1,
            input_minbatchsize: 1,
            chunk_size: 0,
            slice_offset: 0,
            slice_size: 0,
        };
        let mut p = NativeGfProc::new(0, q.clone(), 64, Some(&cpu), 2).unwrap();
        p.set_recovery_slices(None, 3).unwrap();
        p.add_input(SliceBuf::new(vec![7u8; 64].into_boxed_slice(), 64), 0, 1)
            .unwrap();
        p.end_input(2).unwrap();
        // default exponents are all 1 -> coeff = g^(1*1) = 2 -> 2*input;
        // outputs are stored packed - read back through get_output, which
        // delivers the filled buffer via an OutputDone event
        let out = SliceBuf::new(vec![0u8; 64].into_boxed_slice(), 64);
        p.get_output(0, out, 9).unwrap();
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while events.len() < 3 && Instant::now() < deadline {
            if let Some(e) = p.wait_own_event(Duration::from_millis(50)) {
                events.push(e);
            }
        }
        assert_eq!(events.len(), 3);
        let mut saw = false;
        for e in events {
            if let ProcEvent::OutputDone { buf, .. } = e {
                assert_eq!(&buf.data[..8], &[14u8; 8]);
                saw = true;
            }
        }
        assert!(saw);
    }

    /// input_index indexes a 32768-entry gf matrix - values >= 32768 must
    /// error, not panic the worker mid-coefficient-lookup
    #[test]
    fn input_index_out_of_range_errors() {
        let q: EventQueues = Default::default();
        let cpu = CpuSpec {
            method: 0,
            input_batchsize: 1,
            input_minbatchsize: 1,
            chunk_size: 0,
            slice_offset: 0,
            slice_size: 0,
        };
        let mut p = NativeGfProc::new(0, q.clone(), 64, Some(&cpu), 2).unwrap();
        p.set_recovery_slices(None, 3).unwrap();
        let mk = || SliceBuf::new(vec![1u8; 64].into_boxed_slice(), 64);
        assert!(p.add_input(mk(), 32768, 1).is_err());
        assert!(p.add_input(mk(), 40000, 2).is_err());
        // the boundary value is still valid
        assert!(matches!(
            p.add_input(mk(), 32767, 3).unwrap(),
            AddResult::Accepted
        ));
    }

    /// add_input with a buf.len larger than the slice size must error
    /// before prepare() can slice out of bounds
    #[test]
    fn oversized_input_errors() {
        let q: EventQueues = Default::default();
        let cpu = CpuSpec {
            method: 0,
            input_batchsize: 1,
            input_minbatchsize: 1,
            chunk_size: 0,
            slice_offset: 0,
            slice_size: 0,
        };
        let mut p = NativeGfProc::new(0, q.clone(), 64, Some(&cpu), 2).unwrap();
        p.set_recovery_slices(None, 3).unwrap();
        assert!(p
            .add_input(SliceBuf::new(vec![1u8; 128].into_boxed_slice(), 128), 0, 1)
            .is_err());
        // buf.len exceeding buf.data.len() is invalid too (constructed
        // manually - SliceBuf::new debug_asserts against it)
        assert!(p
            .add_input(
                SliceBuf {
                    data: vec![1u8; 32].into_boxed_slice(),
                    len: 64,
                    reclaim: None
                },
                0,
                2
            )
            .is_err());
    }

    /// Chunked generation only feeds inputs with real bytes; a chunk pass
    /// where every file ends before chunk_offset sees no adds at all. Its
    /// outputs must read as zeros - outputs_flat still holds the previous
    /// pass's accumulation, and returning it rewrites pass-0 data at the
    /// new chunk offset (JS oracle caught this as stale recovery data).
    #[test]
    fn empty_round_output_is_zero() {
        let q: EventQueues = Default::default();
        let cpu = CpuSpec {
            method: 0,
            input_batchsize: 1,
            input_minbatchsize: 1,
            chunk_size: 0,
            slice_offset: 0,
            slice_size: 0,
        };
        let mut p = NativeGfProc::new(0, q.clone(), 64, Some(&cpu), 2).unwrap();
        p.set_recovery_slices(None, 2).unwrap();
        let mut drain = |p: &NativeGfProc| {
            while let Some(e) = p.wait_own_event(Duration::from_millis(50)) {
                if matches!(e, ProcEvent::Ended { .. }) {
                    break;
                }
            }
        };
        // round 1: real input, output must be nonzero
        p.add_input(SliceBuf::new(vec![7u8; 64].into_boxed_slice(), 64), 0, 1)
            .unwrap();
        p.end_input(2).unwrap();
        drain(&p);
        let out = SliceBuf::new(vec![0u8; 64].into_boxed_slice(), 64);
        p.get_output(0, out, 3).unwrap();
        let mut nonzero = false;
        while let Some(e) = p.wait_own_event(Duration::from_millis(50)) {
            if let ProcEvent::OutputDone { buf, cksum_valid, .. } = e {
                assert!(buf.data.iter().any(|&b| b != 0));
                assert!(cksum_valid);
                nonzero = true;
                break;
            }
        }
        assert!(nonzero, "first round produced no output event");
        // round 2: end with no adds -> outputs must read as zeros, still
        // checksum-valid (empty pass is not corruption)
        p.end_input(4).unwrap();
        drain(&p);
        let out = SliceBuf::new(vec![0u8; 64].into_boxed_slice(), 64);
        p.get_output(0, out, 5).unwrap();
        while let Some(e) = p.wait_own_event(Duration::from_millis(50)) {
            if let ProcEvent::OutputDone { buf, cksum_valid, .. } = e {
                assert!(buf.data.iter().all(|&b| b == 0));
                assert!(cksum_valid);
                return;
            }
        }
        panic!("no output event for empty round");
    }

    /// get_output after free_mem must not panic - C++ returns a zeroed
    /// buffer with cksum_valid=false semantics
    #[test]
    fn output_after_free_mem_no_panic() {
        let q: EventQueues = Default::default();
        let cpu = CpuSpec {
            method: 0,
            input_batchsize: 1,
            input_minbatchsize: 1,
            chunk_size: 0,
            slice_offset: 0,
            slice_size: 0,
        };
        let mut p = NativeGfProc::new(0, q.clone(), 64, Some(&cpu), 2).unwrap();
        p.set_recovery_slices(None, 3).unwrap();
        p.free_mem().unwrap();
        let out = SliceBuf::new(vec![0u8; 64].into_boxed_slice(), 64);
        p.get_output(0, out, 7).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match p.wait_own_event(Duration::from_millis(50)) {
                Some(ProcEvent::OutputDone { cksum_valid, .. }) => {
                    assert!(!cksum_valid);
                    break;
                }
                Some(ProcEvent::Error { message, .. }) => {
                    panic!("unexpected error: {}", message)
                }
                _ => assert!(Instant::now() < deadline, "timed out"),
            }
        }
    }
}

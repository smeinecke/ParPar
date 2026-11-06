//! Ad-hoc throughput probe: simulate the -s1M batch multiply across N
//! threads and report aggregate GB/s. Not a correctness test - run with
//! `cargo test -p parpar-gf16 --release --test perf_probe -- --nocapture`.

use parpar_gf16::kernel;
use std::time::Instant;

#[test]
#[ignore = "ad-hoc throughput probe - run with --ignored; too slow under qemu CI"]
fn aggregate_muladd_throughput() {
    let k = kernel::pick(0);
    eprintln!("kernel {} {}", k.id(), k.name());
    let tile = 64 * 1024;
    let slice = 1024 * 1024;
    for (nins, nouts, nthreads, interleaved) in [
        (12usize, 205usize, 16usize, false),
        (12, 205, 16, true),
        (12, 205, 8, false),
        (12, 52, 16, false),
        (12, 52, 16, true),
    ] {
        // interleaved: input i's tile t lives at t*(nins*tile)+i*tile so a
        // round's input window is one contiguous span (C++ staging layout)
        let flat: Vec<u8> = vec![0x5au8; nins * slice];
        let ins: Vec<Vec<u8>> = (0..nins).map(|i| vec![(i as u8) ^ 0x5a; slice]).collect();
        let mut outs: Vec<Vec<u8>> = (0..nouts).map(|_| vec![0u8; slice]).collect();
        let per = nouts.div_ceil(nthreads).max(1);
        let reps = 8;
        let t0 = Instant::now();
        let bytes = std::sync::atomic::AtomicU64::new(0);
        for _ in 0..reps {
            std::thread::scope(|sc| {
                let ins = &ins;
                let k = &*k;
                let bytes = &bytes;
                let flat = &flat;
                for grp in outs.chunks_mut(per) {
                    sc.spawn(move || {
                        let mut rng = 0x9e3779b9u32;
                        let mut done = 0u64;
                        for dst in grp.iter_mut() {
                            for t in 0..(slice / tile) {
                                let d = &mut dst[t * tile..(t + 1) * tile];
                                for (i, src) in ins.iter().enumerate() {
                                    rng ^= rng << 13;
                                    rng ^= rng >> 17;
                                    rng ^= rng << 5;
                                    let c = ((rng as u16) & 0xffff) | 1;
                                    let s = if interleaved {
                                        &flat[t * nins * tile + i * tile
                                            ..t * nins * tile + (i + 1) * tile]
                                    } else {
                                        &src[t * tile..(t + 1) * tile]
                                    };
                                    k.muladd(d, s, c);
                                    done += tile as u64;
                                }
                            }
                        }
                        bytes.fetch_add(done, std::sync::atomic::Ordering::Relaxed);
                    });
                }
            });
        }
        let dt = t0.elapsed();
        let bytes = bytes.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "nins={nins} nouts={nouts} thr={nthreads}: {:.2}s  agg {:.1} GB/s ({:.2} GB/s/core)",
            dt.as_secs_f64(),
            bytes as f64 / dt.as_secs_f64() / 1e9,
            bytes as f64 / dt.as_secs_f64() / 1e9 / nthreads as f64
        );
    }

    // single-thread muladd2 at the same shape as the C++ FFI probe
    let nins = 24usize;
    let nouts = 205usize;
    let nrounds = slice / tile;
    let ins: Vec<Vec<u8>> = (0..nins).map(|i| vec![(i as u8) ^ 0x5a; slice]).collect();
    // [round][out][tile] like the C++ probe - dst isn't cache-hot
    let mut outs_flat = vec![0u8; nrounds * nouts * tile];
    let mut coeffs = vec![0u16; nouts * nins];
    let mut rng = 0x9e3779b9u32;
    for c in coeffs.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        *c = (rng as u16) | 1;
    }
    let reps = 8;
    let t0 = Instant::now();
    for _ in 0..reps {
        for r in 0..nrounds {
            for o in 0..nouts {
                let d = &mut outs_flat[(r * nouts + o) * tile..(r * nouts + o + 1) * tile];
                let mut i = 0usize;
                while i + 1 < nins {
                    k.muladd2(
                        d,
                        &ins[i][r * tile..(r + 1) * tile],
                        coeffs[o * nins + i],
                        &ins[i + 1][r * tile..(r + 1) * tile],
                        coeffs[o * nins + i + 1],
                    );
                    i += 2;
                }
            }
        }
    }
    let dt = t0.elapsed();
    let bytes = (nins * nouts * slice * reps) as f64;
    eprintln!(
        "Rust muladd2 single-thread: {:.2}s -> {:.1} GB/s",
        dt.as_secs_f64(),
        bytes / dt.as_secs_f64() / 1e9
    );

    // per-input muladd, contiguous staging arena (C++ fallback layout):
    // input i's chunk r at r*(nins*tile) + i*tile
    let mut arena = vec![0x5au8; nins * slice];
    for (i, input) in ins.iter().enumerate() {
        for r in 0..nrounds {
            arena[r * nins * tile + i * tile..r * nins * tile + (i + 1) * tile]
                .copy_from_slice(&input[r * tile..(r + 1) * tile]);
        }
    }
    let mut outs2 = vec![0u8; nrounds * nouts * tile];
    let t0 = Instant::now();
    for _ in 0..reps {
        for r in 0..nrounds {
            for o in 0..nouts {
                let d = &mut outs2[(r * nouts + o) * tile..(r * nouts + o + 1) * tile];
                for i in 0..nins {
                    k.muladd(
                        d,
                        &arena[r * nins * tile + i * tile..r * nins * tile + (i + 1) * tile],
                        coeffs[o * nins + i],
                    );
                }
            }
        }
    }
    let dt = t0.elapsed();
    let bytes = (nins * nouts * slice * reps) as f64;
    eprintln!(
        "Rust muladd (contig arena) single-thread: {:.2}s -> {:.1} GB/s",
        dt.as_secs_f64(),
        bytes / dt.as_secs_f64() / 1e9
    );

    // same but with separate input Vecs (our current proc layout)
    let mut outs3 = vec![0u8; nrounds * nouts * tile];
    let t0 = Instant::now();
    for _ in 0..reps {
        for r in 0..nrounds {
            for o in 0..nouts {
                let d = &mut outs3[(r * nouts + o) * tile..(r * nouts + o + 1) * tile];
                for i in 0..nins {
                    k.muladd(d, &ins[i][r * tile..(r + 1) * tile], coeffs[o * nins + i]);
                }
            }
        }
    }
    let dt = t0.elapsed();
    let bytes = (nins * nouts * slice * reps) as f64;
    eprintln!(
        "Rust muladd (sep buffers) single-thread: {:.2}s -> {:.1} GB/s",
        dt.as_secs_f64(),
        bytes / dt.as_secs_f64() / 1e9
    );
}

/// The C++ kernel via FFI at the same shape as the real proc: inputs packed
/// into their interleaved staging arena by prepare_packed_cksum, then
/// muladd_multi_packed per (round, output) tile - exactly what
/// compute_worker does. Throughput probe only.
#[cfg(all(target_arch = "x86_64", target_env = "gnu"))]
mod ffi_kernel {
    use std::ffi::c_void;
    use std::time::Instant;
    // force parpar-ffi (and its native archive with the gf16 kernels)
    // into the test binary's link graph
    #[allow(unused_imports)]
    use parpar_ffi as _;
    use parpar_gf16::kernel;

    #[allow(non_snake_case)]
    extern "C" {
        fn gf16_shuffle_init_x86(polynomial: i32) -> *mut c_void;
        fn gf16_shuffle_prepare_packed_cksum_avx2(
            dst: *mut u8,
            src: *const u8,
            srcLen: usize,
            sliceLen: usize,
            inputPackSize: u32,
            inputNum: u32,
            chunkLen: usize,
        );
        fn gf16_shuffle_muladd_avx2(
            scratch: *const c_void,
            dst: *mut u8,
            src: *const u8,
            len: usize,
            val: u16,
            mutScratch: *mut c_void,
        );
        fn gf16_shuffle2x_prepare_packed_cksum_avx2(
            dst: *mut u8,
            src: *const u8,
            srcLen: usize,
            sliceLen: usize,
            inputPackSize: u32,
            inputNum: u32,
            chunkLen: usize,
        );
        fn gf16_shuffle2x_muladd_multi_packed_avx2(
            scratch: *const c_void,
            packedRegions: u32,
            regions: u32,
            dst: *mut u8,
            src: *const u8,
            len: usize,
            coefficients: *const u16,
            mutScratch: *mut c_void,
        );
        fn gf16_shuffle2x_finish_packed_avx2(
            scratch: *const c_void,
            dst: *mut u8,
            src: *const u8,
            len: usize,
        );
    }

    /// 64B-aligned buffer (the C++ kernels ALIGN_ALLOC everything and use
    /// aligned vmovdqa on dst pointers)
    struct Aligned {
        ptr: *mut u8,
        layout: std::alloc::Layout,
        len: usize,
    }
    impl Aligned {
        fn new(len: usize) -> Self {
            let layout = std::alloc::Layout::from_size_align(len, 64).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Aligned { ptr, layout, len }
        }
        fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
    impl Drop for Aligned {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    /// Benchmark the C++ shuffle2x muladd_multi_packed end-to-end: pack
    /// inputs into their staging arena, then run the (round, output)
    /// multiply sweep.
    fn bench_cpp_shuffle2x() {
        let nins = 24usize;
        let nouts = 205usize;
        let slice = 1024 * 1024usize;
        let tile = 64 * 1024usize;
        let nrounds = slice / tile;

        let scratch = unsafe { gf16_shuffle_init_x86(0x1100B) };
        assert!(!scratch.is_null());

        let mut arena = Aligned::new(nins * slice * 2 + (1 << 20));
        let inputs: Vec<Aligned> = (0..nins)
            .map(|i| {
                let mut a = Aligned::new(slice);
                a.as_mut_slice().fill((i as u8) ^ 0x5a);
                a
            })
            .collect();
        for (i, input) in inputs.iter().enumerate() {
            unsafe {
                gf16_shuffle2x_prepare_packed_cksum_avx2(
                    arena.ptr, input.ptr, slice, slice, nins as u32, i as u32, tile,
                );
            }
        }
        assert!(arena.as_slice().iter().any(|&b| b != 0), "pack produced no data");

        let mut outs = Aligned::new(nrounds * nouts * tile);
        let mut coeffs = vec![0u16; nouts * nins];
        let mut rng = 0x9e3779b9u32;
        for c in coeffs.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            *c = (rng as u16) | 1;
        }
        let scratchbuf = Aligned::new(1 << 20);

        // NOTE: odd rep count - each dst cell gets `reps` accumulations of
        // identical products, and GF(2) XOR cancels on even counts
        let reps = 7;
        let t0 = Instant::now();
        for _ in 0..reps {
            for r in 0..nrounds {
                let src = unsafe { arena.ptr.add(r * tile * nins) };
                for o in 0..nouts {
                    let dst = unsafe { outs.ptr.add((r * nouts + o) * tile) };
                    unsafe {
                        gf16_shuffle2x_muladd_multi_packed_avx2(
                            scratch,
                            nins as u32,
                            nins as u32,
                            dst,
                            src,
                            tile,
                            coeffs.as_ptr().add(o * nins),
                            scratchbuf.ptr as _,
                        );
                    }
                }
            }
        }
        let dt = t0.elapsed();
        let bytes = (nins * nouts * slice * reps) as f64;
        eprintln!(
            "C++ shuffle2x muladd_multi_packed single-thread: {:.2}s -> {:.1} GB/s",
            dt.as_secs_f64(),
            bytes / dt.as_secs_f64() / 1e9
        );
        assert!(outs.as_slice().iter().any(|&b| b != 0), "no output written");
    }

    /// Per-input fallback path: what JS `shuffle-avx2` actually runs in this
    /// build - `_mul_add` over contiguous per-input regions.
    fn bench_cpp_muladd_fallback() {
        let nins = 24usize;
        let nouts = 205usize;
        let slice = 1024 * 1024usize;
        let tile = 64 * 1024usize;
        let nrounds = slice / tile;

        let scratch = unsafe { gf16_shuffle_init_x86(0x1100B) };
        assert!(!scratch.is_null());

        // contiguous staging: input i's chunk c at c*(nins*len) + i*len
        let mut arena = Aligned::new(nins * slice + (1 << 20));
        let inputs: Vec<Aligned> = (0..nins)
            .map(|i| {
                let mut a = Aligned::new(slice);
                a.as_mut_slice().fill((i as u8) ^ 0x5a);
                a
            })
            .collect();
        // the shuffle muladd expects the kernel-internal data format; use
        // the real packer (interleave=1 in this build)
        for (i, input) in inputs.iter().enumerate() {
            unsafe {
                gf16_shuffle_prepare_packed_cksum_avx2(
                    arena.ptr, input.ptr, slice, slice, nins as u32, i as u32, tile,
                );
            }
        }
        eprintln!(
            "fallback arena nonzero: {:#x}",
            arena.as_slice().iter().filter(|&&b| b != 0).count()
        );

        let mut outs = Aligned::new(nrounds * nouts * tile);
        let mut coeffs = vec![0u16; nouts * nins];
        let mut rng = 0x9e3779b9u32;
        for c in coeffs.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            *c = (rng as u16) | 1;
        }
        let scratchbuf = Aligned::new(1 << 20);

        // odd rep count - GF(2) XOR accumulation cancels on even counts
        let reps = 7;
        let t0 = Instant::now();
        for _ in 0..reps {
            for r in 0..nrounds {
                for o in 0..nouts {
                    let dst = unsafe { outs.ptr.add((r * nouts + o) * tile) };
                    for i in 0..nins {
                        let src = unsafe { arena.ptr.add(r * tile * nins + i * tile) };
                        unsafe {
                            gf16_shuffle_muladd_avx2(
                                scratch,
                                dst,
                                src,
                                tile,
                                coeffs[o * nins + i],
                                scratchbuf.ptr as _,
                            );
                        }
                    }
                }
            }
        }
        let dt = t0.elapsed();
        let bytes = (nins * nouts * slice * reps) as f64;
        eprintln!(
            "C++ shuffle-avx2 per-input fallback single-thread: {:.2}s -> {:.1} GB/s",
            dt.as_secs_f64(),
            bytes / dt.as_secs_f64() / 1e9
        );
        assert!(outs.as_slice().iter().any(|&b| b != 0), "no output written");
    }

    /// Byte-for-byte clone of the C++ gf16_shuffle_muladd_avx2 inner loop
    /// (gf16_shuffle_avx2.c): 8 tables in ymm9-15 + ymm6 via stack, mask
    /// ymm4, dst planes [lo@+0][hi@+32]. Timing-only probe - verifies
    /// whether the ~1.3x gap is codegen or context.
    ///
    /// # Safety
    /// dst/src must be readable/writable for `len` bytes, 32B-aligned.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn cpp_clone_muladd(dst: *mut u8, src: *const u8, len: usize, tables: &[u8; 256]) {
        // tables: 8 x 32B ymm constants [4 lo-plane, 4 hi-plane]
        core::arch::asm!(
            "vpbroadcastb ymm4, byte ptr [{mask}]",
            "vmovdqa    ymm9,  [{t}]",
            "vmovdqa    ymm10, [{t}+32]",
            "vmovdqa    ymm11, [{t}+64]",
            "vmovdqa    ymm12, [{t}+96]",
            "vmovdqa    ymm13, [{t}+128]",
            "vmovdqa    ymm14, [{t}+160]",
            "vmovdqa    ymm15, [{t}+192]",
            // ymm6 = 8th table (reload from memory each iter like C++)
            "add {src}, {cnt}",
            "add {dst}, {cnt}",
            "neg {cnt}",
            "2:",
            "vmovdqa  ymm6, [{src}+{cnt}+32]",
            "vmovdqa  ymm2, [{src}+{cnt}]",
            "vpand    ymm5, ymm4, [{src}+{cnt}+32]",
            "vpand    ymm1, ymm4, [{src}+{cnt}]",
            "vpsrlw   ymm0, ymm6, 4",
            "vpsrlw   ymm2, ymm2, 4",
            "vpand    ymm0, ymm4, ymm0",
            "vpand    ymm2, ymm4, ymm2",
            "vpshufb  ymm7, ymm10, ymm5",
            "vpshufb  ymm3, ymm14, ymm0",
            "vpshufb  ymm6, ymm13, ymm1",
            "vpshufb  ymm8, ymm12, ymm2",
            "vpshufb  ymm5, ymm15, ymm5",
            "vpxor    ymm6, ymm6, ymm8",
            "vpshufb  ymm0, ymm11, ymm0",
            "vpxor    ymm3, ymm3, ymm5",
            "vpshufb  ymm1, ymm9, ymm1",
            "vpxor    ymm0, ymm0, ymm7",
            "vpxor    ymm3, ymm3, ymm6",
            "vmovdqa  ymm6, [{t}+224]",
            "vpxor    ymm3, ymm3, [{dst}+{cnt}+32]",
            "vpshufb  ymm2, ymm6, ymm2",
            "vmovdqa  [{dst}+{cnt}+32], ymm3",
            "vpxor    ymm1, ymm1, ymm2",
            "vpxor    ymm1, ymm1, ymm0",
            "vpxor    ymm1, ymm1, [{dst}+{cnt}]",
            "vmovdqa  [{dst}+{cnt}], ymm1",
            "add      {cnt}, 0x40",
            "jne      2b",
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            cnt = inout(reg) len => _,
            t = in(reg) tables.as_ptr(),
            mask = in(reg) &[0x0fu8; 1].as_ptr(),
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
        );
    }

    /// Variant of cpp_clone_muladd with the 8th table held in ymm6 as a
    /// register (no per-iter stack reload) and single-load nibble
    /// extraction like our Rust kernel - bisects the codegen delta.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn clone_muladd_8reg(dst: *mut u8, src: *const u8, len: usize, tables: &[u8; 256]) {
        core::arch::asm!(
            "vpbroadcastb ymm8, byte ptr [{mask}]",
            "vmovdqa    ymm0,  [{t}]",
            "vmovdqa    ymm1,  [{t}+32]",
            "vmovdqa    ymm2,  [{t}+64]",
            "vmovdqa    ymm3,  [{t}+96]",
            "vmovdqa    ymm4,  [{t}+128]",
            "vmovdqa    ymm5,  [{t}+160]",
            "vmovdqa    ymm6,  [{t}+192]",
            "vmovdqa    ymm7,  [{t}+224]",
            "add {src}, {cnt}",
            "add {dst}, {cnt}",
            "neg {cnt}",
            "2:",
            "vmovdqu  ymm9,  [{src}+{cnt}]",
            "vmovdqu  ymm10, [{src}+{cnt}+32]",
            "vpsrlw   ymm11, ymm9, 4",
            "vpsrlw   ymm12, ymm10, 4",
            "vpand    ymm10, ymm8, ymm10",
            "vpshufb  ymm13, ymm0, ymm10",
            "vpshufb  ymm10, ymm4, ymm10",
            "vpand    ymm12, ymm8, ymm12",
            "vpshufb  ymm14, ymm1, ymm12",
            "vpshufb  ymm12, ymm5, ymm12",
            "vpand    ymm9, ymm8, ymm9",
            "vpshufb  ymm15, ymm2, ymm9",
            "vpxor    ymm13, ymm13, ymm15",
            "vpxor    ymm13, ymm13, ymm14",
            "vpshufb  ymm9, ymm6, ymm9",
            "vpxor    ymm9, ymm9, ymm10",
            "vpxor    ymm9, ymm9, ymm12",
            "vpand    ymm10, ymm8, ymm11",
            "vpshufb  ymm11, ymm3, ymm10",
            "vpxor    ymm11, ymm13, ymm11",
            "vpshufb  ymm10, ymm7, ymm10",
            "vpxor    ymm9, ymm9, ymm10",
            "vpxor    ymm10, ymm11, [{dst}+{cnt}+32]",
            "vpxor    ymm9, ymm9, [{dst}+{cnt}]",
            "vmovdqu  [{dst}+{cnt}], ymm9",
            "vmovdqu  [{dst}+{cnt}+32], ymm10",
            "add      {cnt}, 0x40",
            "jne      2b",
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            cnt = inout(reg) len => _,
            t = in(reg) tables.as_ptr(),
            mask = in(reg) &[0x0fu8; 1].as_ptr(),
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
        );
    }

    /// Same-buffers A/B: alternate full sweeps of the C++ FFI kernel and our
    /// Rust muladd in one process so environmental noise hits both equally.
    #[test]
    #[ignore = "ad-hoc throughput probe - run with --ignored; too slow under qemu CI"]
    fn interleaved_ab() {
        let nins = 24usize;
        let nouts = 205usize;
        let slice = 1024 * 1024usize;
        let tile = 64 * 1024usize;
        let nrounds = slice / tile;

        let scratch = unsafe { gf16_shuffle_init_x86(0x1100B) };
        assert!(!scratch.is_null());
        let arena = Aligned::new(nins * slice + (1 << 20));
        let inputs: Vec<Aligned> = (0..nins)
            .map(|i| {
                let mut a = Aligned::new(slice);
                a.as_mut_slice().fill((i as u8) ^ 0x5a);
                a
            })
            .collect();
        for (i, input) in inputs.iter().enumerate() {
            unsafe {
                gf16_shuffle_prepare_packed_cksum_avx2(
                    arena.ptr, input.ptr, slice, slice, nins as u32, i as u32, tile,
                );
            }
        }
        let outs = Aligned::new(nrounds * nouts * tile);
        let mut coeffs = vec![0u16; nouts * nins];
        let mut rng = 0x9e3779b9u32;
        for c in coeffs.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            *c = (rng as u16) | 1;
        }
        let scratchbuf = Aligned::new(1 << 20);
        let k = kernel::pick(0);

        let reps = 5; // odd - even counts XOR-cancel dst back to zero
        let mut t_cpp = std::time::Duration::ZERO;
        let mut t_rs = std::time::Duration::ZERO;
        for _ in 0..reps {
            {
                let t0 = Instant::now();
                for r in 0..nrounds {
                    for o in 0..nouts {
                        let dst = unsafe { outs.ptr.add((r * nouts + o) * tile) };
                        for i in 0..nins {
                            let src = unsafe { arena.ptr.add(r * tile * nins + i * tile) };
                            unsafe {
                                gf16_shuffle_muladd_avx2(
                                    scratch,
                                    dst,
                                    src,
                                    tile,
                                    coeffs[o * nins + i],
                                    scratchbuf.ptr as _,
                                );
                            }
                        }
                    }
                }
                t_cpp += t0.elapsed();
            }
            {
                let t0 = Instant::now();
                for r in 0..nrounds {
                    for o in 0..nouts {
                        let d = unsafe {
                            std::slice::from_raw_parts_mut(outs.ptr.add((r * nouts + o) * tile), tile)
                        };
                        for i in 0..nins {
                            let s = unsafe {
                                std::slice::from_raw_parts(
                                    arena.ptr.add(r * tile * nins + i * tile),
                                    tile,
                                )
                            };
                            k.muladd(d, s, coeffs[o * nins + i]);
                        }
                    }
                }
                t_rs += t0.elapsed();
            }
        }
        let bytes = (nins * nouts * slice * reps) as f64;
        eprintln!(
            "interleaved A/B: C++ {:.1} GB/s | Rust {:.1} GB/s",
            bytes / t_cpp.as_secs_f64() / 1e9,
            bytes / t_rs.as_secs_f64() / 1e9
        );

        // hot-cache A/B: same single dst tile + single src tile in a tight
        // loop - pure ALU/frontend, no L3/DRAM traffic
        let inner = 200_000usize;
        let coeff = 0xbeefu16;
        let t0 = Instant::now();
        for _ in 0..inner {
            unsafe {
                gf16_shuffle_muladd_avx2(
                    scratch,
                    outs.ptr,
                    arena.ptr,
                    tile,
                    coeff,
                    scratchbuf.ptr as _,
                );
            }
        }
        let t_cpp_hot = t0.elapsed();
        let t0 = Instant::now();
        for _ in 0..inner {
            let d = unsafe { std::slice::from_raw_parts_mut(outs.ptr, tile) };
            let s = unsafe { std::slice::from_raw_parts(arena.ptr, tile) };
            k.muladd(d, s, coeff);
        }
        let t_rs_hot = t0.elapsed();
        let hbytes = (tile * inner) as f64;
        eprintln!(
            "hot-cache A/B: C++ {:.1} GB/s | Rust {:.1} GB/s",
            hbytes / t_cpp_hot.as_secs_f64() / 1e9,
            hbytes / t_rs_hot.as_secs_f64() / 1e9
        );

        // inline-asm clone of the C++ inner loop: identical instruction
        // stream, isolating codegen from call context
        let mut tab = Aligned::new(256);
        for (i, b) in tab.as_mut_slice().iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        let tables: &[u8; 256] = tab.as_slice().try_into().unwrap();
        let t0 = Instant::now();
        for _ in 0..inner {
            unsafe { cpp_clone_muladd(outs.ptr, arena.ptr, tile, tables) };
        }
        let t_cl = t0.elapsed();
        let t0 = Instant::now();
        for _ in 0..inner {
            unsafe { clone_muladd_8reg(outs.ptr, arena.ptr, tile, tables) };
        }
        let t_cl8 = t0.elapsed();
        eprintln!(
            "hot-cache A/B: C++ {:.1} GB/s | Rust {:.1} GB/s | asm-clone {:.1} GB/s | clone-8reg {:.1} GB/s",
            hbytes / t_cpp_hot.as_secs_f64() / 1e9,
            hbytes / t_rs_hot.as_secs_f64() / 1e9,
            hbytes / t_cl.as_secs_f64() / 1e9,
            hbytes / t_cl8.as_secs_f64() / 1e9
        );
        // dst XORs to zero across mixed-kernel phases - sanity-check with
        // a single fresh accumulate instead
        let mut sanity = Aligned::new(tile);
        unsafe { cpp_clone_muladd(sanity.ptr, arena.ptr, tile, tables) };
        assert!(sanity.as_slice().iter().any(|&b| b != 0));
    }

    #[test]
    #[ignore = "ad-hoc throughput probe - run with --ignored; too slow under qemu CI"]
    fn cpp_kernel_throughput() {
        bench_cpp_shuffle2x();
        bench_cpp_muladd_fallback();
    }
}

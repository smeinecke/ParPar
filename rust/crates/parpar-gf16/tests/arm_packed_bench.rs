//! Ad-hoc A/B probe for the aarch64 clmul fused paths: 8 contiguous src
//! streams vs one block-interleaved region (C++ INTERLEAVE_INPUTS).
//! Run with `cargo test -p parpar-gf16 --release --test arm_packed_bench -- --nocapture`.

#![cfg(target_arch = "aarch64")]

use parpar_gf16::kernel;
use std::time::Instant;

const BLK: usize = 32;

#[test]
#[ignore = "ad-hoc throughput probe - run with --ignored; too slow under qemu CI"]
fn packed_vs_contiguous() {
    let k = kernel::pick(0);
    eprintln!("kernel {} {}", k.id(), k.name());
    if k.id() != 28 && k.id() != 29 {
        eprintln!("not a clmul kernel - probe is meaningless");
        return;
    }
    let nins = 8usize;
    let len = 8192usize;
    let nouts = 16usize;
    let reps = 40usize;

    let srcs: Vec<Vec<u8>> = (0..nins)
        .map(|i| vec![(i as u8) ^ 0x5a; len])
        .collect();
    let coeffs: Vec<u16> = (0..nins).map(|i| 0x100 + i as u16 * 0x777).collect();
    // interleaved region: src k block j at 32*(k + nins*j)
    let mut region = vec![0u8; nins * len];
    for (s, src) in srcs.iter().enumerate() {
        for j in 0..len / BLK {
            region[BLK * (s + nins * j)..BLK * (s + nins * j) + BLK]
                .copy_from_slice(&src[j * BLK..(j + 1) * BLK]);
        }
    }
    let mut outs = vec![0x11u8; nouts * len];

    // A: 8 contiguous streams via muladd_multi
    let inputs: Vec<(&[u8], u16)> = srcs
        .iter()
        .zip(&coeffs)
        .map(|(s, &c)| (s.as_slice(), c))
        .collect();
    let t0 = Instant::now();
    for _ in 0..reps {
        for o in 0..nouts {
            let d = &mut outs[o * len..(o + 1) * len];
            k.muladd_multi(d, &inputs);
        }
    }
    let ta = t0.elapsed().as_secs_f64();
    let gib = reps as f64 * nouts as f64 * nins as f64 * len as f64 / (1 << 30) as f64;
    eprintln!("contiguous 8-src: {:.3}s ({:.2} GiB/s src-side)", ta, gib / ta);

    // B: interleaved region via muladd_multi_packed
    let t0 = Instant::now();
    for _ in 0..reps {
        for o in 0..nouts {
            let d = &mut outs[o * len..(o + 1) * len];
            k.muladd_multi_packed(d, &region, nins, &coeffs);
        }
    }
    let tb = t0.elapsed().as_secs_f64();
    eprintln!("interleaved 8-src: {:.3}s ({:.2} GiB/s src-side)", tb, gib / tb);
}

/// Production-shape aggregate: batch of `nins` inputs x `nouts` outputs
/// over `nrounds` dst tiles, `nthreads` worker threads - measures the
/// sustained multiply rate the pipeline actually sees.
#[test]
#[ignore = "ad-hoc throughput probe - run with --ignored; too slow under qemu CI"]
fn aggregate_packed_throughput() {
    let k = kernel::pick(0);
    eprintln!("kernel {} {}", k.id(), k.name());
    let nins = 16usize;
    let tile = 8192usize;
    let nrounds = 32usize; // ~256KB slice packed len
    let nouts = 360usize;
    for nthreads in [12usize, 8, 4] {
        let len = nrounds * tile;
        let srcs: Vec<Vec<u8>> = (0..nins).map(|i| vec![(i as u8) ^ 0x5a; len]).collect();
        let coeffs: Vec<Vec<u16>> = (0..nins)
            .map(|i| (0..nouts).map(|o| 0x100 + ((i * 31 + o * 7) % 0xf000) as u16).collect())
            .collect();
        // block-interleaved staging per group of 8, chunk-major
        // [round][group][interleaved] like proc.rs / C++ srcPtr
        let ngroups = nins.div_ceil(8);
        let blocks_per_tile = tile / BLK;
        let mut stage = vec![0u8; nrounds * nins * tile];
        for (i, src) in srcs.iter().enumerate() {
            let g = i / 8;
            let ng = (nins - g * 8).min(8);
            let kbase = g * 8 * tile + BLK * (i % 8);
            for j in 0..len / BLK {
                let r = j / blocks_per_tile;
                let off = r * nins * tile + kbase + ng * BLK * (j % blocks_per_tile);
                stage[off..off + BLK].copy_from_slice(&src[j * BLK..(j + 1) * BLK]);
            }
        }
        // dst flat [round][out][tile]
        let mut outs_flat = vec![0u8; nrounds * nouts * tile];
        // tasks: (toff, tl, o) tile-major, contiguous chunks per thread
        let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
        for t in 0..nrounds {
            for o in 0..nouts {
                tasks.push((t * tile, tile, o));
            }
        }
        let ntasks = tasks.len();
        let reps = 3usize;
        // variant A: one shared stage buffer (all threads read same lines)
        // variant B: per-thread stage copies (no shared-line reads)
        for shared in [true, false] {
            let stages: Vec<Vec<u8>> = (0..if shared { 1 } else { nthreads })
                .map(|_| stage.clone())
                .collect();
            let t0 = Instant::now();
            for _ in 0..reps {
                let k = &*k;
                let stages = &stages;
                let coeffs = &coeffs;
                let outs_addr = outs_flat.as_mut_ptr() as usize;
                std::thread::scope(|sc| {
                    let mut it = tasks.iter();
                    for t in 0..nthreads {
                        let lo = t * ntasks / nthreads;
                        let hi = (t + 1) * ntasks / nthreads;
                        let grp: Vec<_> = it.by_ref().take(hi - lo).collect();
                        if grp.is_empty() {
                            continue;
                        }
                        let outs_addr = outs_addr;
                        let stage = &stages[if shared { 0 } else { t }];
                        sc.spawn(move || unsafe {
                            for &&(toff, tl, o) in &grp {
                                let d = std::slice::from_raw_parts_mut(
                                    (outs_addr as *mut u8)
                                        .add((toff / tile) * nouts * tile + o * tile),
                                    tl,
                                );
                                let round_base = (toff / tile) * nins * tile;
                                for g in 0..ngroups {
                                    let ng = (nins - g * 8).min(8);
                                    let base = round_base + g * 8 * tile;
                                    let c8: Vec<u16> = (0..ng)
                                        .map(|kk| coeffs[g * 8 + kk][o])
                                        .collect();
                                    k.muladd_multi_packed(
                                        d,
                                        &stage[base..base + ng * tl],
                                        ng,
                                        &c8,
                                    );
                                }
                            }
                        });
                    }
                });
            }
            let dt = t0.elapsed().as_secs_f64();
            let gib = reps as f64 * nins as f64 * nouts as f64 * len as f64 / (1 << 30) as f64;
            eprintln!(
                "packed agg nins={} nouts={} thr={} shared={}: {:.2}s -> {:.2} GiB/s src-side ({:.2}/core)",
                nins,
                nouts,
                nthreads,
                shared,
                dt,
                gib / dt,
                gib / dt / nthreads as f64
            );
        }
    }
}

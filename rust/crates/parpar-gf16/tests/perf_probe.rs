//! Ad-hoc throughput probe: simulate the -s1M batch multiply across N
//! threads and report aggregate GB/s. Not a correctness test - run with
//! `cargo test -p parpar-gf16 --release --test perf_probe -- --nocapture`.

use parpar_gf16::kernel;
use std::time::Instant;

#[test]
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
}

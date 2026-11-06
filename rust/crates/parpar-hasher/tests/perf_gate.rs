//! Throughput floor for the input hasher - catches catastrophic MD5
//! regressions (the NEON-MD5 ~0.11 GiB/s class bug) without chasing
//! percent-level deltas. Prints the measured rate for CI-log trending.
//!
//! `PARPAR_PERF=0` disables (set in emulated CI). Debug builds skip:
//! only optimized code is worth gating.

use parpar_hasher::LaneProducer;
use std::time::Instant;

#[test]
fn hasher_throughput_floor() {
    if cfg!(debug_assertions) {
        eprintln!("debug build - skipping perf gate");
        return;
    }
    if std::env::var("PARPAR_PERF").map(|v| v == "0").unwrap_or(false) {
        eprintln!("PARPAR_PERF=0 - skipping perf gate");
        return;
    }

    const TOTAL: usize = 256 << 20;
    const FLOOR_MIB: f64 = if cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64") {
        128.0
    } else {
        16.0
    };

    let chunk = vec![0xabu8; 4 << 20];
    let mut best = 0.0f64;
    for _ in 0..3 {
        let mut p = LaneProducer::new(4 << 20);
        let mut states = Vec::new();
        let mut left = TOTAL;
        let t = Instant::now();
        while left > 0 {
            p.update(&chunk, &mut states);
            left -= chunk.len();
            states.clear();
        }
        let mibs = (TOTAL >> 20) as f64 / t.elapsed().as_secs_f64();
        best = best.max(mibs);
    }
    eprintln!("perf_gate: LaneProducer {best:.0} MiB/s (floor {FLOOR_MIB:.0})");
    assert!(
        best >= FLOOR_MIB,
        "hasher at {best:.0} MiB/s - below the {FLOOR_MIB:.0} MiB/s floor"
    );
}

//! Throughput floors for the GF16 kernels - a requirements layer that
//! catches catastrophic regressions (accidental scalar fallback, dead
//! SIMD dispatch, broken staging) rather than chasing percent-level
//! deltas. Floors sit several times below observed rates on the slowest
//! supported hosts, so hosted-CI noise cannot trip them; measured numbers
//! are always printed for CI-log trending.
//!
//! `PARPAR_PERF=0` disables (set in emulated CI - TCG cannot meet any
//! meaningful floor). Debug builds are skipped: only optimized code is
//! worth gating.

use parpar_gf16::kernel::{self, GfKernel};
use std::time::Instant;

fn perf_enabled() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("debug build - skipping perf gate");
        return false;
    }
    if std::env::var("PARPAR_PERF").map(|v| v == "0").unwrap_or(false) {
        eprintln!("PARPAR_PERF=0 - skipping perf gate");
        return false;
    }
    true
}

/// Floor for a kernel's muladd throughput in MiB/s of src traffic.
/// Observed: ~4-5 GiB/s per core on Neoverse-N1, tens of GiB/s on x86
/// AVX2/AVX512. 512 MiB/s is ~8x under the slowest real SIMD host and
/// still catches "fell back to byte-at-a-time" class bugs.
fn muladd_floor_mib() -> f64 {
    if cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64") {
        512.0
    } else {
        64.0
    }
}

/// Measure muladd throughput: dst ^= src*coeff over a 64 KiB region,
/// ~256 MiB of src traffic, best of 3 samples. Returns MiB/s (src-side).
fn muladd_mibs(k: &dyn GfKernel) -> f64 {
    const REGION: usize = 64 << 10;
    let target: usize = 256 << 20;
    let src = vec![0xabu8; REGION];
    let mut dst = vec![0u8; REGION];
    let mut best = 0.0f64;
    for _ in 0..3 {
        let mut left = target;
        let t = Instant::now();
        while left > 0 {
            k.muladd(&mut dst, &src, 0x1234);
            left -= REGION;
        }
        let mibs = (target >> 20) as f64 / t.elapsed().as_secs_f64();
        best = best.max(mibs);
    }
    best
}

/// Every host-available kernel is measured and reported; the auto-picked
/// kernel and at least one explicit kernel must clear the floor.
#[test]
fn kernel_throughput_floors() {
    if !perf_enabled() {
        return;
    }
    let floor = muladd_floor_mib();
    let mut best_any = 0.0f64;
    for id in 0..32 {
        let Some(k) = kernel::pick_explicit(id) else {
            continue;
        };
        let rate = muladd_mibs(&*k);
        eprintln!("perf_gate: method {id} ({}) muladd {rate:.0} MiB/s", k.name());
        best_any = best_any.max(rate);
    }
    let auto = kernel::pick(kernel::auto_id());
    let auto_rate = muladd_mibs(&*auto);
    eprintln!(
        "perf_gate: auto ({}) muladd {auto_rate:.0} MiB/s (floor {floor:.0})",
        auto.name()
    );
    assert!(
        auto_rate >= floor,
        "auto kernel {} at {auto_rate:.0} MiB/s - below the {floor:.0} MiB/s floor",
        auto.name()
    );
    assert!(
        best_any >= floor,
        "no host kernel reaches {floor:.0} MiB/s (best {best_any:.0}) - dispatch likely broken"
    );
}

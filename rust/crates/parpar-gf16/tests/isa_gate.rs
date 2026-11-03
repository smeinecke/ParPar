// Regression: pick_explicit must not return a kernel whose ISA is absent
// (the auto pick once selected ShuffleVbmi on a GFNI+AVX512-no-VBMI CPU -
// the first muladd would have executed an illegal vpermb).
#![cfg(target_arch = "x86_64")]

use parpar_gf16::kernel::pick_explicit;
use std::arch::is_x86_feature_detected as f;

#[test]
fn method14_requires_vbmi() {
    if !f!("avx512vbmi") || !f!("avx512vl") {
        assert!(
            pick_explicit(14).is_none(),
            "ShuffleVbmi must not be selectable without AVX512VBMI+VL"
        );
    }
}

#[test]
fn gated_methods_respect_isa() {
    if !f!("avx2") {
        for m in [12, 15, 19] {
            assert!(pick_explicit(m).is_none(), "method {} needs AVX2", m);
        }
    }
    if !(f!("avx512f") && f!("avx512bw") && f!("avx512vl")) {
        for m in [13, 16] {
            assert!(
                pick_explicit(m).is_none(),
                "method {} needs AVX512F/BW/VL",
                m
            );
        }
    }
    if !(f!("avx512bw") && f!("avx512vl")) {
        assert!(pick_explicit(20).is_none(), "method 20 needs AVX512BW/VL");
    }
    if !(f!("gfni") && f!("avx512f") && f!("avx512bw") && f!("avx512vl")) {
        for m in [23, 26] {
            assert!(pick_explicit(m).is_none(), "method {} needs GFNI+AVX512", m);
        }
    }
    if !f!("gfni") {
        for m in [21, 22, 24, 25] {
            assert!(pick_explicit(m).is_none(), "method {} needs GFNI", m);
        }
    }
}

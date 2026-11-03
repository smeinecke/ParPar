// audit: every pickable kernel reports the canonical C++ method name
const CANON: [&str; 33] = [
    "Auto",
    "Lookup",
    "Lookup (SSE2)",
    "3-part Lookup",
    "Shuffle (NEON)",
    "Shuffle-128 (SVE)",
    "Shuffle-128 (SVE2)",
    "Shuffle2x-128 (SVE2)",
    "Shuffle-512 (SVE2)",
    "Shuffle-128 (RVV)",
    "Shuffle (SSSE3)",
    "Shuffle (AVX)",
    "Shuffle (AVX2)",
    "Shuffle (AVX512)",
    "Shuffle (VBMI)",
    "Shuffle2x (AVX2)",
    "Shuffle2x (AVX512)",
    "Xor (SSE2)",
    "Xor-Jit (SSE2)",
    "Xor-Jit (AVX2)",
    "Xor-Jit (AVX512)",
    "Affine (GFNI)",
    "Affine (GFNI+AVX2)",
    "Affine (GFNI+AVX512)",
    "Affine2x (GFNI)",
    "Affine2x (GFNI+AVX2)",
    "Affine2x (GFNI+AVX512)",
    "Affine (BMM)",
    "CLMul (NEON)",
    "CLMul (SHA3)",
    "CLMul (SVE2)",
    "CLMul (RVV+Zvbc)",
    "INVALID",
];
#[test]
fn kernel_names_canonical() {
    for m in 0..32 {
        if let Some(k) = parpar_gf16::kernel::pick_explicit(m) {
            let n = k.name();
            // AUTO (0) resolves to the concrete method like C++ does
            let expect = if m == 0 {
                CANON[k.id() as usize]
            } else {
                CANON[m as usize]
            };
            assert_eq!(n, expect, "method {}", m);
            if m != 0 {
                assert_eq!(k.id(), m, "id mismatch {}", m);
            }
        }
    }
}

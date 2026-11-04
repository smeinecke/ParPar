//! Kernel dispatch: each GF method provides packed prepare/finish transforms
//! plus a packed multiply-accumulate core, mirroring the C++ Galois16Mul
//! interface. Regions are stored in the kernel's packed layout; get_output
//! unpacks on read so the external contract stays byte-identical.

use crate::gf;

/// Method ids (subset of the C++ GF16_METHODS enum) the native proc knows.
pub mod method {
    pub const AUTO: i32 = 0;
    pub const LOOKUP: i32 = 1;
    pub const SHUFFLE_SSE: i32 = 10;
    pub const XOR_SSE: i32 = 17;
    pub const XORJIT_SSE: i32 = 18;
    pub const SHUFFLE_AVX2: i32 = 12;
    pub const XORJIT_AVX2: i32 = 19;
    pub const XORJIT_AVX512: i32 = 20;
    pub const AFFINE_AVX512: i32 = 23;
    pub const AFFINE2X_AVX512: i32 = 26;
}

pub trait GfKernel: Send + Sync {
    fn id(&self) -> i32;
    fn name(&self) -> &'static str;
    /// Packed block size in bytes (covers block_len/2 u16 words).
    fn block_len(&self) -> usize;
    /// MethodInfo alignment (reporting only - unaligned loads are used).
    fn alignment(&self) -> usize;
    /// MethodInfo stride.
    fn stride(&self) -> usize;
    /// dst = packed form of src (dst.len() >= round_up(src.len(), block_len())).
    fn prepare(&self, dst: &mut [u8], src: &[u8]);
    /// dst = linear form of packed src (dst.len() = src words*2 = region len).
    fn finish(&self, dst: &mut [u8], src: &[u8]);
    /// dst ^= src * coeff over a packed region.
    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16);
    /// dst = src * coeff (overwrite) over a packed region.
    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16);
}

/// The generic lookup kernel: identity packing over 8-byte blocks and the
/// scalar 512-entry table multiply.
pub struct Lookup;

impl GfKernel for Lookup {
    fn id(&self) -> i32 {
        method::LOOKUP
    }
    fn name(&self) -> &'static str {
        "Lookup"
    }
    fn block_len(&self) -> usize {
        8
    }
    fn alignment(&self) -> usize {
        8
    }
    fn stride(&self) -> usize {
        8
    }
    fn prepare(&self, dst: &mut [u8], src: &[u8]) {
        let n = src.len().min(dst.len());
        dst[..n].copy_from_slice(&src[..n]);
        for b in dst[n..].iter_mut() {
            *b = 0;
        }
    }
    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
    }
    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        gf::muladd_region(dst, src, dst.len().min(src.len()), coeff);
    }
    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        gf::mul_region(dst, src, dst.len().min(src.len()), coeff);
    }
}

/// Re-labels a kernel's method id/name for cases where one Rust
/// implementation serves multiple C++ method ids (identical transform, the
/// name is C++ method metadata).
struct Renamed {
    inner: Box<dyn GfKernel>,
    id: i32,
    name: &'static str,
}

fn renamed(inner: impl GfKernel + 'static, id: i32, name: &'static str) -> Box<dyn GfKernel> {
    Box::new(Renamed {
        inner: Box::new(inner),
        id,
        name,
    })
}

impl GfKernel for Renamed {
    fn id(&self) -> i32 {
        self.id
    }
    fn name(&self) -> &'static str {
        self.name
    }
    fn block_len(&self) -> usize {
        self.inner.block_len()
    }
    fn alignment(&self) -> usize {
        self.inner.alignment()
    }
    fn stride(&self) -> usize {
        self.inner.stride()
    }
    fn prepare(&self, dst: &mut [u8], src: &[u8]) {
        self.inner.prepare(dst, src)
    }
    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        self.inner.finish(dst, src)
    }
    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        self.inner.muladd(dst, src, coeff)
    }
    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        self.inner.mul(dst, src, coeff)
    }
}

/// Pick a kernel for a method id. When the requested method has no native
/// implementation yet, the best available native kernel is used instead -
/// all kernels compute identical results, the choice only affects speed.
/// The proc reports the kernel actually used via info().
/// The method id the native auto pick resolves to on this CPU - used by
/// the solver to route `auto` (method 0) through the native kernel table.
pub fn auto_id() -> i32 {
    pick(0).id()
}

pub fn pick(method_id: i32) -> Box<dyn GfKernel> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as f;
        let avx512 = f!("avx512f") && f!("avx512bw") && f!("avx512vl");
        let vbmi = avx512 && f!("avx512vbmi");
        let gfni = f!("gfni");
        // C++ default_method: GFNI affine beats VBMI beats AVX512 beats AVX2
        if gfni && avx512 {
            return match method_id {
                21 => Box::new(crate::affine::AffineGfni),
                22 => Box::new(crate::affine::AffineAvx2),
                23 => Box::new(crate::affine::AffineAvx512),
                24 => Box::new(crate::affine::Affine2xGfni),
                25 => Box::new(crate::affine::Affine2xAvx2),
                26 => Box::new(crate::affine::Affine2xAvx512),
                14 if vbmi => Box::new(crate::shuffle_vbmi::ShuffleVbmi),
                13 => Box::new(crate::shuffle_avx512::ShuffleAvx512),
                16 => Box::new(crate::shuffle2x::Shuffle2xAvx512),
                20 => Box::new(crate::xor_avx512::XorAvx512),
                m if m == method::LOOKUP => Box::new(Lookup),
                2 => renamed(Lookup, 2, "Lookup (SSE2)"),
                3 => renamed(Lookup, 3, "3-part Lookup"),
                _ => Box::new(crate::affine::AffineAvx512),
            };
        }
        if gfni && f!("avx2") {
            return match method_id {
                21 => Box::new(crate::affine::AffineGfni),
                22 => Box::new(crate::affine::AffineAvx2),
                24 => Box::new(crate::affine::Affine2xGfni),
                25 => Box::new(crate::affine::Affine2xAvx2),
                m if m == method::LOOKUP => Box::new(Lookup),
                2 => renamed(Lookup, 2, "Lookup (SSE2)"),
                3 => renamed(Lookup, 3, "3-part Lookup"),
                _ => Box::new(crate::affine::AffineAvx2),
            };
        }
        if vbmi {
            return match method_id {
                14 => Box::new(crate::shuffle_vbmi::ShuffleVbmi),
                13 => Box::new(crate::shuffle_avx512::ShuffleAvx512),
                16 => Box::new(crate::shuffle2x::Shuffle2xAvx512),
                20 => Box::new(crate::xor_avx512::XorAvx512),
                m if m == method::LOOKUP => Box::new(Lookup),
                m if m == 2 || m == 3 => Box::new(Lookup),
                _ => Box::new(crate::shuffle_vbmi::ShuffleVbmi),
            };
        }
        // Shuffle AVX512 > Shuffle AVX2 > Lookup on capable hardware.
        if avx512 {
            return match method_id {
                13 => Box::new(crate::shuffle_avx512::ShuffleAvx512),
                16 => Box::new(crate::shuffle2x::Shuffle2xAvx512),
                20 => Box::new(crate::xor_avx512::XorAvx512),
                m if m == method::LOOKUP => Box::new(Lookup),
                m if m == 2 || m == 3 => Box::new(Lookup),
                _ => Box::new(crate::shuffle_avx512::ShuffleAvx512),
            };
        }
        if f!("avx2") {
            return match method_id {
                // lookup-sse / 3p_lookup: same byte-table math as Lookup
                // (C++ only differs in checksum/packing width)
                m if m == 2 || m == 3 => Box::new(Lookup),
                m if m == method::SHUFFLE_SSE => Box::new(crate::shuffle_sse::ShuffleSse),
                11 => renamed(crate::shuffle_sse::ShuffleSse, 11, "Shuffle (AVX)"),
                // shuffle2x-avx2 (15): dual-coefficient scheduling variant of
                // the same nibble-shuffle math - single-coeff kernel is equivalent
                m if m == 15 => Box::new(crate::shuffle2x::Shuffle2xAvx2),
                m if m == method::XOR_SSE => Box::new(crate::xor_sse::XorSse),
                m if m == method::XORJIT_SSE => {
                    Box::new(crate::xor_sse::XorJitSse(crate::xor_sse::XorSse))
                }
                m if m == method::XORJIT_AVX2 => Box::new(crate::xor_avx2::XorAvx2),
                m if m == method::LOOKUP => Box::new(Lookup),
                // Shuffle AVX2 is the fastest native kernel - auto and
                // unimplemented methods land here.
                _ => Box::new(crate::shuffle_avx2::ShuffleAvx2),
            };
        }
        let _ = method_id;
        return Box::new(Lookup);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64; CLMul needs +pmull (the aes feature)
        // and is the fastest kernel where available.
        let pmull = std::arch::is_aarch64_feature_detected!("aes");
        let sha3 = pmull && std::arch::is_aarch64_feature_detected!("sha3");
        return match method_id {
            m if m == 4 => Box::new(crate::shuffle_neon::ShuffleNeon),
            m if m == 28 && pmull => Box::new(crate::clmul_neon::ClmulNeon),
            m if m == 29 && sha3 => Box::new(crate::clmul_neon::ClmulSha3),
            m if m == method::LOOKUP => Box::new(Lookup),
            m if m == 2 || m == 3 => Box::new(Lookup),
            // C++ prefers CLMul SHA3 > NEON > Shuffle for >3 inputs
            _ if sha3 => Box::new(crate::clmul_neon::ClmulSha3),
            _ if pmull => Box::new(crate::clmul_neon::ClmulNeon),
            _ => Box::new(crate::shuffle_neon::ShuffleNeon),
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = method_id;
        Box::new(Lookup)
    }
}

/// Explicit-method pick: returns None when the requested method is not
/// implemented natively or its ISA is unavailable (mirrors the C++ proc's
/// METHOD_REQUIRES init failure). `pick` keeps the permissive fallback for
/// coexistence paths; new_proc uses this for parity with C++ diagnostics.
pub fn pick_explicit(method_id: i32) -> Option<Box<dyn GfKernel>> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as f;
        let avx2 = f!("avx2");
        let sse2 = f!("sse2");
        let ssse3 = f!("ssse3");
        let avx = f!("avx");
        return match method_id {
            0 => Some(pick(0)),
            m if m == method::LOOKUP => Some(Box::new(Lookup)),
            2 => Some(renamed(Lookup, 2, "Lookup (SSE2)")),
            3 => Some(renamed(Lookup, 3, "3-part Lookup")),
            // lookup-sse / 3p_lookup: equivalent byte-table math in the C++
            // impls, no extra ISA requirements
            m if m == 2 || m == 3 => Some(Box::new(Lookup)),
            m if m == method::SHUFFLE_SSE && ssse3 => {
                Some(Box::new(crate::shuffle_sse::ShuffleSse))
            }
            m if m == 11 && avx => {
                Some(renamed(crate::shuffle_sse::ShuffleSse, 11, "Shuffle (AVX)"))
            }
            m if m == method::SHUFFLE_AVX2 && avx2 => {
                Some(Box::new(crate::shuffle_avx2::ShuffleAvx2))
            }
            m if m == 15 && avx2 => Some(Box::new(crate::shuffle2x::Shuffle2xAvx2)),
            16 if f!("avx512f") && f!("avx512bw") => {
                Some(Box::new(crate::shuffle2x::Shuffle2xAvx512))
            }
            m if m == method::XOR_SSE && sse2 => Some(Box::new(crate::xor_sse::XorSse)),
            m if m == method::XORJIT_SSE && sse2 => {
                Some(Box::new(crate::xor_sse::XorJitSse(crate::xor_sse::XorSse)))
            }
            m if m == method::XORJIT_AVX2 && avx2 => Some(Box::new(crate::xor_avx2::XorAvx2)),
            // Shuffle (AVX512): C++ requires AVX512F+VL+BW
            13 if f!("avx512f") && f!("avx512bw") && f!("avx512vl") => {
                Some(Box::new(crate::shuffle_avx512::ShuffleAvx512))
            }
            // Shuffle (VBMI): C++ requires AVX512VBMI+VL
            14 if f!("avx512vbmi") && f!("avx512vl") => {
                Some(Box::new(crate::shuffle_vbmi::ShuffleVbmi))
            }
            // Xor-Jit (AVX512): C++ requires AVX512BW+VL
            20 if f!("avx512bw") && f!("avx512vl") => Some(Box::new(crate::xor_avx512::XorAvx512)),
            // Affine family: GFNI base + respective vector width
            21 | 24 if f!("gfni") && ssse3 => Some(if method_id == 21 {
                Box::new(crate::affine::AffineGfni) as Box<dyn GfKernel>
            } else {
                Box::new(crate::affine::Affine2xGfni)
            }),
            22 | 25 if f!("gfni") && avx2 => Some(if method_id == 22 {
                Box::new(crate::affine::AffineAvx2) as Box<dyn GfKernel>
            } else {
                Box::new(crate::affine::Affine2xAvx2)
            }),
            23 | 26 if f!("gfni") && f!("avx512f") && f!("avx512bw") && f!("avx512vl") => {
                Some(if method_id == 23 {
                    Box::new(crate::affine::AffineAvx512) as Box<dyn GfKernel>
                } else {
                    Box::new(crate::affine::Affine2xAvx512)
                })
            }
            _ => None,
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        match method_id {
            0 => Some(pick(0)),
            m if m == method::LOOKUP => Some(Box::new(Lookup)),
            2 => Some(renamed(Lookup, 2, "Lookup (SSE2)")),
            3 => Some(renamed(Lookup, 3, "3-part Lookup")),
            m if m == 2 || m == 3 => Some(Box::new(Lookup)),
            // NEON is baseline on aarch64
            4 => Some(Box::new(crate::shuffle_neon::ShuffleNeon)),
            // CLMul needs +pmull (aes feature)
            28 if std::arch::is_aarch64_feature_detected!("aes") => {
                Some(Box::new(crate::clmul_neon::ClmulNeon))
            }
            29 if std::arch::is_aarch64_feature_detected!("aes")
                && std::arch::is_aarch64_feature_detected!("sha3") =>
            {
                Some(Box::new(crate::clmul_neon::ClmulSha3))
            }
            _ => None,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        match method_id {
            0 => Some(pick(0)),
            m if m == method::LOOKUP => Some(Box::new(Lookup)),
            // report the method-table names like the other arch fallbacks
            // so kernel_meta parity holds on scalar-only targets
            2 => Some(renamed(Lookup, 2, "Lookup (SSE2)")),
            3 => Some(renamed(Lookup, 3, "3-part Lookup")),
            _ => None,
        }
    }
}

/// Static per-method info, mirroring `Galois16Mul::info()` on x86-64
/// (id, name, alignment, stride, idealChunkSize, idealInputMultiple).
/// SVE/RVV entries report stride 0, matching `gf16_sve_get_size`/
/// `gf16_rvv_get_size` returning 0 on amd64.
const METHOD_INFO: &[(i32, &str, u64, u64, u64, u32)] = &[
    (1, "Lookup", 8, 8, 32 * 1024, 1),
    (2, "Lookup (SSE2)", 16, 16, 32 * 1024, 1),
    (3, "3-part Lookup", 8, 8, 32 * 1024, 1),
    (4, "Shuffle (NEON)", 32, 32, 16 * 1024, 1),
    (5, "Shuffle-128 (SVE)", 16, 0, 16 * 1024, 3),
    (6, "Shuffle-128 (SVE2)", 16, 0, 16 * 1024, 3),
    (7, "Shuffle2x-128 (SVE2)", 32, 0, 8 * 1024, 6),
    (8, "Shuffle-512 (SVE2)", 64, 0, 8 * 1024, 4),
    (9, "Shuffle-128 (RVV)", 16, 0, 16 * 1024, 3),
    (10, "Shuffle (SSSE3)", 16, 32, 16 * 1024, 1),
    (11, "Shuffle (AVX)", 16, 32, 16 * 1024, 1),
    (12, "Shuffle (AVX2)", 32, 64, 8 * 1024, 1),
    (13, "Shuffle (AVX512)", 64, 128, 8 * 1024, 3),
    (14, "Shuffle (VBMI)", 64, 128, 8 * 1024, 4),
    (15, "Shuffle2x (AVX2)", 32, 32, 8 * 1024, 2),
    (16, "Shuffle2x (AVX512)", 64, 64, 8 * 1024, 6),
    (17, "Xor (SSE2)", 16, 256, 32 * 1024, 1),
    (18, "Xor-Jit (SSE2)", 16, 256, 64 * 1024, 1),
    (19, "Xor-Jit (AVX2)", 32, 512, 128 * 1024, 1),
    (20, "Xor-Jit (AVX512)", 64, 1024, 48 * 1024, 6),
    (21, "Affine (GFNI)", 16, 32, 8 * 1024, 3),
    (22, "Affine (GFNI+AVX2)", 32, 64, 4 * 1024, 3),
    (23, "Affine (GFNI+AVX512)", 64, 128, 4 * 1024, 6),
    (24, "Affine2x (GFNI)", 16, 16, 8 * 1024, 6),
    (25, "Affine2x (GFNI+AVX2)", 32, 32, 4 * 1024, 6),
    (26, "Affine2x (GFNI+AVX512)", 64, 64, 4 * 1024, 12),
    (27, "Affine (BMM)", 64, 64, 4 * 1024, 12),
    (28, "CLMul (NEON)", 32, 32, 8 * 1024, 3),
    (29, "CLMul (SHA3)", 32, 32, 8 * 1024, 3),
    (30, "CLMul (SVE2)", 16, 0, 8 * 1024, 8),
    (31, "CLMul (RVV+Zvbc)", 16, 0, 8 * 1024, 12),
];

/// `Galois16Mul::info(method)` - static method info. AUTO resolves through
/// the native kernel table so the reported method names what new_proc runs.
/// Out-of-range methods report Lookup, mirroring the C++ switch default.
pub fn method_info(method: i32) -> parpar_backend::MethodInfo {
    let resolved = if method == 0 { auto_id() } else { method };
    for &(id, name, alignment, stride, chunk, grouping) in METHOD_INFO {
        if id == resolved {
            return parpar_backend::MethodInfo {
                id,
                name: name.to_string(),
                alignment,
                stride,
                target_chunk: chunk,
                target_grouping: grouping,
            };
        }
    }
    let l = &METHOD_INFO[0];
    parpar_backend::MethodInfo {
        id: 1,
        name: l.1.to_string(),
        alignment: l.2,
        stride: l.3,
        target_chunk: l.4,
        target_grouping: l.5,
    }
}

/// Round a byte count up to the kernel's packed block multiple.
pub fn packed_len(k: &dyn GfKernel, n: usize) -> usize {
    let b = k.block_len();
    n.div_ceil(b) * b
}

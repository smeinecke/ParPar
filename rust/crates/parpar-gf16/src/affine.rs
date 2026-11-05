//! GFNI affine kernels - port of gf16_affine_gfni.c / _avx2.c / _avx512.c
//! (methods 21-23) and gf16_affine2x_x86.h (methods 24-26).
//!
//! The 1x kernels use the shuffle-family packed layout
//! [hi bytes][lo bytes] and four 8x8 GF(2) matrices:
//!   lo_out = M_ll(w.lo) ^ M_lh(w.hi)   hi_out = M_hl(w.lo) ^ M_hh(w.hi)
//! where M_ll(x)=lo(c*x), M_lh(x)=lo(c*(x<<8)), M_hl(x)=hi(c*x),
//! M_hh(x)=hi(c*(x<<8)). gf2p8affine applies a per-qword matrix:
//! out bit n of byte j = parity(A.qword[j/8].byte[7-n] & byte j).
//!
//! The 2x kernels keep per-lane [8 lo|8 hi] and use only two matrix
//! registers: "norm" = [M_ll | M_hh] per lane (same-plane products),
//! "swap" = [M_hl | M_lh] (cross-plane products lane-swapped into place).

#![cfg(target_arch = "x86_64")]

use crate::gf::{gf16_mul, gf16_mul2};
use crate::kernel::GfKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 8x8 GF(2) matrix for f: u8 -> u8, in gf2p8affine qword form:
/// byte[7-n] = mask of input bits producing output bit n.
fn mat_of(f: impl Fn(u8) -> u8) -> u64 {
    let mut q = [0u8; 8];
    for k in 0..8 {
        let v = f(1u8 << k);
        for n in 0..8 {
            if (v >> n) & 1 == 1 {
                q[7 - n] |= 1 << k;
            }
        }
    }
    u64::from_le_bytes(q)
}

fn mats(coeff: u16) -> (u64, u64, u64, u64) {
    // the matrices only sample powers of 2: c_j = coeff*2^j for j in
    // 0..16 via an xtime chain replaces the bit-serial gf16_mul calls.
    let mut c = [0u16; 16];
    c[0] = coeff;
    for j in 1..16 {
        c[j] = gf16_mul2(c[j - 1]);
    }
    let bit = |x: u8| x.trailing_zeros() as usize;
    (
        mat_of(|x| c[bit(x)] as u8),
        mat_of(|x| (c[bit(x)] >> 8) as u8),
        mat_of(|x| c[8 + bit(x)] as u8),
        mat_of(|x| (c[8 + bit(x)] >> 8) as u8),
    )
}

// ---------- 1x kernels: [hi][lo] planes ----------

fn prepare_scalar_1x(dst: &mut [u8], src: &[u8], plane: usize) {
    for b in dst.iter_mut() {
        *b = 0;
    }
    // src reads are linear; bound words to what dst's planes hold
    let words = (src.len() / 2)
        .min(plane)
        .min(dst.len().saturating_sub(plane));
    for i in 0..words {
        dst[i] = src[2 * i + 1];
        dst[plane + i] = src[2 * i];
    }
    if src.len() % 2 == 1 && plane + words < dst.len() {
        dst[plane + words] = src[src.len() - 1];
    }
}

fn finish_scalar_1x(dst: &mut [u8], src: &[u8], plane: usize) {
    // lo reads at src[plane+i] bound words to src.len()-plane
    let words = (dst.len() / 2)
        .min(plane)
        .min(src.len().saturating_sub(plane));
    for i in 0..words {
        dst[2 * i] = src[plane + i];
        dst[2 * i + 1] = src[i];
    }
    if dst.len() % 2 == 1 && plane + words < src.len() {
        dst[dst.len() - 1] = src[plane + words];
    }
}

fn muladd_scalar_1x(dst: &mut [u8], src: &[u8], coeff: u16, add: bool, plane: usize) {
    let words = src.len().min(dst.len()).saturating_sub(plane).min(plane);
    for i in 0..words {
        let w = (src[i] as u16) << 8 | src[plane + i] as u16;
        let p = gf16_mul(w, coeff);
        let d = ((dst[i] as u16) << 8 | dst[plane + i] as u16) ^ p;
        let v = if add { d } else { p };
        dst[i] = (v >> 8) as u8;
        dst[plane + i] = v as u8;
    }
}

macro_rules! affine1x {
    ($modn:ident, $name:ident, $id:expr, $nm:literal, $blk:expr, $plane:expr, $modp:ident,
     $load:ident, $store:ident, $xor:ident, $aff:ident, $set1q:ident, $feat:literal) => {
        mod $modn {
            use super::*;
            pub struct $name;
            const BLK: usize = $blk;
            const PLANE: usize = $plane;

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn muladd_b(dst: *mut u8, src: *const u8, m: &(u64, u64, u64, u64), add: bool) {
                let (m_ll, m_hl, m_lh, m_hh) = *m;
                let ta = $load(src as _); // hi bytes
                let tb = $load(src.add(PLANE) as _); // lo bytes
                let mut tpl = $xor(
                    $aff(ta, $set1q(m_lh as i64), 0),
                    $aff(tb, $set1q(m_ll as i64), 0),
                );
                let mut tph = $xor(
                    $aff(ta, $set1q(m_hh as i64), 0),
                    $aff(tb, $set1q(m_hl as i64), 0),
                );
                if add {
                    tph = $xor(tph, $load(dst as _));
                    tpl = $xor(tpl, $load(dst.add(PLANE) as _));
                }
                $store(dst as _, tph);
                $store(dst.add(PLANE) as _, tpl);
            }

            // Per-call loops must be #[target_feature] fns (see shuffle_avx2).
            #[target_feature(enable = $feat)]
            unsafe fn prepare_run(dst: &mut [u8], src: &[u8]) {
                let mut i = 0;
                while i + BLK <= src.len() {
                    crate::$modp::prepare_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
                    i += BLK;
                }
                if i < src.len() {
                    prepare_scalar_1x(&mut dst[i..], &src[i..], PLANE);
                } else {
                    for b in dst[i..].iter_mut() {
                        *b = 0;
                    }
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn finish_run(dst: &mut [u8], src: &[u8]) {
                let mut i = 0;
                while i + BLK <= dst.len() {
                    crate::$modp::finish_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
                    i += BLK;
                }
                if i < dst.len() {
                    finish_scalar_1x(&mut dst[i..], &src[i..], PLANE);
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn muladd_run(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
                let m = mats(coeff);
                let mut i = 0;
                while i + BLK <= src.len() {
                    muladd_b(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &m, add);
                    i += BLK;
                }
                if i < src.len() {
                    muladd_scalar_1x(&mut dst[i..], &src[i..], coeff, add, PLANE);
                }
            }

            impl GfKernel for $name {
                fn id(&self) -> i32 {
                    $id
                }
                fn name(&self) -> &'static str {
                    $nm
                }
                fn block_len(&self) -> usize {
                    BLK
                }
                fn alignment(&self) -> usize {
                    PLANE
                }
                fn stride(&self) -> usize {
                    BLK
                }

                fn prepare(&self, dst: &mut [u8], src: &[u8]) {
                    unsafe { prepare_run(dst, src) }
                }

                fn finish(&self, dst: &mut [u8], src: &[u8]) {
                    unsafe { finish_run(dst, src) }
                }

                fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
                    unsafe { muladd_run(dst, src, coeff, true) }
                }

                fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
                    unsafe { muladd_run(dst, src, coeff, false) }
                }
            }
        }
        pub use $modn::$name;
    };
}

affine1x!(
    a1x_gfni,
    AffineGfni,
    21,
    "Affine (GFNI)",
    32,
    16,
    shuffle_sse,
    _mm_loadu_si128,
    _mm_storeu_si128,
    _mm_xor_si128,
    _mm_gf2p8affine_epi64_epi8,
    _mm_set1_epi64x,
    "gfni,ssse3"
);
affine1x!(
    a1x_avx2,
    AffineAvx2,
    22,
    "Affine (GFNI+AVX2)",
    64,
    32,
    shuffle_avx2,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_xor_si256,
    _mm256_gf2p8affine_epi64_epi8,
    _mm256_set1_epi64x,
    "gfni,avx2"
);
affine1x!(
    a1x_avx512,
    AffineAvx512,
    23,
    "Affine (GFNI+AVX512)",
    128,
    64,
    shuffle_avx512,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_xor_si512,
    _mm512_gf2p8affine_epi64_epi8,
    _mm512_set1_epi64,
    "gfni,avx512f,avx512bw"
);

// ---------- 2x kernels: per-lane [8 lo|8 hi] ----------

// words a [8 lo|8 hi]-lane region can hold: 8 per full 16B lane, plus a
// partial lane's lo bytes beyond its 8-byte lo half
fn lane_words(len: usize) -> usize {
    (len / 16) * 8 + (len % 16).saturating_sub(8).min(8)
}

fn prepare_scalar_2x(dst: &mut [u8], src: &[u8]) {
    for b in dst.iter_mut() {
        *b = 0;
    }
    // src reads are linear; bound words to what dst's lanes hold
    let words = (src.len() / 2).min(lane_words(dst.len()));
    for i in 0..words {
        let g = i / 8;
        dst[g * 16 + i % 8] = src[2 * i];
        dst[g * 16 + 8 + i % 8] = src[2 * i + 1];
    }
    if src.len() % 2 == 1 {
        let g = words / 8;
        let idx = g * 16 + words % 8;
        if idx < dst.len() {
            dst[idx] = src[src.len() - 1];
        }
    }
}

fn finish_scalar_2x(dst: &mut [u8], src: &[u8]) {
    // hi reads at lane+8 bound words to what src's lanes hold
    let words = (dst.len() / 2).min(lane_words(src.len()));
    for i in 0..words {
        let g = i / 8;
        dst[2 * i] = src[g * 16 + i % 8];
        dst[2 * i + 1] = src[g * 16 + 8 + i % 8];
    }
    if dst.len() % 2 == 1 {
        let g = words / 8;
        let idx = g * 16 + words % 8;
        if idx < src.len() {
            dst[dst.len() - 1] = src[idx];
        }
    }
}

fn muladd_scalar_2x(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    // per-lane [8 lo | 8 hi]: a word needs both bytes, so a lane yields
    // min(8, lane_len - 8) words. Clamp to words readable from both slices.
    let words = lane_words(dst.len().min(src.len()));
    for i in 0..words {
        let g = i / 8;
        let li = g * 16 + i % 8;
        let hi = li + 8;
        let w = (src[hi] as u16) << 8 | src[li] as u16;
        let p = gf16_mul(w, coeff);
        let d = ((dst[hi] as u16) << 8 | dst[li] as u16) ^ p;
        let v = if add { d } else { p };
        dst[li] = v as u8;
        dst[hi] = (v >> 8) as u8;
    }
}

// per-lane byte order for separate_low_high: [8 lo | 8 hi] of the lane's words
fn sep_idx(v: &mut [i8; 64], lanes: usize) {
    for l in 0..lanes {
        for b in 0..16 {
            v[l * 16 + b] = if b < 8 {
                (2 * b) as i8
            } else {
                (2 * (b - 8) + 1) as i8
            };
        }
    }
}

macro_rules! affine2x {
    ($modn:ident, $name:ident, $id:expr, $nm:literal, $blk:expr, $lanes:expr,
     $load:ident, $store:ident, $xor:ident, $aff:ident, $shuf8:ident, $dshuf:ident,
     $mknorm:expr, $mkswap:expr, $feat:literal) => {
        mod $modn {
            use super::*;
            pub struct $name;
            const BLK: usize = $blk;

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn prepare_b(dst: *mut u8, src: *const u8) {
                let v = $load(src as _);
                let mut idx = [0i8; 64];
                sep_idx(&mut idx, $lanes);
                let idx = $load(idx.as_ptr() as _);
                $store(dst as _, $shuf8(v, idx));
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn finish_b(dst: *mut u8, src: *const u8) {
                let v = $load(src as _);
                // interleave per lane: [lo0,hi0,lo1,hi1,...]
                let mut idx = [0i8; 64];
                for l in 0..$lanes {
                    for b in 0..16 {
                        idx[l * 16 + b] = if b % 2 == 0 {
                            (b / 2) as i8
                        } else {
                            (8 + b / 2) as i8
                        };
                    }
                }
                let idx = $load(idx.as_ptr() as _);
                $store(dst as _, $shuf8(v, idx));
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn muladd_b(dst: *mut u8, src: *const u8, m: &(u64, u64, u64, u64), add: bool) {
                let (m_ll, m_hl, m_lh, m_hh) = *m;
                let data = $load(src as _);
                let norm = $mknorm(m_ll, m_hh); // per qword pair [lo mat | hi mat]
                let swap = $mkswap(m_hl, m_lh);
                let mut res = $aff(data, norm, 0);
                let mut sw = $aff(data, swap, 0);
                // swap the qword halves within each lane -> cross products align
                sw = $dshuf(sw);
                res = $xor(res, sw);
                if add {
                    res = $xor(res, $load(dst as _));
                }
                $store(dst as _, res);
            }

            // Per-call loops must be #[target_feature] fns (see shuffle_avx2).
            #[target_feature(enable = $feat)]
            unsafe fn prepare_run(dst: &mut [u8], src: &[u8]) {
                let mut i = 0;
                while i + BLK <= src.len() {
                    prepare_b(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
                    i += BLK;
                }
                if i < src.len() {
                    prepare_scalar_2x(&mut dst[i..], &src[i..]);
                } else {
                    for b in dst[i..].iter_mut() {
                        *b = 0;
                    }
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn finish_run(dst: &mut [u8], src: &[u8]) {
                let mut i = 0;
                while i + BLK <= dst.len() {
                    finish_b(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
                    i += BLK;
                }
                if i < dst.len() {
                    finish_scalar_2x(&mut dst[i..], &src[i..]);
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn muladd_run(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
                let m = mats(coeff);
                let mut i = 0;
                while i + BLK <= src.len() {
                    muladd_b(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &m, add);
                    i += BLK;
                }
                if i < src.len() {
                    muladd_scalar_2x(&mut dst[i..], &src[i..], coeff, add);
                }
            }

            impl GfKernel for $name {
                fn id(&self) -> i32 {
                    $id
                }
                fn name(&self) -> &'static str {
                    $nm
                }
                fn block_len(&self) -> usize {
                    BLK
                }
                fn alignment(&self) -> usize {
                    BLK / $lanes
                }
                fn stride(&self) -> usize {
                    BLK
                }

                fn prepare(&self, dst: &mut [u8], src: &[u8]) {
                    unsafe { prepare_run(dst, src) }
                }

                fn finish(&self, dst: &mut [u8], src: &[u8]) {
                    unsafe { finish_run(dst, src) }
                }

                fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
                    unsafe { muladd_run(dst, src, coeff, true) }
                }

                fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
                    unsafe { muladd_run(dst, src, coeff, false) }
                }
            }
        }
        pub use $modn::$name;
    };
}

#[target_feature(enable = "gfni,ssse3")]
unsafe fn qswap128(v: __m128i) -> __m128i {
    _mm_shuffle_epi32(v, 0b01_00_11_10)
}
#[target_feature(enable = "gfni,avx2")]
unsafe fn qswap256(v: __m256i) -> __m256i {
    _mm256_shuffle_epi32(v, 0b01_00_11_10)
}
#[target_feature(enable = "gfni,avx512f,avx512bw")]
unsafe fn qswap512(v: __m512i) -> __m512i {
    _mm512_shuffle_epi32(v, 0b01_00_11_10)
}

affine2x!(
    a2x_gfni,
    Affine2xGfni,
    24,
    "Affine2x (GFNI)",
    16,
    1,
    _mm_loadu_si128,
    _mm_storeu_si128,
    _mm_xor_si128,
    _mm_gf2p8affine_epi64_epi8,
    _mm_shuffle_epi8,
    qswap128,
    |a: u64, b: u64| _mm_set_epi64x(b as i64, a as i64),
    |a: u64, b: u64| _mm_set_epi64x(b as i64, a as i64),
    "gfni,ssse3"
);
affine2x!(
    a2x_avx2,
    Affine2xAvx2,
    25,
    "Affine2x (GFNI+AVX2)",
    32,
    2,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_xor_si256,
    _mm256_gf2p8affine_epi64_epi8,
    _mm256_shuffle_epi8,
    qswap256,
    |a: u64, b: u64| _mm256_set_epi64x(b as i64, a as i64, b as i64, a as i64),
    |a: u64, b: u64| _mm256_set_epi64x(b as i64, a as i64, b as i64, a as i64),
    "gfni,avx2"
);
affine2x!(
    a2x_avx512,
    Affine2xAvx512,
    26,
    "Affine2x (GFNI+AVX512)",
    64,
    4,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_xor_si512,
    _mm512_gf2p8affine_epi64_epi8,
    _mm512_shuffle_epi8,
    qswap512,
    |a: u64, b: u64| _mm512_set_epi64(
        b as i64, a as i64, b as i64, a as i64, b as i64, a as i64, b as i64, a as i64
    ),
    |a: u64, b: u64| _mm512_set_epi64(
        b as i64, a as i64, b as i64, a as i64, b as i64, a as i64, b as i64, a as i64
    ),
    "gfni,avx512f,avx512bw"
);

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(buf: &mut [u8], seed: u64) {
        let mut rng = seed;
        for (i, b) in buf.iter_mut().enumerate() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            *b = (rng >> 33) as u8;
        }
    }

    fn check(k: &dyn GfKernel, blk: usize) {
        let mut src = [0u8; 512];
        fill(&mut src, 0x999);
        let n = blk * 2;
        let mut packed = vec![0u8; n];
        k.prepare(&mut packed, &src[..n]);
        // roundtrip
        let mut rt = vec![0u8; n];
        k.finish(&mut rt, &packed);
        assert_eq!(&rt[..], &src[..n], "roundtrip");
        for coeff in [1u16, 2, 0x1234, 0xffff, 0x8001, 0x00ff, 0x4000] {
            let mut dst = vec![0u8; n];
            let mut out = vec![0u8; n];
            k.mul(&mut dst, &packed, coeff);
            k.finish(&mut out, &dst);
            for w in 0..(n / 2) {
                let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
                let ow = (out[2 * w + 1] as u16) << 8 | out[2 * w] as u16;
                assert_eq!(ow, gf16_mul(sw, coeff), "w{} coeff {:x}", w, coeff);
            }
        }
        // muladd accumulation
        let mut dst = vec![0u8; n];
        k.prepare(&mut dst, &src[..n]);
        k.muladd(&mut dst, &packed, 0xbeef);
        let mut out = vec![0u8; n];
        k.finish(&mut out, &dst);
        for w in 0..(n / 2) {
            let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
            let want = sw ^ gf16_mul(sw, 0xbeef);
            let ow = (out[2 * w + 1] as u16) << 8 | out[2 * w] as u16;
            assert_eq!(ow, want, "add w{}", w);
        }
    }

    #[test]
    fn gfni_1x() {
        if !std::arch::is_x86_feature_detected!("gfni") {
            return;
        }
        check(&AffineGfni, 32);
    }

    #[test]
    fn avx2_1x() {
        if !(std::arch::is_x86_feature_detected!("gfni")
            && std::arch::is_x86_feature_detected!("avx2"))
        {
            return;
        }
        check(&AffineAvx2, 64);
    }

    #[test]
    fn avx512_1x() {
        if !(std::arch::is_x86_feature_detected!("gfni")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        check(&AffineAvx512, 128);
    }

    #[test]
    fn gfni_2x() {
        if !std::arch::is_x86_feature_detected!("gfni") {
            return;
        }
        check(&Affine2xGfni, 16);
    }

    #[test]
    fn avx2_2x() {
        if !(std::arch::is_x86_feature_detected!("gfni")
            && std::arch::is_x86_feature_detected!("avx2"))
        {
            return;
        }
        check(&Affine2xAvx2, 32);
    }

    #[test]
    fn avx512_2x() {
        if !(std::arch::is_x86_feature_detected!("gfni")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        check(&Affine2xAvx512, 64);
    }
}

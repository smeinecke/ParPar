//! AVX512 shuffle kernel - port of gf16_shuffle_avx512.c /
//! gf16_shuffle_x86.h (MWORD_SIZE=64 variant).
//!
//! Packed block: 128 bytes covering 64 u16 words, stored as
//! [64 high bytes][64 low bytes] (self-consistent layout like the AVX2
//! kernel - internal lane order is free as long as prepare/finish
//! agree).
//!
//! Multiply: same 4x16-entry nibble-table decomposition as the other
//! shuffle kernels; _mm512_shuffle_epi8 is lane-restricted like pshufb
//! so tables broadcast to all four 128-bit lanes.

#![cfg(target_arch = "x86_64")]

use crate::gf::{gf16_mul, gf16_mul2};
use crate::kernel::GfKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct ShuffleAvx512;

const BLK: usize = 128; // packed block: 64 u16 words

/// Eight 16-entry pshufb tables for a coefficient (low/high byte planes x4).
struct Tables {
    lo: [__m512i; 4],
    hi: [__m512i; 4],
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn make_tables(coeff: u16) -> Tables {
    // table k entry x = coeff*(x<<4k) = c_k*x, c_k = coeff*2^(4k); c*x for
    // 4-bit x is the XOR of xtime^j(c) over set bits j of x. Avoids the
    // bit-serial gf16_mul loop per entry.
    let b = |c: u16| -> ([u8; 16], [u8; 16]) {
        let d1 = gf16_mul2(c);
        let d2 = gf16_mul2(d1);
        let d3 = gf16_mul2(d2);
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for x in 1..16usize {
            let mut p = 0u16;
            if x & 1 != 0 {
                p ^= c;
            }
            if x & 2 != 0 {
                p ^= d1;
            }
            if x & 4 != 0 {
                p ^= d2;
            }
            if x & 8 != 0 {
                p ^= d3;
            }
            lo[x] = p as u8;
            hi[x] = (p >> 8) as u8;
        }
        (lo, hi)
    };
    let v = |pair: ([u8; 16], [u8; 16])| -> (__m512i, __m512i) {
        (
            _mm512_broadcast_i32x4(_mm_loadu_si128(pair.0.as_ptr() as _)),
            _mm512_broadcast_i32x4(_mm_loadu_si128(pair.1.as_ptr() as _)),
        )
    };
    let c1 = gf16_mul2(gf16_mul2(gf16_mul2(gf16_mul2(coeff))));
    let c2 = gf16_mul2(gf16_mul2(gf16_mul2(gf16_mul2(c1))));
    let c3 = gf16_mul2(gf16_mul2(gf16_mul2(gf16_mul2(c2))));
    let (l0, h0) = v(b(coeff));
    let (l1, h1) = v(b(c1));
    let (l2, h2) = v(b(c2));
    let (l3, h3) = v(b(c3));
    Tables {
        lo: [l0, l1, l2, l3],
        hi: [h0, h1, h2, h3],
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn prepare_block(dst: *mut u8, src: *const u8) {
    let a = _mm512_loadu_si512(src as _); // words 0..31
    let b = _mm512_loadu_si512(src.add(64) as _); // words 32..63
                                                  // per-lane pshufb: gather each lane's 8 lo/hi bytes into its low qword,
                                                  // then one vpermt2q collects the four lane-qwords of a and b
    let lo_idx = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8,
        6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14,
        12, 10, 8, 6, 4, 2, 0,
    );
    let hi_idx = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9,
        7, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 15,
        13, 11, 9, 7, 5, 3, 1,
    );
    let gather = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    let a_lo = _mm512_shuffle_epi8(a, lo_idx);
    let b_lo = _mm512_shuffle_epi8(b, lo_idx);
    let a_hi = _mm512_shuffle_epi8(a, hi_idx);
    let b_hi = _mm512_shuffle_epi8(b, hi_idx);
    let lo = _mm512_permutex2var_epi64(a_lo, gather, b_lo);
    let hi = _mm512_permutex2var_epi64(a_hi, gather, b_hi);
    _mm512_storeu_si512(dst as _, hi);
    _mm512_storeu_si512(dst.add(64) as _, lo);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn finish_block(dst: *mut u8, src: *const u8) {
    let hi = _mm512_loadu_si512(src as _); // hi bytes, qword j = words 8j..8j+7
    let lo = _mm512_loadu_si512(src.add(64) as _);
    // duplicate plane qword j into lane j's low half, then byte-interleave
    let idx_lo = _mm512_set_epi64(0, 3, 0, 2, 0, 1, 0, 0);
    let idx_hi = _mm512_set_epi64(0, 7, 0, 6, 0, 5, 0, 4);
    let lo_x = _mm512_permutexvar_epi64(idx_lo, lo);
    let hi_x = _mm512_permutexvar_epi64(idx_lo, hi);
    let out0 = _mm512_unpacklo_epi8(lo_x, hi_x); // words 0..31
    let lo_y = _mm512_permutexvar_epi64(idx_hi, lo);
    let hi_y = _mm512_permutexvar_epi64(idx_hi, hi);
    let out1 = _mm512_unpacklo_epi8(lo_y, hi_y); // words 32..63
    _mm512_storeu_si512(dst as _, out0);
    _mm512_storeu_si512(dst.add(64) as _, out1);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn muladd_block(dst: *mut u8, src: *const u8, t: &Tables, add: bool) {
    let nmask = _mm512_set1_epi8(0x0f);
    let hi = _mm512_loadu_si512(src as _);
    let lo = _mm512_loadu_si512(src.add(64) as _);
    let hn = _mm512_and_si512(hi, nmask);
    let hnh = _mm512_and_si512(_mm512_srli_epi16(hi, 4), nmask);
    let ln = _mm512_and_si512(lo, nmask);
    let lnh = _mm512_and_si512(_mm512_srli_epi16(lo, 4), nmask);
    let mut tpl = _mm512_shuffle_epi8(t.lo[0], ln);
    let mut tph = _mm512_shuffle_epi8(t.hi[0], ln);
    tpl = _mm512_xor_si512(tpl, _mm512_shuffle_epi8(t.lo[1], lnh));
    tph = _mm512_xor_si512(tph, _mm512_shuffle_epi8(t.hi[1], lnh));
    tpl = _mm512_xor_si512(tpl, _mm512_shuffle_epi8(t.lo[2], hn));
    tph = _mm512_xor_si512(tph, _mm512_shuffle_epi8(t.hi[2], hn));
    tpl = _mm512_xor_si512(tpl, _mm512_shuffle_epi8(t.lo[3], hnh));
    tph = _mm512_xor_si512(tph, _mm512_shuffle_epi8(t.hi[3], hnh));
    if add {
        tpl = _mm512_xor_si512(tpl, _mm512_loadu_si512(dst.add(64) as _));
        tph = _mm512_xor_si512(tph, _mm512_loadu_si512(dst as _));
    }
    _mm512_storeu_si512(dst as _, tph);
    _mm512_storeu_si512(dst.add(64) as _, tpl);
}

/// Scalar tail helpers - same plane layout as the AVX2 kernel:
/// hi bytes at block[0..k], lo bytes at block[64..64+k] for k words.
pub(crate) fn prepare_scalar(dst: &mut [u8], src: &[u8]) {
    // src reads are linear (2 bytes/word); the bound is words fitting in
    // dst's [hi@0][lo@64] planes, not src.len()-64
    let words = (src.len() / 2).min(64).min(dst.len().saturating_sub(64));
    for b in dst.iter_mut() {
        *b = 0;
    }
    for i in 0..words {
        dst[i] = src[2 * i + 1]; // high byte plane
        dst[64 + i] = src[2 * i]; // low byte plane
    }
    if src.len() % 2 == 1 && 64 + words < dst.len() {
        dst[64 + words] = src[src.len() - 1];
    }
}

pub(crate) fn finish_scalar(dst: &mut [u8], src: &[u8]) {
    let words = (dst.len() / 2).min(64).min(src.len().saturating_sub(64));
    for i in 0..words {
        dst[2 * i] = src[64 + i];
        dst[2 * i + 1] = src[i];
    }
    if dst.len() % 2 == 1 && 64 + words < src.len() {
        *dst.last_mut().unwrap() = src[64 + words];
    }
}

pub(crate) fn muladd_scalar(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let words = (src.len() / 2)
        .min(64)
        .min(src.len().saturating_sub(64))
        .min(dst.len().saturating_sub(64));
    for i in 0..words {
        let w = ((src[i] as u16) << 8) | src[64 + i] as u16;
        let p = gf16_mul(w, coeff);
        if add {
            let cur = ((dst[i] as u16) << 8) | dst[64 + i] as u16;
            let r = cur ^ p;
            dst[i] = (r >> 8) as u8;
            dst[64 + i] = r as u8;
        } else {
            dst[i] = (p >> 8) as u8;
            dst[64 + i] = p as u8;
        }
    }
}

// Per-call loops must be #[target_feature] fns (see shuffle_avx2).
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn prepare_run(dst: &mut [u8], src: &[u8]) {
    let mut i = 0;
    while i + BLK <= src.len() {
        prepare_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK;
    }
    if i < src.len() {
        prepare_scalar(&mut dst[i..], &src[i..]);
    } else {
        for b in dst[i..].iter_mut() {
            *b = 0;
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn finish_run(dst: &mut [u8], src: &[u8]) {
    let mut i = 0;
    while i + BLK <= dst.len() {
        finish_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK;
    }
    if i < dst.len() {
        finish_scalar(&mut dst[i..], &src[i..]);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn muladd_run(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let t = make_tables(coeff);
    let mut i = 0;
    while i + BLK <= src.len() {
        muladd_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &t, add);
        i += BLK;
    }
    if i < src.len() {
        muladd_scalar(&mut dst[i..], &src[i..], coeff, add);
    }
}

impl GfKernel for ShuffleAvx512 {
    fn id(&self) -> i32 {
        13 // GF16_SHUFFLE_AVX512
    }
    fn name(&self) -> &'static str {
        "Shuffle (AVX512)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        64
    }
    fn stride(&self) -> usize {
        128
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::gf16_mul;

    fn have() -> bool {
        std::arch::is_x86_feature_detected!("avx512bw")
    }

    fn fill(buf: &mut [u8], seed: u64) {
        let mut rng = seed;
        for (i, b) in buf.iter_mut().enumerate() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            *b = (rng >> 33) as u8;
        }
    }

    #[test]
    fn prepare_finish_roundtrip() {
        if !have() {
            return;
        }
        let k = ShuffleAvx512;
        for len in [BLK, BLK * 3, 7usize, 130, 0] {
            let mut src = vec![0u8; len];
            fill(&mut src, 0x1234 + len as u64);
            let mut packed = vec![0u8; len.div_ceil(BLK) * BLK];
            k.prepare(&mut packed, &src);
            let mut out = vec![0u8; len];
            k.finish(&mut out, &packed);
            assert_eq!(out, src, "len {}", len);
        }
    }

    #[test]
    fn mul_vs_scalar() {
        if !have() {
            return;
        }
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x987);
        let k = ShuffleAvx512;
        let mut packed = [0u8; BLK * 2];
        k.prepare(&mut packed, &src);
        for coeff in [1u16, 2, 0x1234, 0xffff, 0x8001, 0x4000] {
            let mut dst = [0u8; BLK * 2];
            let mut out = [0u8; BLK * 2];
            k.mul(&mut dst, &packed, coeff);
            k.finish(&mut out, &dst);
            for w in 0..(src.len() / 2) {
                let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
                let ow = (out[2 * w + 1] as u16) << 8 | out[2 * w] as u16;
                assert_eq!(ow, gf16_mul(sw, coeff), "w{} coeff {:x}", w, coeff);
            }
        }
    }

    #[test]
    fn muladd_accumulates() {
        if !have() {
            return;
        }
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x55);
        let mut d0 = [0u8; BLK * 2];
        fill(&mut d0, 0xaa);
        let k = ShuffleAvx512;
        let mut packed = [0u8; BLK * 2];
        k.prepare(&mut packed, &src);
        let mut dst = [0u8; BLK * 2];
        k.prepare(&mut dst, &d0);
        k.muladd(&mut dst, &packed, 0xbeef);
        k.muladd(&mut dst, &packed, 0x1234);
        let mut out = [0u8; BLK * 2];
        k.finish(&mut out, &dst);
        for w in 0..(src.len() / 2) {
            let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
            let d0w = (d0[2 * w + 1] as u16) << 8 | d0[2 * w] as u16;
            let want = d0w ^ gf16_mul(sw, 0xbeef) ^ gf16_mul(sw, 0x1234);
            let ow = (out[2 * w + 1] as u16) << 8 | out[2 * w] as u16;
            assert_eq!(ow, want, "w{}", w);
        }
    }
}

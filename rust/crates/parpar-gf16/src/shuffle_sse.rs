//! SSSE3 shuffle kernel - 128-bit variant of the AVX2 shuffle kernel.
//! Packed block: 32 bytes covering 16 u16 words as [16 hi][16 lo] bytes.
//! Same nibble-table multiply as ShuffleAvx2 with xmm pshufb.

#![cfg(target_arch = "x86_64")]

use crate::gf::{gf16_mul, gf16_mul2};
use crate::kernel::GfKernel;
use std::arch::x86_64::*;

pub struct ShuffleSse;

const BLK: usize = 32; // 16 u16 words

struct Tables {
    lo: [__m128i; 4],
    hi: [__m128i; 4],
}

#[target_feature(enable = "ssse3")]
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
    let v = |pair: ([u8; 16], [u8; 16])| -> (__m128i, __m128i) {
        (
            _mm_loadu_si128(pair.0.as_ptr() as _),
            _mm_loadu_si128(pair.1.as_ptr() as _),
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
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn prepare_block(dst: *mut u8, src: *const u8) {
    let mask = _mm_set1_epi16(0x00ff);
    let a = _mm_loadu_si128(src as _);
    let b = _mm_loadu_si128(src.add(16) as _);
    let hi = _mm_packus_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
    let lo = _mm_packus_epi16(_mm_and_si128(a, mask), _mm_and_si128(b, mask));
    _mm_storeu_si128(dst as _, hi);
    _mm_storeu_si128(dst.add(16) as _, lo);
}

#[inline]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn finish_block(dst: *mut u8, src: *const u8) {
    let hi = _mm_loadu_si128(src as _);
    let lo = _mm_loadu_si128(src.add(16) as _);
    _mm_storeu_si128(dst as _, _mm_unpacklo_epi8(lo, hi));
    _mm_storeu_si128(dst.add(16) as _, _mm_unpackhi_epi8(lo, hi));
}

#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn muladd_block(dst: *mut u8, src: *const u8, t: &Tables, add: bool) {
    let nmask = _mm_set1_epi8(0x0f);
    let hi = _mm_loadu_si128(src as _);
    let lo = _mm_loadu_si128(src.add(16) as _);
    let hn = _mm_and_si128(hi, nmask);
    let hnh = _mm_and_si128(_mm_srli_epi16(hi, 4), nmask);
    let ln = _mm_and_si128(lo, nmask);
    let lnh = _mm_and_si128(_mm_srli_epi16(lo, 4), nmask);
    let mut tpl = _mm_shuffle_epi8(t.lo[0], ln);
    let mut tph = _mm_shuffle_epi8(t.hi[0], ln);
    tpl = _mm_xor_si128(tpl, _mm_shuffle_epi8(t.lo[1], lnh));
    tph = _mm_xor_si128(tph, _mm_shuffle_epi8(t.hi[1], lnh));
    tpl = _mm_xor_si128(tpl, _mm_shuffle_epi8(t.lo[2], hn));
    tph = _mm_xor_si128(tph, _mm_shuffle_epi8(t.hi[2], hn));
    tpl = _mm_xor_si128(tpl, _mm_shuffle_epi8(t.lo[3], hnh));
    tph = _mm_xor_si128(tph, _mm_shuffle_epi8(t.hi[3], hnh));
    if add {
        tpl = _mm_xor_si128(tpl, _mm_loadu_si128(dst.add(16) as _));
        tph = _mm_xor_si128(tph, _mm_loadu_si128(dst as _));
    }
    _mm_storeu_si128(dst as _, tph);
    _mm_storeu_si128(dst.add(16) as _, tpl);
}

// scalar tails - same plane layout as the AVX2 kernel (hi@0, lo@16)
pub(crate) fn prepare_scalar(dst: &mut [u8], src: &[u8]) {
    // src is linear (2 bytes/word); dst is [hi@0][lo@16] planes. The bound
    // is src.len()/2 words (src reads), capped at one block (16 words) and
    // at what the dst planes can hold (dst.len()-16 words).
    let words = (src.len() / 2).min(16).min(dst.len().saturating_sub(16));
    for b in dst.iter_mut() {
        *b = 0;
    }
    for i in 0..words {
        dst[i] = src[2 * i + 1];
        dst[16 + i] = src[2 * i];
    }
    if src.len() % 2 == 1 && 16 + words < dst.len() {
        dst[16 + words] = src[src.len() - 1];
    }
}

pub(crate) fn finish_scalar(dst: &mut [u8], src: &[u8]) {
    // src is [hi@0][lo@16] planes; the lo reads at src[16+i] bound words
    // to src.len()-16
    let words = (dst.len() / 2).min(16).min(src.len().saturating_sub(16));
    for i in 0..words {
        dst[2 * i] = src[16 + i];
        dst[2 * i + 1] = src[i];
    }
    if dst.len() % 2 == 1 && 16 + words < src.len() {
        // odd trailing byte = lo byte of the last partial word
        *dst.last_mut().unwrap() = src[16 + words];
    }
}

pub(crate) fn muladd_scalar(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    // src/dst are [hi@0][lo@16] planes; lo accesses bound words to len-16.
    // Packed regions are always block-multiples so this tail is a
    // misuse-safety net only.
    let words = (src.len() / 2)
        .min(16)
        .min(src.len().saturating_sub(16))
        .min(dst.len().saturating_sub(16));
    for i in 0..words {
        let w = ((src[i] as u16) << 8) | src[16 + i] as u16;
        let p = gf16_mul(w, coeff);
        if add {
            let cur = ((dst[i] as u16) << 8) | dst[16 + i] as u16;
            let r = cur ^ p;
            dst[i] = (r >> 8) as u8;
            dst[16 + i] = r as u8;
        } else {
            dst[i] = (p >> 8) as u8;
            dst[16 + i] = p as u8;
        }
    }
}

// Per-call loops must be #[target_feature] fns (see shuffle_avx2).
#[target_feature(enable = "ssse3")]
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

#[target_feature(enable = "ssse3")]
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

#[target_feature(enable = "ssse3")]
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

impl GfKernel for ShuffleSse {
    fn id(&self) -> i32 {
        crate::kernel::method::SHUFFLE_SSE
    }
    fn name(&self) -> &'static str {
        "Shuffle (SSSE3)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        16
    }
    fn stride(&self) -> usize {
        32
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

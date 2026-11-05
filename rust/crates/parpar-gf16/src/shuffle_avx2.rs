//! AVX2 shuffle kernel - port of gf16_shuffle_avx2.c / gf16_shuffle_x86.h.
//!
//! Packed block: 64 bytes covering 32 u16 words, stored as
//! [32 high bytes][32 low bytes] (natural order via packus+permute fixup -
//! the C++ lane order differs internally but any self-consistent packing
//! produces identical unpacked output).
//!
//! Multiply: c*w decomposed over nibbles - for low byte x of the word,
//! product = c*x ^ c*(x<<4); for high byte, c*(x<<8) ^ c*(x<<12). Each
//! 16-entry nibble table is split into low/high byte planes for pshufb.

#![cfg(target_arch = "x86_64")]

use crate::gf::gf16_mul;
use crate::kernel::GfKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct ShuffleAvx2;

const BLK: usize = 64; // packed block: 32 u16 words

// The per-call loops must be #[target_feature] fns themselves: a feature
// fn cannot inline into a plain caller, so looping over blocks from the
// trait methods would cost a call plus 8 table reloads per 64 bytes.
#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
unsafe fn muladd_run(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let t = make_tables(coeff);
    let nblocks = src.len() / BLK;
    for b in 0..nblocks {
        muladd_block(
            dst.as_mut_ptr().add(b * BLK),
            src.as_ptr().add(b * BLK),
            &t,
            add,
        );
    }
    let i = nblocks * BLK;
    if i < src.len() {
        muladd_scalar(&mut dst[i..], &src[i..], coeff, add);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn muladd2_run(
    dst: &mut [u8],
    src_a: &[u8],
    coeff_a: u16,
    src_b: &[u8],
    coeff_b: u16,
) {
    let n = src_a.len().min(src_b.len());
    let ta = make_tables(coeff_a);
    let tb = make_tables(coeff_b);
    let mut i = 0;
    // 4x unroll: independent block chains give the OoO engine
    // parallel pshufb pipelines instead of a serial block stream
    while i + 4 * BLK <= n {
        for u in 0..4 {
            muladd2_block(
                dst.as_mut_ptr().add(i + u * BLK),
                src_a.as_ptr().add(i + u * BLK),
                &ta,
                src_b.as_ptr().add(i + u * BLK),
                &tb,
            );
        }
        i += 4 * BLK;
    }
    while i + BLK <= n {
        muladd2_block(
            dst.as_mut_ptr().add(i),
            src_a.as_ptr().add(i),
            &ta,
            src_b.as_ptr().add(i),
            &tb,
        );
        i += BLK;
    }
    if i < n {
        muladd_scalar(&mut dst[i..], &src_a[i..n], coeff_a, true);
        muladd_scalar(&mut dst[i..], &src_b[i..n], coeff_b, true);
    }
}

/// Eight 16-entry pshufb tables for a coefficient (low/high byte planes x4).
struct Tables {
    lo: [__m256i; 4],
    hi: [__m256i; 4],
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn make_tables(coeff: u16) -> Tables {
    // table k entry x = coeff*(x<<4k) = c_k*x, c_k = coeff*2^(4k). All 16
    // entries are built at once in u16 lanes: c*x for a 4-bit x is the XOR
    // of xtime^j(c) over set bits j of x. ~30 SIMD ops/table vs 16 scalar
    // bit-serial gf16_mul calls.
    let poly = _mm256_set1_epi16(0x100b);
    let xt = |v: __m256i| -> __m256i {
        _mm256_xor_si256(
            _mm256_add_epi16(v, v),
            _mm256_and_si256(_mm256_srai_epi16::<15>(v), poly),
        )
    };
    let xv = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let m = [
        _mm256_srai_epi16::<15>(_mm256_slli_epi16::<15>(xv)),
        _mm256_srai_epi16::<15>(_mm256_slli_epi16::<14>(xv)),
        _mm256_srai_epi16::<15>(_mm256_slli_epi16::<13>(xv)),
        _mm256_srai_epi16::<15>(_mm256_slli_epi16::<12>(xv)),
    ];
    let ff = _mm256_set1_epi16(0xff);
    let mut lo = [_mm256_setzero_si256(); 4];
    let mut hi = [_mm256_setzero_si256(); 4];
    let mut c = _mm256_set1_epi16(coeff as i16);
    for k in 0..4 {
        let d1 = xt(c);
        let d2 = xt(d1);
        let d3 = xt(d2);
        let p = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_and_si256(c, m[0]), _mm256_and_si256(d1, m[1])),
            _mm256_xor_si256(_mm256_and_si256(d2, m[2]), _mm256_and_si256(d3, m[3])),
        );
        // pack u16 lanes -> 16 lo/hi bytes, duplicated across lanes by
        // packus; permq(0xd8) gathers qwords 0,2 into both halves
        let l8 = _mm256_packus_epi16(_mm256_and_si256(p, ff), _mm256_and_si256(p, ff));
        let h8 = _mm256_srli_epi16::<8>(p);
        let h8 = _mm256_packus_epi16(h8, h8);
        lo[k] = _mm256_permute4x64_epi64(l8, 0b11011000);
        hi[k] = _mm256_permute4x64_epi64(h8, 0b11011000);
        c = xt(xt(xt(xt(c))));
    }
    Tables { lo, hi }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn prepare_block(dst: *mut u8, src: *const u8) {
    let mask = _mm256_set1_epi16(0x00ff);
    let a = _mm256_loadu_si256(src as _); // words 0..15
    let b = _mm256_loadu_si256(src.add(32) as _); // words 16..31
    let hi = _mm256_packus_epi16(_mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8));
    let lo = _mm256_packus_epi16(_mm256_and_si256(a, mask), _mm256_and_si256(b, mask));
    // undo packus per-lane interleaving -> natural order
    let hi = _mm256_permute4x64_epi64(hi, 0b11011000);
    let lo = _mm256_permute4x64_epi64(lo, 0b11011000);
    _mm256_storeu_si256(dst as _, hi);
    _mm256_storeu_si256(dst.add(32) as _, lo);
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn finish_block(dst: *mut u8, src: *const u8) {
    let hi = _mm256_loadu_si256(src as _);
    let lo = _mm256_loadu_si256(src.add(32) as _);
    let x = _mm256_unpacklo_epi8(lo, hi); // w0..7 | w16..23
    let y = _mm256_unpackhi_epi8(lo, hi); // w8..15 | w24..31
    _mm256_storeu_si256(dst as _, _mm256_permute2x128_si256(x, y, 0x20));
    _mm256_storeu_si256(dst.add(32) as _, _mm256_permute2x128_si256(x, y, 0x31));
}

/// c*src for one packed block, as (lo-plane, hi-plane) vectors.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_plane(src: *const u8, t: &Tables) -> (__m256i, __m256i) {
    let nmask = _mm256_set1_epi8(0x0f);
    // C++ structure: each plane is loaded twice so the low-nibble extract
    // folds its load into vpand (mem) - trades load-port uops for FP-port
    // ands, and the FP ports are the saturated resource here.
    let hi_v = _mm256_loadu_si256(src as _);
    let lo_v = _mm256_loadu_si256(src.add(32) as _);
    let hn = _mm256_and_si256(_mm256_loadu_si256(src as _), nmask);
    let ln = _mm256_and_si256(_mm256_loadu_si256(src.add(32) as _), nmask);
    let hnh = _mm256_and_si256(_mm256_srli_epi16(hi_v, 4), nmask);
    let lnh = _mm256_and_si256(_mm256_srli_epi16(lo_v, 4), nmask);
    let mut tpl = _mm256_shuffle_epi8(t.lo[0], ln);
    let mut tph = _mm256_shuffle_epi8(t.hi[0], ln);
    tpl = _mm256_xor_si256(tpl, _mm256_shuffle_epi8(t.lo[1], lnh));
    tph = _mm256_xor_si256(tph, _mm256_shuffle_epi8(t.hi[1], lnh));
    tpl = _mm256_xor_si256(tpl, _mm256_shuffle_epi8(t.lo[2], hn));
    tph = _mm256_xor_si256(tph, _mm256_shuffle_epi8(t.hi[2], hn));
    tpl = _mm256_xor_si256(tpl, _mm256_shuffle_epi8(t.lo[3], hnh));
    tph = _mm256_xor_si256(tph, _mm256_shuffle_epi8(t.hi[3], hnh));
    (tpl, tph)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn muladd_block(dst: *mut u8, src: *const u8, t: &Tables, add: bool) {
    let (mut tpl, mut tph) = mul_plane(src, t);
    if add {
        tpl = _mm256_xor_si256(tpl, _mm256_loadu_si256(dst.add(32) as _));
        tph = _mm256_xor_si256(tph, _mm256_loadu_si256(dst as _));
    }
    _mm256_storeu_si256(dst as _, tph);
    _mm256_storeu_si256(dst.add(32) as _, tpl);
}

/// dst ^= ca*sa ^ cb*sb for one packed block - both products accumulate
/// into a single dst load/store (C++ procRegions=2 fused pass).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn muladd2_block(
    dst: *mut u8,
    src_a: *const u8,
    ta: &Tables,
    src_b: *const u8,
    tb: &Tables,
) {
    let (mut tpl, mut tph) = mul_plane(src_a, ta);
    let (pl, ph) = mul_plane(src_b, tb);
    tpl = _mm256_xor_si256(tpl, pl);
    tph = _mm256_xor_si256(tph, ph);
    tpl = _mm256_xor_si256(tpl, _mm256_loadu_si256(dst.add(32) as _));
    tph = _mm256_xor_si256(tph, _mm256_loadu_si256(dst as _));
    _mm256_storeu_si256(dst as _, tph);
    _mm256_storeu_si256(dst.add(32) as _, tpl);
}

/// Scalar tail helpers - a partial packed block keeps the same plane
/// layout as a full one: hi bytes at block[0..k], lo bytes at
/// block[32..32+k] for k words (rest zero).
pub(crate) fn prepare_scalar(dst: &mut [u8], src: &[u8]) {
    // an odd trailing byte forms a lo-only word (hi = 0)
    let words = (src.len() / 2).min(32).min(dst.len().saturating_sub(32));
    for b in dst.iter_mut() {
        *b = 0;
    }
    for i in 0..words {
        dst[i] = src[2 * i + 1]; // high byte plane
        dst[32 + i] = src[2 * i]; // low byte plane
    }
    if src.len() % 2 == 1 && 32 + words < dst.len() {
        dst[32 + words] = src[src.len() - 1];
    }
}

pub(crate) fn finish_scalar(dst: &mut [u8], src: &[u8]) {
    // lo reads at src[32+i] bound words to src.len()-32
    let words = (dst.len() / 2).min(32).min(src.len().saturating_sub(32));
    for i in 0..words {
        dst[2 * i] = src[32 + i];
        dst[2 * i + 1] = src[i];
    }
    if dst.len() % 2 == 1 && 32 + words < src.len() {
        *dst.last_mut().unwrap() = src[32 + words];
    }
}

pub(crate) fn muladd_scalar(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    // last partial block: words at hi[0..k], lo[32..32+k]. Clamp to words
    // readable from both planes - short tails are out-of-contract anyway.
    let words = (src.len() / 2)
        .min(32)
        .min(src.len().saturating_sub(32))
        .min(dst.len().saturating_sub(32));
    for i in 0..words {
        let w = ((src[i] as u16) << 8) | src[32 + i] as u16;
        let p = gf16_mul(w, coeff);
        if add {
            let cur = ((dst[i] as u16) << 8) | dst[32 + i] as u16;
            let r = cur ^ p;
            dst[i] = (r >> 8) as u8;
            dst[32 + i] = r as u8;
        } else {
            dst[i] = (p >> 8) as u8;
            dst[32 + i] = p as u8;
        }
    }
}

impl GfKernel for ShuffleAvx2 {
    fn id(&self) -> i32 {
        crate::kernel::method::SHUFFLE_AVX2
    }
    fn name(&self) -> &'static str {
        "Shuffle (AVX2)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        32
    }
    fn stride(&self) -> usize {
        64
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

    fn muladd2(&self, dst: &mut [u8], src_a: &[u8], coeff_a: u16, src_b: &[u8], coeff_b: u16) {
        let n = src_a.len().min(src_b.len());
        unsafe { muladd2_run(&mut dst[..n], &src_a[..n], coeff_a, &src_b[..n], coeff_b) }
        // a staged input can be shorter than its partner (slice size grew
        // after staging) - the longer tail still contributes
        if src_a.len() > n {
            self.muladd(&mut dst[n..], &src_a[n..], coeff_a);
        }
        if src_b.len() > n {
            self.muladd(&mut dst[n..], &src_b[n..], coeff_b);
        }
    }

    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe { muladd_run(dst, src, coeff, false) }
    }
}

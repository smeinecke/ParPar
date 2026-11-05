//! Shuffle2x kernels - port of gf16_shuffle2x_x86.h for AVX2 (method 15)
//! and AVX512 (method 16).
//!
//! Packed layout per block: [lo bytes of all words][hi bytes of all words].
//! The multiply exploits per-lane table shuffles: the lo bytes sit in the
//! low lanes and hi bytes in the high lanes, so a single vpshufb pass
//! computes lo-plane results in lanes 0-1 and hi-plane results in lanes
//! 2-3; the cross products get lane-swapped back into place.

#![cfg(target_arch = "x86_64")]

use crate::gf::{gf16_mul, gf16_mul2};
use crate::kernel::GfKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------- AVX2 variant (method 15): 32-byte blocks, 16 words ----------

pub struct Shuffle2xAvx2;
const BLK2: usize = 32;

struct Tbl2x {
    norm_lo: __m256i,
    swap_lo: __m256i,
    norm_hi: __m256i,
    swap_hi: __m256i,
}

/// pl[n][k] = lo byte of coeff*(k<<4n), ph[n][k] = hi byte. c_n*k for
/// 4-bit k is the XOR of xtime^j(c_n) over set bits j of k - avoids the
/// bit-serial gf16_mul loop per entry.
fn product_planes(coeff: u16) -> ([[u8; 16]; 4], [[u8; 16]; 4]) {
    let mut pl = [[0u8; 16]; 4];
    let mut ph = [[0u8; 16]; 4];
    let mut c = coeff;
    for n in 0..4 {
        let d1 = gf16_mul2(c);
        let d2 = gf16_mul2(d1);
        let d3 = gf16_mul2(d2);
        for k in 1..16usize {
            let mut p = 0u16;
            if k & 1 != 0 {
                p ^= c;
            }
            if k & 2 != 0 {
                p ^= d1;
            }
            if k & 4 != 0 {
                p ^= d2;
            }
            if k & 8 != 0 {
                p ^= d3;
            }
            pl[n][k] = p as u8;
            ph[n][k] = (p >> 8) as u8;
        }
        for _ in 0..4 {
            c = gf16_mul2(c);
        }
    }
    (pl, ph)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn make_tables2x(coeff: u16) -> Tbl2x {
    let (pl, ph) = product_planes(coeff);
    let load = |lo: &[u8; 16], hi: &[u8; 16]| {
        let mut v = [0u8; 32];
        v[..16].copy_from_slice(lo);
        v[16..].copy_from_slice(hi);
        _mm256_loadu_si256(v.as_ptr() as _)
    };
    Tbl2x {
        // lane0 serves lo bytes (word bits 3:0 / 7:4), lane1 serves hi
        // bytes (bits 11:8 / 15:12); swap tables carry the cross products
        norm_lo: load(&pl[0], &ph[2]),
        swap_lo: load(&ph[0], &pl[2]),
        norm_hi: load(&pl[1], &ph[3]),
        swap_hi: load(&ph[1], &pl[3]),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn muladd_block2x(dst: *mut u8, src: *const u8, t: &Tbl2x, add: bool) {
    let mask = _mm256_set1_epi8(0x0f);
    let data = _mm256_loadu_si256(src as _);
    let lo4 = _mm256_and_si256(data, mask);
    let hi4 = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask);
    let mut res = _mm256_shuffle_epi8(t.norm_lo, lo4);
    res = _mm256_xor_si256(res, _mm256_shuffle_epi8(t.norm_hi, hi4));
    let mut sw = _mm256_shuffle_epi8(t.swap_lo, lo4);
    sw = _mm256_xor_si256(sw, _mm256_shuffle_epi8(t.swap_hi, hi4));
    sw = _mm256_permute2x128_si256(sw, sw, 0x01);
    res = _mm256_xor_si256(res, sw);
    if add {
        res = _mm256_xor_si256(res, _mm256_loadu_si256(dst as _));
    }
    _mm256_storeu_si256(dst as _, res);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn prepare_block2x(dst: *mut u8, src: *const u8) {
    let v = _mm256_loadu_si256(src as _);
    // per-lane separate: [8 lo bytes | 8 hi bytes] of the lane's 8 words
    let sep = _mm256_shuffle_epi8(
        v,
        _mm256_set_epi8(
            15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1, 14,
            12, 10, 8, 6, 4, 2, 0,
        ),
    ); // [L0lo | L0hi | L1lo | L1hi]
       // gather qwords [0,2,1,3] -> [16 lo][16 hi]
    _mm256_storeu_si256(dst as _, _mm256_permute4x64_epi64(sep, 0xd8));
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn finish_block2x(dst: *mut u8, src: *const u8) {
    let v = _mm256_loadu_si256(src as _); // [16 lo][16 hi]
                                          // qwords [lo0-7, lo8-15, hi0-7, hi8-15] -> [lo0-7|hi0-7 | lo8-15|hi8-15]
    let v = _mm256_permute4x64_epi64(v, 0xd8);
    // interleave each lane's byte halves back into u16 words
    let v = _mm256_shuffle_epi8(
        v,
        _mm256_set_epi8(
            15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0, 15, 7, 14, 6, 13, 5, 12, 4, 11,
            3, 10, 2, 9, 1, 8, 0,
        ),
    );
    _mm256_storeu_si256(dst as _, v);
}

fn prepare_scalar2x(dst: &mut [u8], src: &[u8]) {
    for b in dst.iter_mut() {
        *b = 0;
    }
    // src reads are linear; bound words to what dst's [lo@0][hi@16]
    // planes hold (one block = 16 words)
    let words = (src.len() / 2).min(16).min(dst.len().saturating_sub(16));
    for i in 0..words {
        dst[i] = src[2 * i];
        dst[16 + i] = src[2 * i + 1];
    }
    if src.len() % 2 == 1 && words < dst.len() {
        dst[words] = src[2 * words];
    }
}

fn finish_scalar2x(dst: &mut [u8], src: &[u8]) {
    // hi reads at src[16+i] bound words to src.len()-16
    let words = (dst.len() / 2).min(16).min(src.len().saturating_sub(16));
    for i in 0..words {
        dst[2 * i] = src[i];
        dst[2 * i + 1] = src[16 + i];
    }
    if dst.len() % 2 == 1 && words < src.len() {
        dst[2 * words] = src[words];
    }
}

fn muladd_scalar2x(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let len = dst.len().min(src.len());
    let words = (len / 2).min(16).min(len.saturating_sub(16));
    for i in 0..words {
        let w = (src[16 + i] as u16) << 8 | src[i] as u16;
        let p = gf16_mul(w, coeff);
        let d = ((dst[16 + i] as u16) << 8 | dst[i] as u16) ^ p;
        let v = if add { d } else { p };
        dst[i] = v as u8;
        dst[16 + i] = (v >> 8) as u8;
    }
    if len % 2 == 1 && 16 + words < src.len() && 16 + words < dst.len() {
        let w = src[words] as u16;
        let p = gf16_mul(w, coeff);
        dst[words] = if add { dst[words] ^ p as u8 } else { p as u8 };
    }
}

// Per-call loops must be #[target_feature] fns (see shuffle_avx2).
#[target_feature(enable = "avx2")]
unsafe fn prepare_run_2x(dst: &mut [u8], src: &[u8]) {
    let mut i = 0;
    while i + BLK2 <= src.len() {
        prepare_block2x(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK2;
    }
    if i < src.len() {
        prepare_scalar2x(&mut dst[i..], &src[i..]);
    } else {
        for b in dst[i..].iter_mut() {
            *b = 0;
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn finish_run_2x(dst: &mut [u8], src: &[u8]) {
    let mut i = 0;
    while i + BLK2 <= dst.len() {
        finish_block2x(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK2;
    }
    if i < dst.len() {
        finish_scalar2x(&mut dst[i..], &src[i..]);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn muladd_run_2x(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let t = make_tables2x(coeff);
    let mut i = 0;
    while i + BLK2 <= src.len() {
        muladd_block2x(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &t, add);
        i += BLK2;
    }
    if i < src.len() {
        muladd_scalar2x(&mut dst[i..], &src[i..], coeff, add);
    }
}

impl GfKernel for Shuffle2xAvx2 {
    fn id(&self) -> i32 {
        15 // GF16_SHUFFLE2X_AVX2
    }
    fn name(&self) -> &'static str {
        "Shuffle2x (AVX2)"
    }
    fn block_len(&self) -> usize {
        BLK2
    }
    fn alignment(&self) -> usize {
        32
    }
    fn stride(&self) -> usize {
        32
    }

    fn prepare(&self, dst: &mut [u8], src: &[u8]) {
        unsafe { prepare_run_2x(dst, src) }
    }

    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        unsafe { finish_run_2x(dst, src) }
    }

    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe { muladd_run_2x(dst, src, coeff, true) }
    }

    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe { muladd_run_2x(dst, src, coeff, false) }
    }
}

// ---------- AVX512 variant (method 16): 64-byte blocks, 32 words ----------

pub struct Shuffle2xAvx512;
const BLK2W: usize = 64;

struct Tbl2xw {
    norm_lo: __m512i,
    swap_lo: __m512i,
    norm_hi: __m512i,
    swap_hi: __m512i,
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn make_tables2xw(coeff: u16) -> Tbl2xw {
    let (pl, ph) = product_planes(coeff);
    // lanes 0,1 serve lo bytes; lanes 2,3 serve hi bytes
    let load = |a: &[u8; 16], b: &[u8; 16]| {
        let mut v = [0u8; 64];
        v[..16].copy_from_slice(a);
        v[16..32].copy_from_slice(a);
        v[32..48].copy_from_slice(b);
        v[48..].copy_from_slice(b);
        _mm512_loadu_si512(v.as_ptr() as _)
    };
    Tbl2xw {
        norm_lo: load(&pl[0], &ph[2]),
        swap_lo: load(&ph[0], &pl[2]),
        norm_hi: load(&pl[1], &ph[3]),
        swap_hi: load(&ph[1], &pl[3]),
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn muladd_block2xw(dst: *mut u8, src: *const u8, t: &Tbl2xw, add: bool) {
    let mask = _mm512_set1_epi8(0x0f);
    let data = _mm512_loadu_si512(src as _);
    let lo4 = _mm512_and_si512(data, mask);
    let hi4 = _mm512_and_si512(_mm512_srli_epi16(data, 4), mask);
    let mut res = _mm512_shuffle_epi8(t.norm_lo, lo4);
    res = _mm512_xor_si512(res, _mm512_shuffle_epi8(t.norm_hi, hi4));
    let mut sw = _mm512_shuffle_epi8(t.swap_lo, lo4);
    sw = _mm512_xor_si512(sw, _mm512_shuffle_epi8(t.swap_hi, hi4));
    // swap 256-bit halves: lanes [0,1] <-> [2,3]
    sw = _mm512_shuffle_i32x4(sw, sw, 0x4e);
    res = _mm512_xor_si512(res, sw);
    if add {
        res = _mm512_xor_si512(res, _mm512_loadu_si512(dst as _));
    }
    _mm512_storeu_si512(dst as _, res);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn prepare_block2xw(dst: *mut u8, src: *const u8) {
    let v = _mm512_loadu_si512(src as _);
    // per-lane gather of lo/hi bytes into the lane's low qword
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
    let lo = _mm512_shuffle_epi8(v, lo_idx); // lo qs at qword 2k of each lane
    let hi = _mm512_shuffle_epi8(v, hi_idx);
    // collect even qwords of lo -> [32 lo], odd of hi... gather order:
    // lo plane = qwords {0,2,4,6} of lo, hi plane = qwords {0,2,4,6} of hi
    let evens = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    // one zmm holds the whole block: [32 lo bytes][32 hi bytes]
    _mm512_storeu_si512(dst as _, _mm512_permutex2var_epi64(lo, evens, hi));
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn finish_block2xw(dst: *mut u8, src: *const u8) {
    let v = _mm512_loadu_si512(src as _); // [32 lo][32 hi]
                                          // qwords [lo_a,lo_b,lo_c,lo_d, hi_a,..] -> lanes [lo_k | hi_k]
    let v = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0), v);
    // interleave each lane's byte halves back into u16 words
    let shuf = _mm512_set_epi8(
        15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0, 15, 7, 14, 6, 13, 5, 12, 4, 11, 3,
        10, 2, 9, 1, 8, 0, 15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0, 15, 7, 14, 6, 13,
        5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0,
    );
    _mm512_storeu_si512(dst as _, _mm512_shuffle_epi8(v, shuf));
}

fn prepare_scalar2xw(dst: &mut [u8], src: &[u8]) {
    for b in dst.iter_mut() {
        *b = 0;
    }
    // src reads are linear; bound words to what dst's [lo@0][hi@32]
    // planes hold (one block = 32 words)
    let words = (src.len() / 2).min(32).min(dst.len().saturating_sub(32));
    for i in 0..words {
        dst[i] = src[2 * i];
        dst[32 + i] = src[2 * i + 1];
    }
    if src.len() % 2 == 1 && words < dst.len() {
        dst[words] = src[2 * words];
    }
}

fn finish_scalar2xw(dst: &mut [u8], src: &[u8]) {
    // hi reads at src[32+i] bound words to src.len()-32
    let words = (dst.len() / 2).min(32).min(src.len().saturating_sub(32));
    for i in 0..words {
        dst[2 * i] = src[i];
        dst[2 * i + 1] = src[32 + i];
    }
    if dst.len() % 2 == 1 && words < src.len() {
        dst[2 * words] = src[words];
    }
}

fn muladd_scalar2xw(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let len = dst.len().min(src.len());
    let words = (len / 2).min(32).min(len.saturating_sub(32));
    for i in 0..words {
        let w = (src[32 + i] as u16) << 8 | src[i] as u16;
        let p = gf16_mul(w, coeff);
        let d = ((dst[32 + i] as u16) << 8 | dst[i] as u16) ^ p;
        let v = if add { d } else { p };
        dst[i] = v as u8;
        dst[32 + i] = (v >> 8) as u8;
    }
    if len % 2 == 1 && 32 + words < src.len() && 32 + words < dst.len() {
        let w = src[words] as u16;
        let p = gf16_mul(w, coeff);
        dst[words] = if add { dst[words] ^ p as u8 } else { p as u8 };
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn prepare_run_2xw(dst: &mut [u8], src: &[u8]) {
    let mut i = 0;
    while i + BLK2W <= src.len() {
        prepare_block2xw(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK2W;
    }
    if i < src.len() {
        prepare_scalar2xw(&mut dst[i..], &src[i..]);
    } else {
        for b in dst[i..].iter_mut() {
            *b = 0;
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn finish_run_2xw(dst: &mut [u8], src: &[u8]) {
    let mut i = 0;
    while i + BLK2W <= dst.len() {
        finish_block2xw(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK2W;
    }
    if i < dst.len() {
        finish_scalar2xw(&mut dst[i..], &src[i..]);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn muladd_run_2xw(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let t = make_tables2xw(coeff);
    let mut i = 0;
    while i + BLK2W <= src.len() {
        muladd_block2xw(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &t, add);
        i += BLK2W;
    }
    if i < src.len() {
        muladd_scalar2xw(&mut dst[i..], &src[i..], coeff, add);
    }
}

impl GfKernel for Shuffle2xAvx512 {
    fn id(&self) -> i32 {
        16 // GF16_SHUFFLE2X_AVX512
    }
    fn name(&self) -> &'static str {
        "Shuffle2x (AVX512)"
    }
    fn block_len(&self) -> usize {
        BLK2W
    }
    fn alignment(&self) -> usize {
        64
    }
    fn stride(&self) -> usize {
        64
    }

    fn prepare(&self, dst: &mut [u8], src: &[u8]) {
        unsafe { prepare_run_2xw(dst, src) }
    }

    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        unsafe { finish_run_2xw(dst, src) }
    }

    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe { muladd_run_2xw(dst, src, coeff, true) }
    }

    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe { muladd_run_2xw(dst, src, coeff, false) }
    }
}

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
        let mut src = [0u8; 256];
        fill(&mut src, 0x777);
        let n = blk * 2;
        let mut packed = vec![0u8; n];
        k.prepare(&mut packed, &src[..n]);
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
        // roundtrip
        let mut rt = vec![0u8; n];
        k.finish(&mut rt, &packed);
        assert_eq!(&rt[..], &src[..n]);
    }

    #[test]
    fn avx2_roundtrip_mul() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        check(&Shuffle2xAvx2, BLK2);
    }

    #[test]
    fn avx512_roundtrip_mul() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        check(&Shuffle2xAvx512, BLK2W);
    }
}

//! AVX512-VBMI shuffle kernel - port of gf16_shuffle_vbmi.c.
//!
//! Same packed layout as ShuffleAvx512 ([64 hi bytes][64 low bytes]).
//! Multiply: vpermb can index any of 64 table bytes, so the 16-bit
//! product decomposes into THREE disjoint windows (6+6+4) instead of
//! four nibble tables:
//!   w = w[5:0] | (w[11:6] << 6) | (w[15:12] << 12)
//!   c*w = c*w[5:0] ^ c*(w[11:6]<<6) ^ c*(w[15:12]<<12)
//! Each table is 64 entries; products come out as lo/hi byte planes.

#![cfg(target_arch = "x86_64")]

use crate::gf::gf16_mul2;
use crate::kernel::GfKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct ShuffleVbmi;

const BLK: usize = 128; // packed block: 64 u16 words

/// Six 64-entry vpermb tables (lo/hi byte planes x 3 windows).
struct Tables {
    lo: [__m512i; 3],
    hi: [__m512i; 3],
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn make_tables(coeff: u16) -> Tables {
    // table entry x = coeff*(x<<shift) = c_s*x, c_s = coeff*2^shift; c*x
    // for 6-bit x is the XOR of xtime^j(c) over set bits j of x. Avoids
    // the bit-serial gf16_mul loop per entry.
    let t = |c: u16| -> ([u8; 64], [u8; 64]) {
        let mut d = [0u16; 6];
        d[0] = c;
        for j in 1..6 {
            d[j] = gf16_mul2(d[j - 1]);
        }
        let mut lo = [0u8; 64];
        let mut hi = [0u8; 64];
        for x in 1..64usize {
            let mut p = 0u16;
            for j in 0..6 {
                if x & (1 << j) != 0 {
                    p ^= d[j];
                }
            }
            lo[x] = p as u8;
            hi[x] = (p >> 8) as u8;
        }
        (lo, hi)
    };
    let x6 = |mut v: u16| {
        for _ in 0..6 {
            v = gf16_mul2(v);
        }
        v
    };
    let (l0, h0) = t(coeff);
    let (l1, h1) = t(x6(coeff));
    let (l2, h2) = t(x6(x6(coeff)));
    let load = |v: &[u8; 64]| _mm512_loadu_si512(v.as_ptr() as _);
    Tables {
        lo: [load(&l0), load(&l1), load(&l2)],
        hi: [load(&h0), load(&h1), load(&h2)],
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn muladd_block(dst: *mut u8, src: *const u8, t: &Tables, add: bool) {
    let ta = _mm512_loadu_si512(src as _); // hi bytes
    let tb = _mm512_loadu_si512(src.add(64) as _); // lo bytes

    // idx1 = w[11:6] = {H[3:0], L[7:6]} per byte: (ta&0x0f)<<2 has no
    // epi16 bleed (4-bit values shifted by 2 stay within their byte);
    // tb>>6 bleeds the next byte's low 2 bits into bits 2-3, masked by 3.
    let idx1 = _mm512_or_si512(
        _mm512_slli_epi16(_mm512_and_si512(ta, _mm512_set1_epi8(0x0f)), 2),
        _mm512_and_si512(_mm512_srli_epi16(tb, 6), _mm512_set1_epi8(3)),
    );
    // idx2 = w[15:12] = H[7:4] = (ta>>4)&0x0f (epi16 bleed at bits 4-7 masked)
    let idx2 = _mm512_and_si512(_mm512_srli_epi16(ta, 4), _mm512_set1_epi8(0x0f));
    // idx0 = w[5:0] = L[5:0] - vpermb masks index bits to 6 anyway

    let mut tpl = _mm512_permutexvar_epi8(tb, t.lo[0]);
    let mut tph = _mm512_permutexvar_epi8(tb, t.hi[0]);
    tpl = _mm512_xor_si512(tpl, _mm512_permutexvar_epi8(idx1, t.lo[1]));
    tph = _mm512_xor_si512(tph, _mm512_permutexvar_epi8(idx1, t.hi[1]));
    tpl = _mm512_xor_si512(tpl, _mm512_permutexvar_epi8(idx2, t.lo[2]));
    tph = _mm512_xor_si512(tph, _mm512_permutexvar_epi8(idx2, t.hi[2]));

    if add {
        tpl = _mm512_xor_si512(tpl, _mm512_loadu_si512(dst.add(64) as _));
        tph = _mm512_xor_si512(tph, _mm512_loadu_si512(dst as _));
    }
    _mm512_storeu_si512(dst as _, tph);
    _mm512_storeu_si512(dst.add(64) as _, tpl);
}

// Per-call loops must be #[target_feature] fns (see shuffle_avx2).
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn prepare_run(dst: &mut [u8], src: &[u8]) {
    let mut i = 0;
    while i + BLK <= src.len() {
        crate::shuffle_avx512::prepare_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK;
    }
    if i < src.len() {
        crate::shuffle_avx512::prepare_scalar(&mut dst[i..], &src[i..]);
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
        crate::shuffle_avx512::finish_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i));
        i += BLK;
    }
    if i < dst.len() {
        crate::shuffle_avx512::finish_scalar(&mut dst[i..], &src[i..]);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn muladd_run(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let t = make_tables(coeff);
    let mut i = 0;
    while i + BLK <= src.len() {
        muladd_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &t, add);
        i += BLK;
    }
    if i < src.len() {
        crate::shuffle_avx512::muladd_scalar(&mut dst[i..], &src[i..], coeff, add);
    }
}

impl GfKernel for ShuffleVbmi {
    fn id(&self) -> i32 {
        14 // GF16_SHUFFLE_VBMI
    }
    fn name(&self) -> &'static str {
        "Shuffle (VBMI)"
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
        std::arch::is_x86_feature_detected!("avx512vbmi")
    }

    fn fill(buf: &mut [u8], seed: u64) {
        let mut rng = seed;
        for (i, b) in buf.iter_mut().enumerate() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            *b = (rng >> 33) as u8;
        }
    }

    #[test]
    fn mul_vs_scalar() {
        if !have() {
            return;
        }
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x987);
        let k = ShuffleVbmi;
        let mut packed = [0u8; BLK * 2];
        k.prepare(&mut packed, &src);
        for coeff in [1u16, 2, 0x1234, 0xffff, 0x8001, 0x4000, 0x00ff] {
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
        let k = ShuffleVbmi;
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

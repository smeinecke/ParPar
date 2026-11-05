//! NEON CLMul kernel (aarch64, +pmull) - port of gf16_clmul_neon.c /
//! gf16_clmul_neon_base.h / gf16_clmul_neon.h.
//!
//! Packed block: 32 bytes covering 16 u16 words in natural byte order
//! (same layout as ShuffleNeon - vld2q deinterleaves at load time).
//!
//! Multiply: coefficient c and word w decompose as c_lo + c_hi*x^8 and
//! w_lo + w_hi*x^8; the 16-bit polynomial products come from PMULL on
//! lo/hi/mid (Karatsuba) byte pairs, then Barrett-reduce mod 0x1100b.

#![cfg(target_arch = "aarch64")]

use crate::gf::gf16_mul;
use crate::kernel::GfKernel;
use std::arch::aarch64::*;

pub struct ClmulNeon;

const BLK: usize = 32; // packed block: 16 u16 words

#[inline]
unsafe fn pmull_low(a: poly8x16_t, b: poly8x16_t) -> poly16x8_t {
    vmull_p8(vget_low_p8(a), vget_low_p8(b))
}
#[inline]
unsafe fn pmull_high(a: poly8x16_t, b: poly8x16_t) -> poly16x8_t {
    vmull_high_p8(a, b)
}
#[inline]
unsafe fn veorq_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    vreinterpretq_p8_u8(veorq_u8(vreinterpretq_u8_p8(a), vreinterpretq_u8_p8(b)))
}
#[inline]
unsafe fn veorq_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    vreinterpretq_p16_u16(veorq_u16(
        vreinterpretq_u16_p16(a),
        vreinterpretq_u16_p16(b),
    ))
}
#[inline]
unsafe fn eor3q_u8(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    veorq_u8(a, veorq_u8(b, c))
}

/// Per-coefficient broadcast vectors: [c_lo, c_hi, c_lo^c_hi].
pub(crate) struct Coeffs([poly8x16_t; 3]);

#[inline]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn make_coeffs(val: u16) -> Coeffs {
    let lo = vdupq_n_p8((val & 0xff) as u8);
    let hi = vdupq_n_p8((val >> 8) as u8);
    Coeffs([lo, hi, veorq_p8(lo, hi)])
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn reduction(
    low1: &mut poly16x8_t,
    low2: &mut poly16x8_t,
    mid1: poly16x8_t,
    mid2: poly16x8_t,
    high1: &mut poly16x8_t,
    high2: &mut poly16x8_t,
) {
    // put data in proper form
    let hb = vreinterpretq_u8_p16(*high1);
    let hb2 = vreinterpretq_u8_p16(*high2);
    let mut hib0 = vuzp1q_u8(hb, hb2);
    let hib1 = vuzp2q_u8(hb, hb2);
    let lb = vreinterpretq_u8_p16(*low1);
    let lb2 = vreinterpretq_u8_p16(*low2);
    let lob0 = vuzp1q_u8(lb, lb2);
    let mut lob1 = vuzp2q_u8(lb, lb2);
    let mb = vreinterpretq_u8_p16(mid1);
    let mb2 = vreinterpretq_u8_p16(mid2);
    let mib0 = vuzp1q_u8(mb, mb2);
    let mib1 = vuzp2q_u8(mb, mb2);

    // merge mid into high/low
    let libytes = veorq_u8(hib0, lob1);
    lob1 = eor3q_u8(libytes, lob0, mib0);
    hib0 = eor3q_u8(libytes, hib1, mib1);

    // Barrett reduction; first reduction coefficient is 0x1111a,
    // multiply hibytes by 0x11110
    let mut th0 = vsriq_n_u8::<4>(vshlq_n_u8::<4>(hib1), hib0);
    let mut th1 = veorq_u8(hib1, vshrq_n_u8::<4>(hib1));
    th0 = eor3q_u8(th0, th1, hib0);

    // subsequent polynomial multiplication doesn't need the low bits of
    // th0 to be correct, so trim these now for a shorter dep chain;
    // computes th0_hi3 ^= th0_hi3 >> 2 in one table op (no SHA3 path)
    let mut th0_hi3 = vshrq_n_u8::<5>(th0);
    th0_hi3 = vqtbl1q_u8(
        vld1q_u8([0, 1, 2, 3, 5, 4, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr()),
        th0_hi3,
    );

    // mul by 0x1a => we only care about upper byte
    th0 = veorq_u8(th0, vshrq_n_u8::<5>(hib1));

    // multiply by polynomial: 0x100b
    let red_l = vdupq_n_p8(0x0b);
    let h1_new = vsliq_n_u8::<4>(th0_hi3, th0);
    th1 = vreinterpretq_u8_p8(vmulq_p8(vreinterpretq_p8_u8(th1), red_l));
    let h0_new = vreinterpretq_u8_p8(vmulq_p8(vreinterpretq_p8_u8(th0), red_l));

    *low1 = vreinterpretq_p16_u8(lob0);
    *low2 = vreinterpretq_p16_u8(h0_new);
    *high1 = vreinterpretq_p16_u8(veorq_u8(h1_new, th1));
    *high2 = vreinterpretq_p16_u8(lob1);
}

#[inline]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn muladd_block(dst: *mut u8, src: *const u8, c: &Coeffs, add: bool) {
    let data = vld2q_p8(src); // data.0 = lo bytes, data.1 = hi bytes
    let mid = veorq_p8(data.0, data.1);

    let mut low1 = pmull_low(data.0, c.0[0]);
    let mut low2 = pmull_high(data.0, c.0[0]);
    let mid1 = pmull_low(mid, c.0[2]);
    let mid2 = pmull_high(mid, c.0[2]);
    let mut high1 = pmull_low(data.1, c.0[1]);
    let mut high2 = pmull_high(data.1, c.0[1]);

    reduction(&mut low1, &mut low2, mid1, mid2, &mut high1, &mut high2);

    let r0 = vreinterpretq_u8_p16(veorq_p16(low1, low2));
    let r1 = vreinterpretq_u8_p16(veorq_p16(high1, high2));
    if add {
        let d = vld2q_u8(dst);
        vst2q_u8(dst, uint8x16x2_t(veorq_u8(r0, d.0), veorq_u8(r1, d.1)));
    } else {
        vst2q_u8(dst, uint8x16x2_t(r0, r1));
    }
}

/// Scalar tail - natural u16 word order.
pub(crate) fn muladd_scalar(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let words = (src.len() / 2).min(dst.len() / 2);
    for i in 0..words {
        let w = u16::from_le_bytes([src[2 * i], src[2 * i + 1]]);
        let p = gf16_mul(w, coeff);
        let r = if add {
            u16::from_le_bytes([dst[2 * i], dst[2 * i + 1]]) ^ p
        } else {
            p
        };
        dst[2 * i..2 * i + 2].copy_from_slice(&r.to_le_bytes());
    }
    if src.len() % 2 == 1 && 2 * words < dst.len() {
        let i = words;
        let p = gf16_mul(src[2 * i] as u16, coeff);
        let r = if add { (dst[2 * i] as u16) ^ p } else { p };
        dst[2 * i] = r as u8;
    }
}

// Per-call loops must be #[target_feature] fns (see shuffle_avx2).
#[target_feature(enable = "aes")]
unsafe fn muladd_run(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
    let c = make_coeffs(coeff);
    let mut i = 0;
    while i + BLK <= src.len() {
        muladd_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &c, add);
        i += BLK;
    }
    if i < src.len() {
        muladd_scalar(&mut dst[i..], &src[i..], coeff, add);
    }
}

impl GfKernel for ClmulNeon {
    fn id(&self) -> i32 {
        28 // GF16_CLMUL_NEON
    }
    fn name(&self) -> &'static str {
        "CLMul (NEON)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        32
    }
    fn stride(&self) -> usize {
        32
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
        unsafe { muladd_run(dst, src, coeff, true) }
    }

    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe { muladd_run(dst, src, coeff, false) }
    }
}

/// Method 29: the C++ SHA3 variant differs only in multi-source merge
/// (veor3q) - single-source multiply is identical to CLMul NEON.
pub struct ClmulSha3;

impl GfKernel for ClmulSha3 {
    fn id(&self) -> i32 {
        29 // GF16_CLMUL_SHA3
    }
    fn name(&self) -> &'static str {
        "CLMul (SHA3)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        32
    }
    fn stride(&self) -> usize {
        32
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

    fn fill(buf: &mut [u8], seed: u64) {
        let mut rng = seed;
        for (i, b) in buf.iter_mut().enumerate() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            *b = (rng >> 33) as u8;
        }
    }

    #[test]
    fn prepare_finish_roundtrip() {
        let k = ClmulNeon;
        for len in [BLK, BLK * 3, 7usize, 33, 0] {
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
        if !std::arch::is_aarch64_feature_detected!("aes") {
            return;
        }
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x987);
        let k = ClmulNeon;
        for coeff in [1u16, 2, 0x1234, 0xffff, 0x8001, 0x4000, 0x0100, 0x00ff] {
            let mut dst = [0u8; BLK * 2];
            k.mul(&mut dst, &src, coeff);
            for w in 0..(src.len() / 2) {
                let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
                let ow = (dst[2 * w + 1] as u16) << 8 | dst[2 * w] as u16;
                assert_eq!(ow, gf16_mul(sw, coeff), "w{} coeff {:x}", w, coeff);
            }
        }
    }

    #[test]
    fn sha3_mul_vs_scalar() {
        if !std::arch::is_aarch64_feature_detected!("sha3") {
            return;
        }
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x31337);
        let k = ClmulSha3;
        let mut dst = [0u8; BLK * 2];
        let mut out = [0u8; BLK * 2];
        k.mul(&mut dst, &src, 0x1234);
        k.finish(&mut out, &dst);
        for w in 0..(src.len() / 2) {
            let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
            let ow = (out[2 * w + 1] as u16) << 8 | out[2 * w] as u16;
            assert_eq!(ow, gf16_mul(sw, 0x1234), "w{}", w);
        }
        // auto-pick prefers sha3 when available
        assert_eq!(crate::kernel::pick(0).id(), 29);
    }

    #[test]
    fn muladd_accumulates() {
        if !std::arch::is_aarch64_feature_detected!("aes") {
            return;
        }
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x55);
        let mut d0 = [0u8; BLK * 2];
        fill(&mut d0, 0xaa);
        let k = ClmulNeon;
        let mut dst = d0;
        k.muladd(&mut dst, &src, 0xbeef);
        k.muladd(&mut dst, &src, 0x1234);
        for w in 0..(src.len() / 2) {
            let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
            let d0w = (d0[2 * w + 1] as u16) << 8 | d0[2 * w] as u16;
            let want = d0w ^ gf16_mul(sw, 0xbeef) ^ gf16_mul(sw, 0x1234);
            let ow = (dst[2 * w + 1] as u16) << 8 | dst[2 * w] as u16;
            assert_eq!(ow, want, "w{}", w);
        }
    }
}

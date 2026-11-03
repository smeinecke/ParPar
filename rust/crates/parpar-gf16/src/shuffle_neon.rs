//! NEON shuffle kernel (aarch64) - port of gf16_shuffle_neon.c.
//!
//! Packed block: 32 bytes covering 16 u16 words in natural byte order -
//! vld2q_u8 deinterleaves low/high byte planes at load time and vst2q_u8
//! reinterleaves on store, so prepare/finish are plain copies.
//!
//! Multiply: same 4x16-entry nibble-table math as the x86 shuffle kernels -
//! tbl[k] maps nibble value v to GF16 product coeff*(v<<(4k)), split into
//! low/high byte planes for vqtbl1q_u8 lookups.

#![cfg(target_arch = "aarch64")]

use crate::gf::gf16_mul;
use crate::kernel::GfKernel;
use std::arch::aarch64::*;

pub struct ShuffleNeon;

const BLK: usize = 32; // packed block: 16 u16 words

/// Eight 16-entry vqtbl1q tables for a coefficient (low/high byte planes x4).
struct Tables {
    l: [uint8x16_t; 4],
    h: [uint8x16_t; 4],
}

unsafe fn make_tables(coeff: u16) -> Tables {
    let t = |k: u32| -> (uint8x16_t, uint8x16_t) {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for x in 0..16usize {
            let p = gf16_mul(coeff, (x as u16) << k);
            lo[x] = p as u8;
            hi[x] = (p >> 8) as u8;
        }
        (vld1q_u8(lo.as_ptr()), vld1q_u8(hi.as_ptr()))
    };
    let (l0, h0) = t(0);
    let (l1, h1) = t(4);
    let (l2, h2) = t(8);
    let (l3, h3) = t(12);
    Tables {
        l: [l0, l1, l2, l3],
        h: [h0, h1, h2, h3],
    }
}

#[inline]
unsafe fn muladd_block(dst: *mut u8, src: *const u8, t: &Tables, add: bool) {
    let nmask = vdupq_n_u8(0x0f);
    let v = vld2q_u8(src); // v.0 = low bytes, v.1 = high bytes of 16 words
    let ll = vandq_u8(v.0, nmask);
    let lh = vshrq_n_u8(v.0, 4);
    let hl = vandq_u8(v.1, nmask);
    let hh = vshrq_n_u8(v.1, 4);

    // gf16_shuffle_neon_round1: low byte nibbles use tables 0/1,
    // high byte nibbles use tables 2/3
    let mut rl = vqtbl1q_u8(t.l[0], ll);
    rl = veorq_u8(rl, vqtbl1q_u8(t.l[1], lh));
    rl = veorq_u8(rl, vqtbl1q_u8(t.l[2], hl));
    rl = veorq_u8(rl, vqtbl1q_u8(t.l[3], hh));
    let mut rh = vqtbl1q_u8(t.h[0], ll);
    rh = veorq_u8(rh, vqtbl1q_u8(t.h[1], lh));
    rh = veorq_u8(rh, vqtbl1q_u8(t.h[2], hl));
    rh = veorq_u8(rh, vqtbl1q_u8(t.h[3], hh));

    if add {
        let d = vld2q_u8(dst);
        rl = veorq_u8(rl, d.0);
        rh = veorq_u8(rh, d.1);
    }
    vst2q_u8(dst, uint8x16x2_t(rl, rh));
}

/// Scalar tail - natural u16 word order, so a partial block is just
/// trailing words/bytes at the end of the region.
fn muladd_scalar(dst: &mut [u8], src: &[u8], coeff: u16, add: bool) {
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
        // odd trailing byte forms a lo-only word (hi = 0)
        let i = words;
        let w = src[2 * i] as u16;
        let p = gf16_mul(w, coeff);
        let r = if add { (dst[2 * i] as u16) ^ p } else { p };
        dst[2 * i] = r as u8;
    }
}

impl GfKernel for ShuffleNeon {
    fn id(&self) -> i32 {
        4 // GF16_SHUFFLE_NEON
    }
    fn name(&self) -> &'static str {
        "Shuffle (NEON)"
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
        dst[..src.len()].copy_from_slice(src);
        for b in dst[src.len()..].iter_mut() {
            *b = 0;
        }
    }
    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        dst.copy_from_slice(&src[..dst.len()]);
    }

    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe {
            let t = make_tables(coeff);
            let mut i = 0;
            while i + BLK <= src.len() {
                muladd_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &t, true);
                i += BLK;
            }
            if i < src.len() {
                muladd_scalar(&mut dst[i..], &src[i..], coeff, true);
            }
        }
    }

    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        unsafe {
            let t = make_tables(coeff);
            let mut i = 0;
            while i + BLK <= src.len() {
                muladd_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &t, false);
                i += BLK;
            }
            if i < src.len() {
                muladd_scalar(&mut dst[i..], &src[i..], coeff, false);
            }
        }
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
        let k = ShuffleNeon;
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
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x987);
        let k = ShuffleNeon;
        let mut packed = [0u8; BLK * 2];
        k.prepare(&mut packed, &src);
        for coeff in [1u16, 2, 0x1234, 0xffff, 0x8001, 0x4000] {
            let mut dst = [0u8; BLK * 2];
            k.mul(&mut dst, &packed, coeff);
            for w in 0..(src.len() / 2) {
                let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
                let ow = (dst[2 * w + 1] as u16) << 8 | dst[2 * w] as u16;
                assert_eq!(ow, gf16_mul(sw, coeff), "w{} coeff {:x}", w, coeff);
            }
        }
    }

    #[test]
    fn muladd_accumulates() {
        let mut src = [0u8; BLK * 2];
        fill(&mut src, 0x55);
        let mut d0 = [0u8; BLK * 2];
        fill(&mut d0, 0xaa);
        let k = ShuffleNeon;
        let mut dst = [0u8; BLK * 2];
        dst.copy_from_slice(&d0);
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

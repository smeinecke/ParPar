//! Static AVX2 port of the Xor-Jit multiply (gf16_xor_avx2.c semantics).
//!
//! The C++ emits specialized machine code per coefficient; this version keeps
//! the same packed bit-plane layout and dependency math but runs a fixed
//! loop - all planes are 32-byte ymm rows so the core is pure vpxor.
//!
//! Packed block = 512 bytes = 16 plane rows of 32B. Row r (0..16) holds the
//! bit-plane of word-bit (15-r): rows 0..7 are high-byte bits 7..0 of the
//! high byte (word bits 15..8), rows 8..15 the low byte (bits 7..0).
//! Within a row, u32 word j = plane bits for input group j (32 words).

#![cfg(target_arch = "x86_64")]

use crate::gf::gf16_mul;
use crate::kernel::GfKernel;
use std::arch::x86_64::*;

pub struct XorAvx2;

const BLK: usize = 512; // 256 words per block
const GROUP: usize = 64; // 32 words per bit-plane group

/// Split a 64-byte group into hi/lo byte vectors (gf16_xor_prep_split, AVX2).
#[target_feature(enable = "avx2")]
unsafe fn prep_split(ta: __m256i, tb: __m256i) -> (__m256i, __m256i) {
    let pa = _mm256_set_epi32(
        0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200, 0x0f0d0b09, 0x07050301, 0x0e0c0a08,
        0x06040200,
    );
    let pb = _mm256_set_epi32(
        0x0e0c0a08, 0x06040200, 0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200, 0x0f0d0b09,
        0x07050301,
    );
    let t1 = _mm256_shuffle_epi8(ta, pa);
    let t2 = _mm256_shuffle_epi8(tb, pb);
    let th = _mm256_permute4x64_epi64(_mm256_blend_epi32(t1, t2, 0x33), 0b10_00_11_01);
    let tl = _mm256_permute4x64_epi64(_mm256_blend_epi32(t2, t1, 0x33), 0b11_01_10_00);
    (th, tl)
}

/// movemask + double: write 8 bit-planes of `v` to `dst` (u32 index `j`).
#[target_feature(enable = "avx2")]
unsafe fn prep_write(dst: *mut u32, j: usize, mut v: __m256i) {
    for i in 0..8 {
        dst.add(i * 8 + j)
            .write_unaligned(_mm256_movemask_epi8(v) as u32);
        v = _mm256_add_epi8(v, v);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn prepare_group(block: *mut u8, src: *const u8, j: usize) {
    let ta = _mm256_loadu_si256(src as _);
    let tb = _mm256_loadu_si256(src.add(32) as _);
    let (th, tl) = prep_split(ta, tb);
    let d = block as *mut u32;
    prep_write(d, j, th);
    prep_write(d.add(64), j, tl);
}

/// Rebuild word w_in_group of group j from the plane u32 at
/// row_base + (7-b)*32 + j*4 (hi) / +256 (lo). `src` is the block base.
fn finish_group(dst: &mut [u8], src: &[u8], j: usize, words: usize) {
    let plane = |r: usize| -> u32 {
        unsafe { (src.as_ptr().add(r * 32 + j * 4) as *const u32).read_unaligned() }
    };
    for w in 0..words {
        let mut lo = 0u8;
        let mut hi = 0u8;
        for b in 0..8 {
            hi |= (((plane(7 - b) >> w) & 1) as u8) << b;
            lo |= (((plane(8 + 7 - b) >> w) & 1) as u8) << b;
        }
        dst[2 * w] = lo;
        if 2 * w + 1 < dst.len() {
            dst[2 * w + 1] = hi;
        }
    }
}

/// deps[k] = bitmask of input bits contributing to output bit k for coeff.
fn make_deps(coeff: u16) -> [u32; 16] {
    let mut deps = [0u32; 16];
    for i in 0..16 {
        let v = gf16_mul(coeff, 1u16 << i);
        for k in 0..16 {
            if (v >> k) & 1 == 1 {
                deps[k] |= 1 << i;
            }
        }
    }
    deps
}

/// One 512B block: out row (15-k) = XOR of in rows (15-i) for i in deps[k].
#[target_feature(enable = "avx2")]
unsafe fn mul_block(dst: *mut u8, src: *const u8, deps: &[u32; 16], add: bool) {
    let mut p = [_mm256_setzero_si256(); 16];
    for r in 0..16 {
        p[r] = _mm256_loadu_si256(src.add(r * 32) as _);
    }
    for k in 0..16 {
        let mut m = deps[k];
        let mut acc = _mm256_setzero_si256();
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            acc = _mm256_xor_si256(acc, p[15 - i]);
            m &= m - 1;
        }
        let r = (15 - k) * 32;
        if add {
            acc = _mm256_xor_si256(acc, _mm256_loadu_si256(dst.add(r) as _));
        }
        _mm256_storeu_si256(dst.add(r) as _, acc);
    }
}

impl GfKernel for XorAvx2 {
    fn id(&self) -> i32 {
        crate::kernel::method::XORJIT_AVX2
    }
    fn name(&self) -> &'static str {
        "Xor-Jit (AVX2)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        32
    }
    fn stride(&self) -> usize {
        512
    }

    fn prepare(&self, dst: &mut [u8], src: &[u8]) {
        let mut blk = 0usize;
        while blk < dst.len() {
            let bdst = &mut dst[blk..blk + BLK];
            for j in 0..8 {
                let off = j * GROUP;
                let have = src.len().saturating_sub(blk + off);
                if have >= GROUP {
                    unsafe { prepare_group(bdst.as_mut_ptr(), src.as_ptr().add(blk + off), j) };
                } else {
                    // partial/absent group: zero-pad then transform (or zero)
                    let mut tmp = [0u8; GROUP];
                    if have > 0 {
                        tmp[..have].copy_from_slice(&src[blk + off..blk + off + have]);
                        unsafe { prepare_group(bdst.as_mut_ptr(), tmp.as_ptr(), j) };
                    } else {
                        for p in 0..16 {
                            for b in bdst[p * 32 + j * 4..p * 32 + j * 4 + 4].iter_mut() {
                                *b = 0;
                            }
                        }
                    }
                }
            }
            blk += BLK;
        }
    }

    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        let mut done = 0usize;
        while done < dst.len() {
            let blk = done - done % BLK;
            for j in 0..8 {
                if done >= dst.len() {
                    break;
                }
                let rem = (dst.len() - done).min(GROUP);
                let words = rem.div_ceil(2);
                finish_group(&mut dst[done..done + rem], &src[blk..], j, words);
                done += rem;
            }
        }
    }

    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        if coeff == 0 {
            return;
        }
        let deps = make_deps(coeff);
        unsafe {
            let mut i = 0;
            while i + BLK <= src.len() {
                mul_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &deps, true);
                i += BLK;
            }
            debug_assert_eq!(src.len() - i, 0, "muladd region must be block-aligned");
        }
    }

    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        if coeff == 0 {
            for b in dst.iter_mut() {
                *b = 0;
            }
            return;
        }
        let deps = make_deps(coeff);
        unsafe {
            let mut i = 0;
            while i + BLK <= src.len() {
                mul_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), &deps, false);
                i += BLK;
            }
            debug_assert_eq!(src.len() - i, 0, "mul region must be block-aligned");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::gf16_mul;

    #[test]
    fn prepare_finish_roundtrip() {
        let mut src = [0u8; BLK];
        let mut rng = 0x1234u64;
        for (i, b) in src.iter_mut().enumerate() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            *b = (rng >> 33) as u8;
        }
        let k = XorAvx2;
        let mut packed = [0u8; BLK];
        k.prepare(&mut packed, &src);
        let mut out = [0u8; BLK];
        k.finish(&mut out, &packed);
        assert_eq!(out, src);
    }

    #[test]
    fn mul_vs_scalar() {
        let mut src = [0u8; BLK];
        let mut rng = 0x987u64;
        for b in src.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 33) as u8;
        }
        let k = XorAvx2;
        let mut packed = [0u8; BLK];
        k.prepare(&mut packed, &src);
        for coeff in [1u16, 2, 0x1234, 0xffff, 0x8001] {
            let mut dst = [0u8; BLK];
            k.mul(&mut dst, &packed, coeff);
            let mut out = [0u8; BLK];
            k.finish(&mut out, &dst);
            for w in 0..256 {
                let sw = (src[2 * w + 1] as u16) << 8 | src[2 * w] as u16;
                let ow = (out[2 * w + 1] as u16) << 8 | out[2 * w] as u16;
                assert_eq!(ow, gf16_mul(sw, coeff), "w{} coeff {:x}", w, coeff);
            }
        }
    }
}

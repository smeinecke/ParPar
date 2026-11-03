//! Static AVX512 port of the Xor-Jit multiply (gf16_xor_avx512.c semantics).
//!
//! Same packed bit-plane layout as XorAvx2 but with 64-byte rows: a block
//! is 1024 bytes = 16 plane rows of 64B; row r u64 lane j holds the bit
//! plane for 64 words of group j (128-byte source groups, 8 per block).
//! Rows 0..7 are hi-byte bits 7..0 (word bits 15..8), rows 8..15 the lo
//! byte (bits 7..0).

#![cfg(target_arch = "x86_64")]

use crate::gf::gf16_mul;
use crate::kernel::GfKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct XorAvx512;

const BLK: usize = 1024; // 512 words per block
const GROUP: usize = 128; // 64 words per bit-plane group

/// movepi8 + double: write 8 bit-planes of `v` to `dst` (u64 index `j`).
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn prep_write(dst: *mut u64, j: usize, mut v: __m512i) {
    for i in 0..8 {
        dst.add(i * 8 + j).write_unaligned(_mm512_movepi8_mask(v));
        v = _mm512_add_epi8(v, v);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn prepare_group(block: *mut u8, src: *const u8, j: usize) {
    // split 128B into [64 hi bytes][64 lo bytes] then extract planes
    let mut tmp = [0u8; 128];
    crate::shuffle_avx512::prepare_block(tmp.as_mut_ptr(), src);
    let th = _mm512_loadu_si512(tmp.as_ptr() as _);
    let tl = _mm512_loadu_si512(tmp.as_ptr().add(64) as _);
    let d = block as *mut u64;
    prep_write(d, j, th);
    prep_write(d.add(64), j, tl);
}

/// Rebuild word w_in_group of group j from the plane u64 at
/// row_base + (7-b)*64 + j*8 (hi) / +512 (lo). `src` is the block base.
fn finish_group(dst: &mut [u8], src: &[u8], j: usize, words: usize) {
    let plane = |r: usize| -> u64 {
        unsafe { (src.as_ptr().add(r * 64 + j * 8) as *const u64).read_unaligned() }
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

/// One 1024B block: out row (15-k) = XOR of in rows (15-i) for i in deps[k].
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn mul_block(dst: *mut u8, src: *const u8, deps: &[u32; 16], add: bool) {
    let mut p = [_mm512_setzero_si512(); 16];
    for i in 0..16 {
        p[i] = _mm512_loadu_si512(src.add(i * 64) as _);
    }
    for k in 0..16 {
        let mut m = deps[k];
        let mut acc = _mm512_setzero_si512();
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            acc = _mm512_xor_si512(acc, p[15 - i]);
            m &= m - 1;
        }
        let r = (15 - k) * 64;
        if add {
            acc = _mm512_xor_si512(acc, _mm512_loadu_si512(dst.add(r) as _));
        }
        _mm512_storeu_si512(dst.add(r) as _, acc);
    }
}

impl GfKernel for XorAvx512 {
    fn id(&self) -> i32 {
        crate::kernel::method::XORJIT_AVX512
    }
    fn name(&self) -> &'static str {
        "Xor-Jit (AVX512)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        64
    }
    fn stride(&self) -> usize {
        1024
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
                    let mut tmp = [0u8; GROUP];
                    if have > 0 {
                        tmp[..have].copy_from_slice(&src[blk + off..blk + off + have]);
                        unsafe { prepare_group(bdst.as_mut_ptr(), tmp.as_ptr(), j) };
                    } else {
                        for p in 0..16 {
                            for b in bdst[p * 64 + j * 8..p * 64 + j * 8 + 8].iter_mut() {
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
                let words = (rem + 1) / 2;
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

    fn have() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
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
        let mut src = vec![0u8; BLK + 100];
        fill(&mut src, 0xdecaf);
        let k = XorAvx512;
        let mut packed = vec![0u8; BLK * 2];
        k.prepare(&mut packed, &src);
        let mut out = vec![0u8; src.len()];
        k.finish(&mut out, &packed);
        assert_eq!(out, src);
    }

    #[test]
    fn mul_vs_scalar() {
        if !have() {
            return;
        }
        let mut src = vec![0u8; BLK * 2];
        fill(&mut src, 0x777);
        let k = XorAvx512;
        let mut packed = vec![0u8; BLK * 2];
        k.prepare(&mut packed, &src);
        for coeff in [1u16, 2, 0x1234, 0xffff, 0x8001, 0x00ff, 0x4000, 3] {
            let mut dst = vec![0u8; BLK * 2];
            let mut out = vec![0u8; BLK * 2];
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
        let mut src = vec![0u8; BLK * 2];
        fill(&mut src, 0x33);
        let mut d0 = vec![0u8; BLK * 2];
        fill(&mut d0, 0x66);
        let k = XorAvx512;
        let mut packed = vec![0u8; BLK * 2];
        k.prepare(&mut packed, &src);
        let mut dst = vec![0u8; BLK * 2];
        k.prepare(&mut dst, &d0);
        k.muladd(&mut dst, &packed, 0xbeef);
        k.muladd(&mut dst, &packed, 0x1234);
        let mut out = vec![0u8; BLK * 2];
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

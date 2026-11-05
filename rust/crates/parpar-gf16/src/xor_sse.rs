//! Static SSE2 port of the Xor multiply family (gf16_xor_sse2.c semantics).
//!
//! Same bit-plane layout as XorAvx2 at 128-bit width: packed block = 256
//! bytes = 16 plane rows of 16B. Row r holds the bit-plane of word-bit
//! (15-r); rows 0..7 = high-byte bits, 8..15 = low-byte. Within a row,
//! u32 word j (0..4) covers input group j of 32 words.

#![cfg(target_arch = "x86_64")]

use crate::gf::gf16_mul2;
use crate::kernel::GfKernel;
use std::arch::x86_64::*;

pub struct XorSse;

const BLK: usize = 256; // 128 words per block
const GROUP: usize = 64; // 32 words per bit-plane group

/// Split a 64B group into hi/lo byte vectors - 4 xmm words: [h01 h23 l01 l23]
/// where h01 = hi bytes of words 0..15, h23 = words 16..31, same for lo.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn prep_group(src: *const u8) -> (__m128i, __m128i, __m128i, __m128i) {
    let mask = _mm_set1_epi16(0x00ff);
    let x0 = _mm_loadu_si128(src as _);
    let x1 = _mm_loadu_si128(src.add(16) as _);
    let x2 = _mm_loadu_si128(src.add(32) as _);
    let x3 = _mm_loadu_si128(src.add(48) as _);
    let h01 = _mm_packus_epi16(_mm_srli_epi16(x0, 8), _mm_srli_epi16(x1, 8));
    let h23 = _mm_packus_epi16(_mm_srli_epi16(x2, 8), _mm_srli_epi16(x3, 8));
    let l01 = _mm_packus_epi16(_mm_and_si128(x0, mask), _mm_and_si128(x1, mask));
    let l23 = _mm_packus_epi16(_mm_and_si128(x2, mask), _mm_and_si128(x3, mask));
    (h01, h23, l01, l23)
}

/// movemask + double: write 8 bit-planes of the (a,b) pair to `dst` u32 j.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn prep_write(dst: *mut u32, j: usize, mut a: __m128i, mut b: __m128i) {
    for i in 0..8 {
        let m = (_mm_movemask_epi8(b) << 16) | _mm_movemask_epi8(a);
        dst.add(i * 4 + j).write_unaligned(m as u32);
        a = _mm_add_epi8(a, a);
        b = _mm_add_epi8(b, b);
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn prepare_group(block: *mut u8, src: *const u8, j: usize) {
    let (h01, h23, l01, l23) = prep_group(src);
    let d = block as *mut u32;
    prep_write(d, j, h01, h23);
    prep_write(d.add(32), j, l01, l23);
}

/// Rebuild words of group j from plane u32 at row_base + r*16 + j*4.
fn finish_group(dst: &mut [u8], src: &[u8], j: usize, words: usize) {
    let plane = |r: usize| -> u32 {
        unsafe { (src.as_ptr().add(r * 16 + j * 4) as *const u32).read_unaligned() }
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
    // coeff*2^i = xtime^i(coeff) - a mul2 chain replaces bit-serial muls
    let mut deps = [0u32; 16];
    let mut v = coeff;
    for i in 0..16 {
        for k in 0..16 {
            if (v >> k) & 1 == 1 {
                deps[k] |= 1 << i;
            }
        }
        v = gf16_mul2(v);
    }
    deps
}

/// One 256B block: out row (15-k) = XOR of in rows (15-i) for i in deps[k].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn mul_block(dst: *mut u8, src: *const u8, deps: &[u32; 16], add: bool) {
    let mut p = [_mm_setzero_si128(); 16];
    for r in 0..16 {
        p[r] = _mm_loadu_si128(src.add(r * 16) as _);
    }
    for k in 0..16 {
        let mut m = deps[k];
        let mut acc = _mm_setzero_si128();
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            acc = _mm_xor_si128(acc, p[15 - i]);
            m &= m - 1;
        }
        let r = (15 - k) * 16;
        if add {
            acc = _mm_xor_si128(acc, _mm_loadu_si128(dst.add(r) as _));
        }
        _mm_storeu_si128(dst.add(r) as _, acc);
    }
}

// Per-call loops must be #[target_feature] fns (see shuffle_avx2).
#[target_feature(enable = "sse2")]
unsafe fn prepare_run(dst: &mut [u8], src: &[u8]) {
    let mut blk = 0usize;
    while blk < dst.len() {
        let bdst = &mut dst[blk..blk + BLK];
        for j in 0..4 {
            let off = j * GROUP;
            let have = src.len().saturating_sub(blk + off);
            if have >= GROUP {
                prepare_group(bdst.as_mut_ptr(), src.as_ptr().add(blk + off), j);
            } else {
                let mut tmp = [0u8; GROUP];
                if have > 0 {
                    tmp[..have].copy_from_slice(&src[blk + off..blk + off + have]);
                    prepare_group(bdst.as_mut_ptr(), tmp.as_ptr(), j);
                } else {
                    for p in 0..16 {
                        for b in bdst[p * 16 + j * 4..p * 16 + j * 4 + 4].iter_mut() {
                            *b = 0;
                        }
                    }
                }
            }
        }
        blk += BLK;
    }
}

#[target_feature(enable = "sse2")]
unsafe fn muladd_run(dst: &mut [u8], src: &[u8], deps: &[u32; 16], add: bool) {
    let mut i = 0;
    while i + BLK <= src.len() {
        mul_block(dst.as_mut_ptr().add(i), src.as_ptr().add(i), deps, add);
        i += BLK;
    }
    debug_assert_eq!(src.len() - i, 0, "mul region must be block-aligned");
}

impl GfKernel for XorSse {
    fn id(&self) -> i32 {
        crate::kernel::method::XOR_SSE
    }
    fn name(&self) -> &'static str {
        "Xor (SSE2)"
    }
    fn block_len(&self) -> usize {
        BLK
    }
    fn alignment(&self) -> usize {
        16
    }
    fn stride(&self) -> usize {
        256
    }

    fn prepare(&self, dst: &mut [u8], src: &[u8]) {
        unsafe { prepare_run(dst, src) }
    }

    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        let mut done = 0usize;
        while done < dst.len() {
            let blk = done - done % BLK;
            for j in 0..4 {
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
        unsafe { muladd_run(dst, src, &deps, true) }
    }

    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        if coeff == 0 {
            for b in dst.iter_mut() {
                *b = 0;
            }
            return;
        }
        let deps = make_deps(coeff);
        unsafe { muladd_run(dst, src, &deps, false) }
    }
}

/// Method 18: C++ generates XOR code at runtime; the static bit-plane kernel
/// computes the identical transform, so it serves both SSE XOR methods.
pub struct XorJitSse(pub XorSse);

impl GfKernel for XorJitSse {
    fn id(&self) -> i32 {
        crate::kernel::method::XORJIT_SSE
    }
    fn name(&self) -> &'static str {
        "Xor-Jit (SSE2)"
    }
    fn block_len(&self) -> usize {
        self.0.block_len()
    }
    fn alignment(&self) -> usize {
        self.0.alignment()
    }
    fn stride(&self) -> usize {
        self.0.stride()
    }
    fn prepare(&self, dst: &mut [u8], src: &[u8]) {
        self.0.prepare(dst, src)
    }
    fn finish(&self, dst: &mut [u8], src: &[u8]) {
        self.0.finish(dst, src)
    }
    fn muladd(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        self.0.muladd(dst, src, coeff)
    }
    fn mul(&self, dst: &mut [u8], src: &[u8], coeff: u16) {
        self.0.mul(dst, src, coeff)
    }
}

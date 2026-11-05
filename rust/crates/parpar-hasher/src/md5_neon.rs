//! NEON multi-buffer MD5 kernels - ports of hasher/md5mb-neon.h (4-lane
//! u32x4) and hasher/md5x2-neon.h (2-lane u32x2) using the intrinsic paths
//! (the C++ asm variants are the same algorithm tuned per core).
//!
//! Vector lane = MD5 stream; state is stored lane-major ([stream][a,b,c,d])
//! like the SSE/AVX2 paths.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
const K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];
const MI: [usize; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8,
    13, 2, 7, 12, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 0, 7, 14, 5, 12, 3, 10, 1,
    8, 15, 6, 13, 4, 11, 2, 9,
];

#[inline(always)]
unsafe fn w32(p: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(p[i * 4..i * 4 + 4].try_into().unwrap())
}

// ---------- 4-lane (md5mb): u32x4, lane = stream ----------

#[target_feature(enable = "neon")]
pub unsafe fn md5_mb4_block_neon(st: *mut [[u32; 4]; 4], ptrs: &[&[u8]; 4]) {
    let st = &mut *st;
    let mut m = [vdupq_n_u32(0); 16];
    for i in 0..16 {
        m[i] = vld1q_u32(
            [
                w32(ptrs[0], i),
                w32(ptrs[1], i),
                w32(ptrs[2], i),
                w32(ptrs[3], i),
            ]
            .as_ptr(),
        );
    }
    let ld = |w: usize| vld1q_u32([st[0][w], st[1][w], st[2][w], st[3][w]].as_ptr());
    let mut a = ld(0);
    let mut b = ld(1);
    let mut c = ld(2);
    let mut d = ld(3);
    let (a0, b0, c0, d0) = (a, b, c, d);

    // (x << r) | (x >> (32-r)) via signed vector counts (vshl: -k shifts right)
    let rot = |x: uint32x4_t, r: u32| -> uint32x4_t {
        vorrq_u32(
            vshlq_u32(x, vdupq_n_s32(r as i32)),
            vshlq_u32(x, vdupq_n_s32(r as i32 - 32)),
        )
    };
    let f_f = |b: uint32x4_t, c: uint32x4_t, d: uint32x4_t| vbslq_u32(b, c, d);
    let f_g = |b: uint32x4_t, c: uint32x4_t, d: uint32x4_t| vbslq_u32(d, b, c);
    let f_h = |b: uint32x4_t, c: uint32x4_t, d: uint32x4_t| veorq_u32(veorq_u32(d, c), b);
    let f_i = |b: uint32x4_t, c: uint32x4_t, d: uint32x4_t| veorq_u32(vornq_u32(b, d), c);

    for step in 0..64 {
        let g = match step / 16 {
            0 => f_f(b, c, d),
            1 => f_g(b, c, d),
            2 => f_h(b, c, d),
            _ => f_i(b, c, d),
        };
        let t = vaddq_u32(
            vaddq_u32(vaddq_u32(a, g), m[MI[step]]),
            vdupq_n_u32(K[step]),
        );
        a = d;
        d = c;
        c = b;
        b = vaddq_u32(b, rot(t, S[step]));
    }

    a = vaddq_u32(a, a0);
    b = vaddq_u32(b, b0);
    c = vaddq_u32(c, c0);
    d = vaddq_u32(d, d0);

    let mut t = [0u32; 4];
    for (w, v) in [a, b, c, d].iter().enumerate() {
        vst1q_u32(t.as_mut_ptr(), *v);
        for l in 0..4 {
            st[l][w] = t[l];
        }
    }
}

// ---------- 2-lane (md5x2): u32x2, lane = stream ----------

// unused: the input hasher prefers scalar on aarch64 (serial MD5 gains
// nothing from 2-lane SIMD); kept as reference/verification code
#[allow(dead_code)]
#[target_feature(enable = "neon")]
pub unsafe fn md5_block_x2_neon(state: *mut u32, src0: *const u8, src1: *const u8) {
    let s0 = std::slice::from_raw_parts(src0, 64);
    let s1 = std::slice::from_raw_parts(src1, 64);
    let st = &mut *(state as *mut [[u32; 4]; 2]);

    let mut m = [vdup_n_u32(0); 16];
    for i in 0..16 {
        m[i] = vld1_u32([w32(s0, i), w32(s1, i)].as_ptr());
    }
    let ld = |w: usize| vld1_u32([st[0][w], st[1][w]].as_ptr());
    let mut a = ld(0);
    let mut b = ld(1);
    let mut c = ld(2);
    let mut d = ld(3);
    let (a0, b0, c0, d0) = (a, b, c, d);

    let rot = |x: uint32x2_t, r: u32| -> uint32x2_t {
        vorr_u32(
            vshl_u32(x, vdup_n_s32(r as i32)),
            vshl_u32(x, vdup_n_s32(r as i32 - 32)),
        )
    };
    let f_f = |b: uint32x2_t, c: uint32x2_t, d: uint32x2_t| vbsl_u32(b, c, d);
    let f_g = |b: uint32x2_t, c: uint32x2_t, d: uint32x2_t| vbsl_u32(d, b, c);
    let f_h = |b: uint32x2_t, c: uint32x2_t, d: uint32x2_t| veor_u32(veor_u32(d, c), b);
    let f_i = |b: uint32x2_t, c: uint32x2_t, d: uint32x2_t| veor_u32(vorn_u32(b, d), c);

    for step in 0..64 {
        let g = match step / 16 {
            0 => f_f(b, c, d),
            1 => f_g(b, c, d),
            2 => f_h(b, c, d),
            _ => f_i(b, c, d),
        };
        let t = vadd_u32(vadd_u32(vadd_u32(a, g), m[MI[step]]), vdup_n_u32(K[step]));
        a = d;
        d = c;
        c = b;
        b = vadd_u32(b, rot(t, S[step]));
    }

    a = vadd_u32(a, a0);
    b = vadd_u32(b, b0);
    c = vadd_u32(c, c0);
    d = vadd_u32(d, d0);

    let mut t = [0u32; 2];
    for (w, v) in [a, b, c, d].iter().enumerate() {
        vst1_u32(t.as_mut_ptr(), *v);
        st[0][w] = t[0];
        st[1][w] = t[1];
    }
}

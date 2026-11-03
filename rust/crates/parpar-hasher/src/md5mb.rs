//! Multi-buffer MD5 (md5mb) - N independent MD5 streams compressed in
//! parallel. State is stored lane-wise ([lane][a,b,c,d]); the x86_64 paths
//! pack 4 lanes into SSE2 vectors or 8 into AVX2 (md5mb-sse.h/hasher_avx2
//! equivalents).
//!
//! Streams always advance together (equal lengths enforced by caller).

#[cfg(not(target_arch = "x86_64"))]
use crate::md5::md5_process_block;
use crate::md5::MD5_BLOCKSIZE as MD5_BLOCK;

// state[l][w] - lane l, word w (a,b,c,d)
pub struct Md5Mb<const N: usize> {
    st: [[u32; 4]; N],
    tmp: [[u8; MD5_BLOCK]; N],
    tmp_len: usize,
    data_len: u64,
}

const IV: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

/// widest group size this build can run
pub fn max_lanes() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return 8;
        }
        return 4;
    }
    // NEON is baseline on aarch64 - always a 4-lane path
    #[cfg(target_arch = "aarch64")]
    return 4;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    1
}

impl<const N: usize> Md5Mb<N> {
    pub fn new() -> Self {
        Md5Mb {
            st: [IV; N],
            tmp: [[0; MD5_BLOCK]; N],
            tmp_len: 0,
            data_len: 0,
        }
    }

    fn block(st: &mut [[u32; 4]; N], ptrs: [&[u8]; N]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if N == 8 && is_x86_feature_detected!("avx2") {
                md5_mb8_block_avx2(
                    &mut *(st as *mut _ as *mut [[u32; 4]; 8]),
                    &*(ptrs.as_ptr() as *const [&[u8]; 8]),
                );
                return;
            }
            if N == 4 {
                md5_mb4_block_sse(
                    &mut *(st as *mut _ as *mut [[u32; 4]; 4]),
                    &*(ptrs.as_ptr() as *const [&[u8]; 4]),
                );
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if N == 4 {
                unsafe {
                    crate::md5_neon::md5_mb4_block_neon(
                        &mut *(st as *mut _ as *mut [[u32; 4]; 4]),
                        &*(ptrs.as_ptr() as *const [&[u8]; 4]),
                    );
                    return;
                }
            }
            for (l, p) in ptrs.iter().enumerate() {
                md5_process_block(&mut st[l], &p[..MD5_BLOCK]);
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        for (l, p) in ptrs.iter().enumerate() {
            md5_process_block(&mut st[l], &p[..MD5_BLOCK]);
        }
        #[cfg(target_arch = "x86_64")]
        {
            // scalar fallback for N widths without a SIMD path
            for (l, p) in ptrs.iter().enumerate() {
                crate::md5::md5_process_block(&mut st[l], &p[..MD5_BLOCK]);
            }
        }
    }

    /// Feed `len` bytes of each of the N streams (equal length).
    pub fn update(&mut self, bufs: [&[u8]; N]) {
        let len = bufs[0].len();
        self.data_len += len as u64;
        let mut p = 0usize;

        if self.tmp_len > 0 {
            let wanted = MD5_BLOCK - self.tmp_len;
            let copy = len.min(wanted);
            for (i, b) in bufs.iter().enumerate() {
                self.tmp[i][self.tmp_len..self.tmp_len + copy].copy_from_slice(&b[..copy]);
            }
            if len < wanted {
                self.tmp_len += len;
                return;
            }
            Self::block(&mut self.st, std::array::from_fn(|i| &self.tmp[i][..]));
            p = wanted;
        }

        while p + MD5_BLOCK <= len {
            Self::block(
                &mut self.st,
                std::array::from_fn(|i| &bufs[i][p..p + MD5_BLOCK]),
            );
            p += MD5_BLOCK;
        }

        self.tmp_len = len - p;
        if self.tmp_len > 0 {
            for (i, b) in bufs.iter().enumerate() {
                self.tmp[i][..self.tmp_len].copy_from_slice(&b[p..]);
            }
        }
    }

    /// Finalize all N streams, write digests lane-order to out[N][16].
    /// The padding is absorbed into a copy of the state so repeated calls
    /// stay idempotent (matches C++ end()+get() split).
    pub fn finish(&mut self, out: &mut [[u8; 16]; N]) {
        // pad each lane's tmp identically (all lanes share data_len):
        // 0x80, zeros to 56 (wrapping to a second block), LE bit length
        let bit_len = self.data_len.wrapping_mul(8);
        let mut pads = [[0u8; MD5_BLOCK * 2]; N];
        for l in 0..N {
            pads[l][..self.tmp_len].copy_from_slice(&self.tmp[l][..self.tmp_len]);
            pads[l][self.tmp_len] = 0x80;
            let n = if self.tmp_len < 56 { 64 } else { 128 };
            pads[l][n - 8..n].copy_from_slice(&bit_len.to_le_bytes());
        }
        let mut st = self.st;
        let n = if self.tmp_len < 56 { 64 } else { 128 };
        for off in (0..n).step_by(MD5_BLOCK) {
            Self::block(
                &mut st,
                std::array::from_fn(|i| &pads[i][off..off + MD5_BLOCK]),
            );
        }
        for l in 0..N {
            for (w, word) in st[l].iter().enumerate() {
                out[l][w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }
}

// shared round tables
#[cfg(target_arch = "x86_64")]
const S: [i32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
const MI: [usize; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8,
    13, 2, 7, 12, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 0, 7, 14, 5, 12, 3, 10, 1,
    8, 15, 6, 13, 4, 11, 2, 9,
];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn md5_mb4_block_sse(st: *mut [[u32; 4]; 4], ptrs: &[&[u8]; 4]) {
    use std::arch::x86_64::*;
    let st = &mut *st;
    let mut m = [_mm_setzero_si128(); 16];
    for i in 0..16 {
        let w = |l: usize| u32::from_le_bytes(ptrs[l][i * 4..i * 4 + 4].try_into().unwrap()) as i32;
        m[i] = _mm_set_epi32(w(3), w(2), w(1), w(0));
    }
    let mut a = _mm_set_epi32(
        st[3][0] as i32,
        st[2][0] as i32,
        st[1][0] as i32,
        st[0][0] as i32,
    );
    let mut b = _mm_set_epi32(
        st[3][1] as i32,
        st[2][1] as i32,
        st[1][1] as i32,
        st[0][1] as i32,
    );
    let mut c = _mm_set_epi32(
        st[3][2] as i32,
        st[2][2] as i32,
        st[1][2] as i32,
        st[0][2] as i32,
    );
    let mut d = _mm_set_epi32(
        st[3][3] as i32,
        st[2][3] as i32,
        st[1][3] as i32,
        st[0][3] as i32,
    );
    let (a0, b0, c0, d0) = (a, b, c, d);

    let rot = |x: __m128i, r: i32| -> __m128i {
        let n = _mm_cvtsi32_si128(r);
        let n32 = _mm_cvtsi32_si128(32 - r);
        _mm_or_si128(_mm_sll_epi32(x, n), _mm_srl_epi32(x, n32))
    };
    let f_f = |b: __m128i, c: __m128i, d: __m128i| -> __m128i {
        _mm_xor_si128(_mm_and_si128(_mm_xor_si128(c, d), b), d)
    };
    let f_g = |b: __m128i, c: __m128i, d: __m128i| -> __m128i {
        _mm_or_si128(_mm_and_si128(d, b), _mm_andnot_si128(d, c))
    };
    let f_h =
        |b: __m128i, c: __m128i, d: __m128i| -> __m128i { _mm_xor_si128(_mm_xor_si128(d, c), b) };
    let f_i = |b: __m128i, c: __m128i, d: __m128i| -> __m128i {
        _mm_xor_si128(_mm_or_si128(_mm_xor_si128(d, _mm_set1_epi8(-1)), b), c)
    };

    for step in 0..64 {
        let g = match step / 16 {
            0 => f_f(b, c, d),
            1 => f_g(b, c, d),
            2 => f_h(b, c, d),
            _ => f_i(b, c, d),
        };
        let t = _mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(a, g), m[MI[step]]),
            _mm_set1_epi32(K[step] as i32),
        );
        a = d;
        d = c;
        c = b;
        b = _mm_add_epi32(b, rot(t, S[step]));
    }

    a = _mm_add_epi32(a, a0);
    b = _mm_add_epi32(b, b0);
    c = _mm_add_epi32(c, c0);
    d = _mm_add_epi32(d, d0);

    let mut t = [0i32; 4];
    for (w, v) in [a, b, c, d].iter().enumerate() {
        _mm_storeu_si128(t.as_mut_ptr() as _, *v);
        for l in 0..4 {
            st[l][w] = t[l] as u32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn md5_mb8_block_avx2(st: *mut [[u32; 4]; 8], ptrs: &[&[u8]; 8]) {
    use std::arch::x86_64::*;
    let st = &mut *st;
    let mut m = [_mm256_setzero_si256(); 16];
    for i in 0..16 {
        let w = |l: usize| u32::from_le_bytes(ptrs[l][i * 4..i * 4 + 4].try_into().unwrap()) as i32;
        m[i] = _mm256_set_epi32(w(7), w(6), w(5), w(4), w(3), w(2), w(1), w(0));
    }
    let ld = |w: usize| {
        _mm256_set_epi32(
            st[7][w] as i32,
            st[6][w] as i32,
            st[5][w] as i32,
            st[4][w] as i32,
            st[3][w] as i32,
            st[2][w] as i32,
            st[1][w] as i32,
            st[0][w] as i32,
        )
    };
    let mut a = ld(0);
    let mut b = ld(1);
    let mut c = ld(2);
    let mut d = ld(3);
    let (a0, b0, c0, d0) = (a, b, c, d);

    let rot = |x: __m256i, r: i32| -> __m256i {
        _mm256_or_si256(
            _mm256_sll_epi32(x, _mm_cvtsi32_si128(r)),
            _mm256_srl_epi32(x, _mm_cvtsi32_si128(32 - r)),
        )
    };
    let f_f = |b: __m256i, c: __m256i, d: __m256i| -> __m256i {
        _mm256_xor_si256(_mm256_and_si256(_mm256_xor_si256(c, d), b), d)
    };
    let f_g = |b: __m256i, c: __m256i, d: __m256i| -> __m256i {
        _mm256_or_si256(_mm256_and_si256(d, b), _mm256_andnot_si256(d, c))
    };
    let f_h = |b: __m256i, c: __m256i, d: __m256i| -> __m256i {
        _mm256_xor_si256(_mm256_xor_si256(d, c), b)
    };
    let f_i = |b: __m256i, c: __m256i, d: __m256i| -> __m256i {
        _mm256_xor_si256(
            _mm256_or_si256(_mm256_xor_si256(d, _mm256_set1_epi8(-1)), b),
            c,
        )
    };

    for step in 0..64 {
        let g = match step / 16 {
            0 => f_f(b, c, d),
            1 => f_g(b, c, d),
            2 => f_h(b, c, d),
            _ => f_i(b, c, d),
        };
        let t = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(a, g), m[MI[step]]),
            _mm256_set1_epi32(K[step] as i32),
        );
        a = d;
        d = c;
        c = b;
        b = _mm256_add_epi32(b, rot(t, S[step]));
    }

    a = _mm256_add_epi32(a, a0);
    b = _mm256_add_epi32(b, b0);
    c = _mm256_add_epi32(c, c0);
    d = _mm256_add_epi32(d, d0);

    let mut t = [0i32; 8];
    for (w, v) in [a, b, c, d].iter().enumerate() {
        _mm256_storeu_si256(t.as_mut_ptr() as _, *v);
        for l in 0..8 {
            st[l][w] = t[l] as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::md5::Md5Ctx;

    fn fill(seed: u64, len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut x = seed;
        for _ in 0..len {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push((x >> 33) as u8);
        }
        v
    }

    fn check<const N: usize>(len: usize) {
        let bufs: Vec<Vec<u8>> = (0..N as u64 + 1).map(|s| fill(s, len)).collect();
        let mut m = Md5Mb::<N>::new();
        m.update(std::array::from_fn(|i| &bufs[i][..]));
        let mut out = [[0u8; 16]; N];
        m.finish(&mut out);
        for (l, b) in bufs.iter().take(N).enumerate() {
            let mut c = Md5Ctx::new();
            c.update(b);
            assert_eq!(out[l], c.finish(), "lane {} len {}", l, len);
        }
    }

    #[test]
    fn mb_matches_scalar() {
        for &len in &[0usize, 1, 63, 64, 65, 127, 128, 129, 1000, 4096] {
            check::<4>(len);
            check::<8>(len);
        }
    }

    #[test]
    fn finish_is_idempotent() {
        // finish() absorbs padding into a state copy - a second call must
        // return the same digest (C++ separates end() from get())
        let bufs: Vec<Vec<u8>> = (0..4u64).map(|s| fill(s + 3, 200)).collect();
        let mut m = Md5Mb::<4>::new();
        m.update(std::array::from_fn(|i| &bufs[i][..]));
        let mut o1 = [[0u8; 16]; 4];
        let mut o2 = [[0u8; 16]; 4];
        m.finish(&mut o1);
        m.finish(&mut o2);
        assert_eq!(o1, o2);
    }

    #[test]
    fn mb_split_updates() {
        let bufs: Vec<Vec<u8>> = (0..8u64).map(|s| fill(s + 7, 300)).collect();
        let mut m1 = Md5Mb::<8>::new();
        m1.update(std::array::from_fn(|i| &bufs[i][..]));
        let mut o1 = [[0u8; 16]; 8];
        m1.finish(&mut o1);

        let mut m2 = Md5Mb::<8>::new();
        m2.update(std::array::from_fn(|i| &bufs[i][..170]));
        m2.update(std::array::from_fn(|i| &bufs[i][170..]));
        let mut o2 = [[0u8; 16]; 8];
        m2.finish(&mut o2);
        assert_eq!(o1, o2);
    }
}

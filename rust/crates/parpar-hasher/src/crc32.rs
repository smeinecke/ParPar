//! CRC32 (reflected, poly 0xEDB88320) with the C++ crc_zeropad.c augmentation:
//! extends a finished CRC to cover `zeroPad` additional zero bytes without
//! materialising them.

/// Standard byte-table CRC32 state, internal (pre-complement) form.
const fn make_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB88320
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}
static TABLE: [u32; 256] = make_table();

/// PCLMULQDQ CRC32 - port of crc_clmul.h (Intel/zlib-ng folding method).
/// State is 4 x 16-byte lanes; update() consumes 64-byte blocks, finish()
/// folds a <64 byte tail and produces the same value as the table CRC.
#[cfg(target_arch = "x86_64")]
pub(crate) mod clmul {
    use std::arch::x86_64::*;

    const SHF: [u32; 60] = [
        0x84838281, 0x88878685, 0x8c8b8a89, 0x008f8e8d, 0x85848382, 0x89888786, 0x8d8c8b8a,
        0x01008f8e, 0x86858483, 0x8a898887, 0x8e8d8c8b, 0x0201008f, 0x87868584, 0x8b8a8988,
        0x8f8e8d8c, 0x03020100, 0x88878685, 0x8c8b8a89, 0x008f8e8d, 0x04030201, 0x89888786,
        0x8d8c8b8a, 0x01008f8e, 0x05040302, 0x8a898887, 0x8e8d8c8b, 0x0201008f, 0x06050403,
        0x8b8a8988, 0x8f8e8d8c, 0x03020100, 0x07060504, 0x8c8b8a89, 0x008f8e8d, 0x04030201,
        0x08070605, 0x8d8c8b8a, 0x01008f8e, 0x05040302, 0x09080706, 0x8e8d8c8b, 0x0201008f,
        0x06050403, 0x0a090807, 0x8f8e8d8c, 0x03020100, 0x07060504, 0x0b0a0908, 0x008f8e8d,
        0x04030201, 0x08070605, 0x0c0b0a09, 0x01008f8e, 0x05040302, 0x09080706, 0x0d0c0b0a,
        0x0201008f, 0x06050403, 0x0a090807, 0x0e0d0c0b,
    ];

    pub fn available() -> bool {
        std::arch::is_x86_feature_detected!("pclmulqdq")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
    }

    #[inline]
    unsafe fn fold_merge(src: __m128i, data: __m128i) -> __m128i {
        let f = _mm_set_epi32(
            0x00000001,
            0x54442bd4u32 as i32,
            0x00000001,
            0xc6e41596u32 as i32,
        );
        _mm_xor_si128(
            _mm_xor_si128(_mm_clmulepi64_si128(src, f, 0x01), data),
            _mm_clmulepi64_si128(src, f, 0x10),
        )
    }

    pub unsafe fn init(state: *mut __m128i) {
        *state = _mm_cvtsi32_si128(0x9db42487u32 as i32);
        for i in 1..4 {
            *state.add(i) = _mm_setzero_si128();
        }
    }

    #[target_feature(enable = "pclmulqdq", enable = "ssse3", enable = "sse4.1")]
    pub unsafe fn block(state: *mut __m128i, src: *const u8) {
        for i in 0..4 {
            let d = _mm_loadu_si128(src.add(16 * i) as _);
            *state.add(i) = fold_merge(*state.add(i), d);
        }
    }

    /// crc_finish_clmul: fold the <64 byte tail, then reduce the 4 lanes to
    /// the final CRC32.
    #[target_feature(enable = "pclmulqdq", enable = "ssse3", enable = "sse4.1")]
    pub unsafe fn finish(state: *const __m128i, src: &[u8]) -> u32 {
        let mut crc = [*state, *state.add(1), *state.add(2), *state.add(3)];
        let mut p = src.as_ptr();
        let mut len = src.len();

        if len >= 48 {
            let t0 = _mm_loadu_si128(p as _);
            let t1 = _mm_loadu_si128(p.add(16) as _);
            let t2 = _mm_loadu_si128(p.add(32) as _);
            let t3 = crc[3];
            crc[3] = fold_merge(crc[2], t2);
            crc[2] = fold_merge(crc[1], t1);
            crc[1] = fold_merge(crc[0], t0);
            crc[0] = t3;
        } else if len >= 32 {
            let t0 = _mm_loadu_si128(p as _);
            let t1 = _mm_loadu_si128(p.add(16) as _);
            let t2 = crc[2];
            let t3 = crc[3];
            crc[3] = fold_merge(crc[1], t1);
            crc[2] = fold_merge(crc[0], t0);
            crc[1] = t3;
            crc[0] = t2;
        } else if len >= 16 {
            let t0 = _mm_loadu_si128(p as _);
            let t3 = crc[3];
            crc[3] = fold_merge(crc[0], t0);
            crc[0] = crc[1];
            crc[1] = crc[2];
            crc[2] = t3;
        }
        p = p.add(len & 48);
        len &= 15;

        if len > 0 {
            let shl = _mm_loadu_si128((SHF.as_ptr() as *const __m128i).add(len - 1));
            let shr = _mm_xor_si128(shl, _mm_set1_epi8(-128));
            let mut tmp = [0u8; 16];
            tmp[..len].copy_from_slice(std::slice::from_raw_parts(p, len));
            let t0 = _mm_loadu_si128(tmp.as_ptr() as _);

            let t1 = _mm_shuffle_epi8(crc[0], shl);
            crc[0] = _mm_or_si128(_mm_shuffle_epi8(crc[0], shr), _mm_shuffle_epi8(crc[1], shl));
            crc[1] = _mm_or_si128(_mm_shuffle_epi8(crc[1], shr), _mm_shuffle_epi8(crc[2], shl));
            crc[2] = _mm_or_si128(_mm_shuffle_epi8(crc[2], shr), _mm_shuffle_epi8(crc[3], shl));
            crc[3] = _mm_or_si128(_mm_shuffle_epi8(crc[3], shr), _mm_shuffle_epi8(t0, shl));
            crc[3] = fold_merge(t1, crc[3]);
        }

        let fold = _mm_set_epi32(
            0x00000001,
            0x751997d0u32 as i32,
            0x00000000,
            0xccaa009eu32 as i32,
        );
        let mut t0 = _mm_xor_si128(
            _mm_xor_si128(crc[1], _mm_clmulepi64_si128(crc[0], fold, 0x10)),
            _mm_clmulepi64_si128(crc[0], fold, 0x01),
        );
        t0 = _mm_xor_si128(
            _mm_xor_si128(crc[2], _mm_clmulepi64_si128(t0, fold, 0x10)),
            _mm_clmulepi64_si128(t0, fold, 0x01),
        );
        t0 = _mm_xor_si128(
            _mm_xor_si128(crc[3], _mm_clmulepi64_si128(t0, fold, 0x10)),
            _mm_clmulepi64_si128(t0, fold, 0x01),
        );

        let fold = _mm_set_epi32(
            0x00000001,
            0x63cd6124u32 as i32,
            0x00000000,
            0xccaa009eu32 as i32,
        );
        let mut t1 = _mm_xor_si128(_mm_clmulepi64_si128(t0, fold, 0), _mm_srli_si128(t0, 8));
        t0 = _mm_slli_si128(t1, 4);
        t0 = _mm_clmulepi64_si128(t0, fold, 0x10);
        t1 = _mm_and_si128(t1, _mm_set_epi32(0, -1, -1, 0));
        t0 = _mm_xor_si128(t0, t1);

        let fold = _mm_set_epi32(
            0x00000001,
            0xdb710640u32 as i32,
            0x00000000,
            0xf7011641u32 as i32,
        );
        let mut t1 = _mm_clmulepi64_si128(t0, fold, 0);
        t1 = _mm_clmulepi64_si128(t1, fold, 0x10);
        t0 = _mm_xor_si128(t0, _mm_set_epi32(0, -1, -1, 0));
        t1 = _mm_xor_si128(t1, t0);
        _mm_extract_epi32(t1, 2) as u32
    }
}

#[cfg(target_arch = "x86_64")]
#[repr(align(16))]
#[derive(Clone, Copy)]
pub struct ClmulState([u8; 64]);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32_arm(crc: u32, mut data: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let mut crc = crc;
    while data.len() >= 8 {
        crc = __crc32d(crc, u64::from_le_bytes(data[..8].try_into().unwrap()));
        data = &data[8..];
    }
    if data.len() >= 4 {
        crc = __crc32w(crc, u32::from_le_bytes(data[..4].try_into().unwrap()));
        data = &data[4..];
    }
    if data.len() >= 2 {
        crc = __crc32h(crc, u16::from_le_bytes(data[..2].try_into().unwrap()));
        data = &data[2..];
    }
    if !data.is_empty() {
        crc = __crc32b(crc, data[0]);
    }
    crc
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn arm_crc_available() -> bool {
    std::arch::is_aarch64_feature_detected!("crc")
}

pub enum Crc32 {
    Table(u32),
    /// 64-byte clmul lane state
    #[cfg(target_arch = "x86_64")]
    Clmul(ClmulState),
    /// running CRC word, ARMv8 CRC instructions
    #[cfg(target_arch = "aarch64")]
    Arm(u32),
}

impl Crc32 {
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        if clmul::available() {
            let mut st = ClmulState([0u8; 64]);
            unsafe { clmul::init(st.0.as_mut_ptr() as _) };
            return Crc32::Clmul(st);
        }
        #[cfg(target_arch = "aarch64")]
        if arm_crc_available() {
            return Crc32::Arm(0xFFFF_FFFF);
        }
        Crc32::Table(0xFFFF_FFFF)
    }
    /// Caller feeds full 64-byte blocks only (as in the C++ input hasher).
    pub fn update(&mut self, data: &[u8]) {
        match self {
            Crc32::Table(st) => {
                let mut c = *st;
                for &b in data {
                    c = TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
                }
                *st = c;
            }
            #[cfg(target_arch = "x86_64")]
            Crc32::Clmul(st) => {
                debug_assert!(data.len() % 64 == 0);
                unsafe {
                    for i in 0..data.len() / 64 {
                        clmul::block(st.0.as_mut_ptr() as _, data.as_ptr().add(i * 64));
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            Crc32::Arm(st) => unsafe {
                *st = crc32_arm(*st, data);
            },
        }
    }
    /// crc_finish(state, tail, len): feed the remaining <64 bytes and return
    /// the final-form CRC32 value.
    pub fn finish(&self, tail: &[u8]) -> u32 {
        match self {
            #[cfg(not(target_arch = "aarch64"))]
            Crc32::Table(_) => {
                let mut s = self.clone();
                s.update(tail);
                match s {
                    Crc32::Table(st) => !st,
                    #[cfg(target_arch = "x86_64")]
                    Crc32::Clmul(_) => unreachable!(),
                }
            }
            #[cfg(target_arch = "aarch64")]
            Crc32::Table(_) | Crc32::Arm(_) => {
                let mut s = self.clone();
                s.update(tail);
                match s {
                    Crc32::Table(st) => !st,
                    Crc32::Arm(st) => !st,
                }
            }
            #[cfg(target_arch = "x86_64")]
            Crc32::Clmul(st) => unsafe { clmul::finish(st.0.as_ptr() as _, tail) },
        }
    }
}

impl Clone for Crc32 {
    fn clone(&self) -> Self {
        match self {
            Crc32::Table(st) => Crc32::Table(*st),
            #[cfg(target_arch = "x86_64")]
            Crc32::Clmul(st) => Crc32::Clmul(*st),
            #[cfg(target_arch = "aarch64")]
            Crc32::Arm(st) => Crc32::Arm(*st),
        }
    }
}

// crc_zeropad.c
#[inline]
fn negate(n: u32) -> u32 {
    0u32.wrapping_sub(n)
}

fn crc_multiply(mut a: u32, mut b: u32) -> u32 {
    let mut res = 0u32;
    for _ in 0..31 {
        res ^= negate(b >> 31) & a;
        a = (a >> 1) ^ (0xEDB88320 & negate(a & 1));
        b <<= 1;
    }
    res ^ (negate(b >> 31) & a)
}

/// pre-computed 2^(2^n), first 3 entries removed
static CRC_POWER: [u32; 32] = [
    0x00800000, 0x00008000, 0xedb88320, 0xb1e6b092, 0xa06a2517, 0xed627dae, 0x88d14467, 0xd7bbfe6a,
    0xec447f11, 0x8e7ea170, 0x6427800e, 0x4d47bae0, 0x09fe548f, 0x83852d0f, 0x30362f1a, 0x7b5a9cc3,
    0x31fec169, 0x9fec022a, 0x6c8dedc4, 0x15d6874d, 0x5fde7a4e, 0xbad90e37, 0x2e4e5eef, 0x4eaba214,
    0xa8a472c0, 0x429a969e, 0x148d302a, 0xc40ba6d0, 0xc4e22c3c, 0x40000000, 0x20000000, 0x08000000,
];

/// crc_zeroPad(crc, zeroPad): extend a final-form CRC32 to cover `zero_pad`
/// appended zero bytes.
pub fn crc_zero_pad(crc: u32, mut zero_pad: u64) -> u32 {
    let mut crc = !crc;
    let mut power = 0usize;
    while zero_pad != 0 {
        if zero_pad & 1 != 0 {
            crc = crc_multiply(crc, CRC_POWER[power]);
        }
        zero_pad >>= 1;
        power = (power + 1) & 31;
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_table(blocks: &[u8], tail: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in blocks.iter().chain(tail) {
            c = TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        !c
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn clmul_vs_table() {
        if !clmul::available() {
            return;
        }
        let mut rng = 0x12345u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        for len in [64usize, 128, 192, 64 * 7] {
            let data: Vec<u8> = (0..len).map(|_| next()).collect();
            for tail_len in 0..64usize {
                let tail: Vec<u8> = (0..tail_len).map(|_| next()).collect();
                let mut c = Crc32::new();
                c.update(&data);
                let got = c.finish(&tail);
                let want = run_table(&data, &tail);
                assert_eq!(got, want, "len={} tail={}", len, tail_len);
            }
        }
    }
}

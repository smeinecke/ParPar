//! GF(2^16) arithmetic over polynomial 0x1100B (x^16 + x^12 + x^3 + x + 1),
//! ported from gf16/gfmat_coeff.c and gf16/gf16_lookup.c.

pub const GF16_POLYNOMIAL: u32 = 0x1100B;

struct GfTables {
    /// gf_exp[v>>3] = 2^v for v&7==0; gf_exp[8192+i] = polynomial-mask
    /// correction for the 3 shifted-out bits (see gfmat_coeff.c).
    gf_exp: Vec<u16>,
    /// input_diff[i] = log(base_i) - 2i; combined form in `gfmat_input_log`.
    input_diff: Vec<i8>,
}

static TABLES: std::sync::OnceLock<GfTables> = std::sync::OnceLock::new();

fn tables() -> &'static GfTables {
    TABLES.get_or_init(|| {
        let mut input_diff = vec![0i8; 32768];
        let mut gf_exp = vec![0u16; 8192 + 128];
        let mut exp = 0u32;
        let mut n = 1u32;
        for i in 0..32768usize {
            loop {
                if exp & 7 == 0 {
                    gf_exp[(exp >> 3) as usize] = n as u16;
                }
                exp += 1;
                n <<= 1;
                if n > 65535 {
                    n ^= GF16_POLYNOMIAL;
                }
                if exp % 3 != 0 && exp % 5 != 0 && exp % 17 != 0 && exp % 257 != 0 {
                    break;
                }
            }
            input_diff[i] = (exp as i64 - i as i64 * 2) as i8;
        }
        // correction values for the missing bottom 3 bits of exp
        for i in 0..128usize {
            let mut n = (i as u32) << 9;
            for _ in 0..7 {
                n <<= 1;
                if n > 65535 {
                    n ^= GF16_POLYNOMIAL;
                }
            }
            gf_exp[8192 + i] = n as u16;
        }
        GfTables { gf_exp, input_diff }
    })
}

/// gf16_exp(v): 2^v in GF(2^16), v in 0..65534.
#[inline]
pub fn gf16_exp(v: u16) -> u16 {
    let t = tables();
    let mut result = t.gf_exp[(v >> 3) as usize] as u32;
    result <<= v & 7;
    (result ^ t.gf_exp[8192 + (result >> 16) as usize] as u32) as u16
}

/// gfmat_input_log(inputBlock): discrete log of the coefficient base for
/// input slice `input_block` (inputBlock < 32768).
#[inline]
pub fn gfmat_input_log(input_block: u16) -> u16 {
    let t = tables();
    (input_block as u32)
        .wrapping_mul(2)
        .wrapping_add(t.input_diff[input_block as usize] as i32 as u32) as u16
}

/// (inputLog * recoveryBlock) % 65535 via the MultiPar bit hack.
#[inline]
pub fn gfmat_coeff_log(input_log: u16, recovery_block: u16) -> u16 {
    let mut result = input_log as u32 * recovery_block as u32;
    result = (result >> 16) + (result & 65535);
    result += result >> 16;
    result as u16
}

/// Coefficient for (input_log, recovery exponent): base^exp in GF.
#[inline]
pub fn gfmat_coeff_from_log(input_log: u16, recovery_block: u16) -> u16 {
    gf16_exp(gfmat_coeff_log(input_log, recovery_block))
}

/// gfmat_coeff(inputBlock, recoveryBlock).
#[inline]
pub fn gfmat_coeff(input_block: u16, recovery_block: u16) -> u16 {
    gfmat_coeff_from_log(gfmat_input_log(input_block), recovery_block)
}

/// GF multiply by 2 (the x-shift with polynomial mask).
#[inline(always)]
pub fn gf16_mul2(v: u16) -> u16 {
    let v = v as u32;
    ((v << 1) ^ (0u32.wrapping_sub(v >> 15) & GF16_POLYNOMIAL)) as u16
}

/// Arbitrary GF(2^16) multiply (bit-serial; used only for checksum paths).
pub fn gf16_mul(mut a: u16, mut b: u16) -> u16 {
    let mut r = 0u16;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        a = gf16_mul2(a);
        b >>= 1;
    }
    r
}

/// Build the 512-entry lookup table of gf16_lookup.c's calc_table:
/// `lt[x] = coeff * x`, `lt[256+x] = coeff * (x << 8)` for x in 0..256.
pub fn calc_table(coefficient: u16) -> Box<[u16; 512]> {
    let mut lt = Box::new([0u16; 512]);
    let mut coeff = coefficient;
    let mut j = 1usize;
    while j < 256 {
        for k in 0..j {
            lt[k + j] = coeff ^ lt[k];
        }
        coeff = gf16_mul2(coeff);
        j <<= 1;
    }
    // lt[256+x] = coeff * (x<<8): continues the same doubling sequence
    let mut j = 1usize;
    while j < 256 {
        for k in 0..j {
            lt[256 + k + j] = coeff ^ lt[256 + k];
        }
        coeff = gf16_mul2(coeff);
        j <<= 1;
    }
    lt
}

/// dst[x] ^= src[x] * coeff for each u16 word (len must be even).
/// Equivalent to gf16_lookup_muladd.
pub fn muladd_region(dst: &mut [u8], src: &[u8], len: usize, coefficient: u16) {
    let lt = calc_table(coefficient);
    debug_assert!(len <= dst.len() && len <= src.len());
    let mut p = 0usize;
    // 64-bit fast path like the C impl: four words at a time.
    while p + 8 <= len {
        let s = &src[p..p + 8];
        let v0 = lt[s[0] as usize] ^ lt[256 + s[1] as usize];
        let v1 = lt[s[2] as usize] ^ lt[256 + s[3] as usize];
        let v2 = lt[s[4] as usize] ^ lt[256 + s[5] as usize];
        let v3 = lt[s[6] as usize] ^ lt[256 + s[7] as usize];
        let packed = (v0 as u64) | ((v1 as u64) << 16) | ((v2 as u64) << 32) | ((v3 as u64) << 48);
        let cur = u64::from_le_bytes(dst[p..p + 8].try_into().unwrap());
        dst[p..p + 8].copy_from_slice(&(cur ^ packed).to_le_bytes());
        p += 8;
    }
    while p + 2 <= len {
        let v = lt[src[p] as usize] ^ lt[256 + src[p + 1] as usize];
        dst[p] ^= v as u8;
        dst[p + 1] ^= (v >> 8) as u8;
        p += 2;
    }
}

/// dst = src * coeff (overwrite).
pub fn mul_region(dst: &mut [u8], src: &[u8], len: usize, coefficient: u16) {
    for b in dst[..len].iter_mut() {
        *b = 0;
    }
    muladd_region(dst, src, len, coefficient);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_basics() {
        // exp/log sanity: 2^0 = 1, 2^1 = 2
        assert_eq!(gf16_exp(0), 1);
        assert_eq!(gf16_exp(1), 2);
        assert_eq!(gf16_exp(4), 16);
        // input 0 -> log 1 -> coeff(rec 0) = g^0 = 1
        assert_eq!(gfmat_input_log(0), 1);
        assert_eq!(gfmat_coeff(0, 0), 1);
        assert_eq!(gfmat_coeff(0, 1), 2);
        // mul2 polynomial reduction
        assert_eq!(gf16_mul2(0x8000), 0x100b);
        assert_eq!(gf16_mul2(0x8001), 0x1009);
        assert_eq!(gf16_mul(0x1234, 1), 0x1234);
        assert_eq!(gf16_mul(0x1234, 0), 0);
    }

    #[test]
    fn lookup_table() {
        let lt = calc_table(0x1234);
        assert_eq!(lt[0], 0);
        assert_eq!(lt[1], 0x1234);
        assert_eq!(lt[2], gf16_mul2(0x1234));
        assert_eq!(lt[3], 0x1234 ^ gf16_mul2(0x1234));
        assert_eq!(lt[256 + 1], gf16_mul(0x1234, 0x100));
        for x in 0..256usize {
            assert_eq!(lt[x], gf16_mul(0x1234, x as u16), "lo {}", x);
            assert_eq!(lt[256 + x], gf16_mul(0x1234, (x as u16) << 8), "hi {}", x);
        }
    }

    #[test]
    fn muladd() {
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut dst = [0u8; 10];
        muladd_region(&mut dst, &src, 10, 1);
        assert_eq!(dst, src);
        // coeff 2: each u16 word doubled in GF
        let mut dst2 = [0u8; 10];
        muladd_region(&mut dst2, &src, 10, 2);
        for i in 0..5 {
            let w = u16::from_le_bytes([src[2 * i], src[2 * i + 1]]);
            assert_eq!(
                u16::from_le_bytes([dst2[2 * i], dst2[2 * i + 1]]),
                gf16_mul2(w)
            );
        }
    }
}

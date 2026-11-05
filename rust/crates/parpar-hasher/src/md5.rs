//! Scalar MD5 primitives ported from hasher/md5-scalar-base.h, md5-base.h and
//! md5-final.c. Lane-0 (block) and lane-1 (file) states live side by side in
//! one `Md5X2` as in the C++ md5x2-scalar implementation.

pub const MD5_BLOCKSIZE: usize = 64;
const IV: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

#[inline(always)]
fn f(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | (!x & z)
}
#[inline(always)]
fn g(x: u32, y: u32, z: u32) -> u32 {
    (x & z) | (y & !z)
}
#[inline(always)]
fn h(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}
#[inline(always)]
fn i(x: u32, y: u32, z: u32) -> u32 {
    y ^ (x | !z)
}

/// Single 64-byte MD5 compression block over `state` (4 words, LE I/O).
#[inline(always)]
pub fn md5_process_block(state: &mut [u32; 4], block: &[u8]) {
    debug_assert_eq!(block.len(), MD5_BLOCKSIZE);
    let mut x = [0u32; 16];
    for (j, w) in x.iter_mut().enumerate() {
        *w = u32::from_le_bytes(block[j * 4..j * 4 + 4].try_into().unwrap());
    }
    let (mut a, mut b, mut c, mut d) = (state[0], state[1], state[2], state[3]);
    let (oa, ob, oc, od) = (a, b, c, d);
    macro_rules! rx {
        ($f:ident, $a:ident, $b:ident, $c:ident, $d:ident, $k:expr, $r:expr, $t:expr) => {{
            $a = $a
                .wrapping_add($t)
                .wrapping_add($f($b, $c, $d))
                .wrapping_add(x[$k]);
            $a = $a.rotate_left($r).wrapping_add($b);
        }};
    }
    // Round 0 (F)
    rx!(f, a, b, c, d, 0, 7, 0xd76aa478);
    rx!(f, d, a, b, c, 1, 12, 0xe8c7b756);
    rx!(f, c, d, a, b, 2, 17, 0x242070db);
    rx!(f, b, c, d, a, 3, 22, 0xc1bdceee);
    rx!(f, a, b, c, d, 4, 7, 0xf57c0faf);
    rx!(f, d, a, b, c, 5, 12, 0x4787c62a);
    rx!(f, c, d, a, b, 6, 17, 0xa8304613);
    rx!(f, b, c, d, a, 7, 22, 0xfd469501);
    rx!(f, a, b, c, d, 8, 7, 0x698098d8);
    rx!(f, d, a, b, c, 9, 12, 0x8b44f7af);
    rx!(f, c, d, a, b, 10, 17, 0xffff5bb1);
    rx!(f, b, c, d, a, 11, 22, 0x895cd7be);
    rx!(f, a, b, c, d, 12, 7, 0x6b901122);
    rx!(f, d, a, b, c, 13, 12, 0xfd987193);
    rx!(f, c, d, a, b, 14, 17, 0xa679438e);
    rx!(f, b, c, d, a, 15, 22, 0x49b40821);
    // Round 1 (G)
    rx!(g, a, b, c, d, 1, 5, 0xf61e2562);
    rx!(g, d, a, b, c, 6, 9, 0xc040b340);
    rx!(g, c, d, a, b, 11, 14, 0x265e5a51);
    rx!(g, b, c, d, a, 0, 20, 0xe9b6c7aa);
    rx!(g, a, b, c, d, 5, 5, 0xd62f105d);
    rx!(g, d, a, b, c, 10, 9, 0x02441453);
    rx!(g, c, d, a, b, 15, 14, 0xd8a1e681);
    rx!(g, b, c, d, a, 4, 20, 0xe7d3fbc8);
    rx!(g, a, b, c, d, 9, 5, 0x21e1cde6);
    rx!(g, d, a, b, c, 14, 9, 0xc33707d6);
    rx!(g, c, d, a, b, 3, 14, 0xf4d50d87);
    rx!(g, b, c, d, a, 8, 20, 0x455a14ed);
    rx!(g, a, b, c, d, 13, 5, 0xa9e3e905);
    rx!(g, d, a, b, c, 2, 9, 0xfcefa3f8);
    rx!(g, c, d, a, b, 7, 14, 0x676f02d9);
    rx!(g, b, c, d, a, 12, 20, 0x8d2a4c8a);
    // Round 2 (H)
    rx!(h, a, b, c, d, 5, 4, 0xfffa3942);
    rx!(h, d, a, b, c, 8, 11, 0x8771f681);
    rx!(h, c, d, a, b, 11, 16, 0x6d9d6122);
    rx!(h, b, c, d, a, 14, 23, 0xfde5380c);
    rx!(h, a, b, c, d, 1, 4, 0xa4beea44);
    rx!(h, d, a, b, c, 4, 11, 0x4bdecfa9);
    rx!(h, c, d, a, b, 7, 16, 0xf6bb4b60);
    rx!(h, b, c, d, a, 10, 23, 0xbebfbc70);
    rx!(h, a, b, c, d, 13, 4, 0x289b7ec6);
    rx!(h, d, a, b, c, 0, 11, 0xeaa127fa);
    rx!(h, c, d, a, b, 3, 16, 0xd4ef3085);
    rx!(h, b, c, d, a, 6, 23, 0x04881d05);
    rx!(h, a, b, c, d, 9, 4, 0xd9d4d039);
    rx!(h, d, a, b, c, 12, 11, 0xe6db99e5);
    rx!(h, c, d, a, b, 15, 16, 0x1fa27cf8);
    rx!(h, b, c, d, a, 2, 23, 0xc4ac5665);
    // Round 3 (I)
    rx!(i, a, b, c, d, 0, 6, 0xf4292244);
    rx!(i, d, a, b, c, 7, 10, 0x432aff97);
    rx!(i, c, d, a, b, 14, 15, 0xab9423a7);
    rx!(i, b, c, d, a, 5, 21, 0xfc93a039);
    rx!(i, a, b, c, d, 12, 6, 0x655b59c3);
    rx!(i, d, a, b, c, 3, 10, 0x8f0ccc92);
    rx!(i, c, d, a, b, 10, 15, 0xffeff47d);
    rx!(i, b, c, d, a, 1, 21, 0x85845dd1);
    rx!(i, a, b, c, d, 8, 6, 0x6fa87e4f);
    rx!(i, d, a, b, c, 15, 10, 0xfe2ce6e0);
    rx!(i, c, d, a, b, 6, 15, 0xa3014314);
    rx!(i, b, c, d, a, 13, 21, 0x4e0811a1);
    rx!(i, a, b, c, d, 4, 6, 0xf7537e82);
    rx!(i, d, a, b, c, 11, 10, 0xbd3af235);
    rx!(i, c, d, a, b, 2, 15, 0x2ad7d2bb);
    rx!(i, b, c, d, a, 9, 21, 0xeb86d391);
    state[0] = oa.wrapping_add(a);
    state[1] = ob.wrapping_add(b);
    state[2] = oc.wrapping_add(c);
    state[3] = od.wrapping_add(d);
}

/// Round constants as 32 u64 pairs: consecutive steps share one 64-bit
/// value, letting codegen emit a single `ldp`/`lsr` pair (the C++ arm64
/// asm trick) instead of mov+movk per constant.
static MD5_KP: [u64; 32] = [
    0xe8c7b756_d76aa478, 0xc1bdceee_242070db, 0x4787c62a_f57c0faf, 0xfd469501_a8304613,
    0x8b44f7af_698098d8, 0x895cd7be_ffff5bb1, 0xfd987193_6b901122, 0x49b40821_a679438e,
    0xc040b340_f61e2562, 0xe9b6c7aa_265e5a51, 0x02441453_d62f105d, 0xe7d3fbc8_d8a1e681,
    0xc33707d6_21e1cde6, 0x455a14ed_f4d50d87, 0xfcefa3f8_a9e3e905, 0x8d2a4c8a_676f02d9,
    0x8771f681_fffa3942, 0xfde5380c_6d9d6122, 0x4bdecfa9_a4beea44, 0xbebfbc70_f6bb4b60,
    0xeaa127fa_289b7ec6, 0x04881d05_d4ef3085, 0xe6db99e5_d9d4d039, 0xc4ac5665_1fa27cf8,
    0x432aff97_f4292244, 0xfc93a039_ab9423a7, 0x8f0ccc92_655b59c3, 0x85845dd1_ffeff47d,
    0xfe2ce6e0_6fa87e4f, 0x4e0811a1_a3014314, 0xbd3af235_f7537e82, 0xeb86d391_2ad7d2bb,
];

/// Two blocks for two independent states, stepped in lockstep so the
/// compiler can interleave the chains - each chain's serial rotate/add
/// latency gets covered by the other's ops (the trick the C++ arm64 asm
/// uses). Roughly 2x the throughput of two sequential blocks on wide
/// scalar cores.
pub fn md5_process_block_x2(
    state0: &mut [u32; 4],
    block0: &[u8],
    state1: &mut [u32; 4],
    block1: &[u8],
) {
    debug_assert_eq!(block0.len(), MD5_BLOCKSIZE);
    debug_assert_eq!(block1.len(), MD5_BLOCKSIZE);
    let mut x = [0u32; 16];
    let mut y = [0u32; 16];
    for (j, (w, v)) in x.iter_mut().zip(y.iter_mut()).enumerate() {
        *w = u32::from_le_bytes(block0[j * 4..j * 4 + 4].try_into().unwrap());
        *v = u32::from_le_bytes(block1[j * 4..j * 4 + 4].try_into().unwrap());
    }
    let (mut a, mut b, mut c, mut d) = (state0[0], state0[1], state0[2], state0[3]);
    let (mut p, mut q, mut r, mut s) = (state1[0], state1[1], state1[2], state1[3]);
    let (oa, ob, oc, od) = (a, b, c, d);
    let (op, oq, or_, os) = (p, q, r, s);
    // rp! = one pair of steps (both chains each): the constant pair is a
    // u64 so codegen can use one load + lsr for both steps
    macro_rules! rp {
        ($f:ident, $a:ident, $b:ident, $c:ident, $d:ident,
         $p:ident, $q:ident, $r2:ident, $s:ident,
         $k0:expr, $k1:expr, $rot0:expr, $rot1:expr, $kp:expr) => {{
            let kp: u64 = $kp;
            $a = $a
                .wrapping_add(kp as u32)
                .wrapping_add($f($b, $c, $d))
                .wrapping_add(x[$k0]);
            $a = $a.rotate_left($rot0).wrapping_add($b);
            $p = $p
                .wrapping_add(kp as u32)
                .wrapping_add($f($q, $r2, $s))
                .wrapping_add(y[$k0]);
            $p = $p.rotate_left($rot0).wrapping_add($q);
            $d = $d
                .wrapping_add((kp >> 32) as u32)
                .wrapping_add($f($a, $b, $c))
                .wrapping_add(x[$k1]);
            $d = $d.rotate_left($rot1).wrapping_add($a);
            $s = $s
                .wrapping_add((kp >> 32) as u32)
                .wrapping_add($f($p, $q, $r2))
                .wrapping_add(y[$k1]);
            $s = $s.rotate_left($rot1).wrapping_add($p);
        }};
    }
    // Round 0 (F)
    rp!(f, a, b, c, d, p, q, r, s, 0, 1, 7, 12, MD5_KP[0]);
    rp!(f, c, d, a, b, r, s, p, q, 2, 3, 17, 22, MD5_KP[1]);
    rp!(f, a, b, c, d, p, q, r, s, 4, 5, 7, 12, MD5_KP[2]);
    rp!(f, c, d, a, b, r, s, p, q, 6, 7, 17, 22, MD5_KP[3]);
    rp!(f, a, b, c, d, p, q, r, s, 8, 9, 7, 12, MD5_KP[4]);
    rp!(f, c, d, a, b, r, s, p, q, 10, 11, 17, 22, MD5_KP[5]);
    rp!(f, a, b, c, d, p, q, r, s, 12, 13, 7, 12, MD5_KP[6]);
    rp!(f, c, d, a, b, r, s, p, q, 14, 15, 17, 22, MD5_KP[7]);
    // Round 1 (G)
    rp!(g, a, b, c, d, p, q, r, s, 1, 6, 5, 9, MD5_KP[8]);
    rp!(g, c, d, a, b, r, s, p, q, 11, 0, 14, 20, MD5_KP[9]);
    rp!(g, a, b, c, d, p, q, r, s, 5, 10, 5, 9, MD5_KP[10]);
    rp!(g, c, d, a, b, r, s, p, q, 15, 4, 14, 20, MD5_KP[11]);
    rp!(g, a, b, c, d, p, q, r, s, 9, 14, 5, 9, MD5_KP[12]);
    rp!(g, c, d, a, b, r, s, p, q, 3, 8, 14, 20, MD5_KP[13]);
    rp!(g, a, b, c, d, p, q, r, s, 13, 2, 5, 9, MD5_KP[14]);
    rp!(g, c, d, a, b, r, s, p, q, 7, 12, 14, 20, MD5_KP[15]);
    // Round 2 (H)
    rp!(h, a, b, c, d, p, q, r, s, 5, 8, 4, 11, MD5_KP[16]);
    rp!(h, c, d, a, b, r, s, p, q, 11, 14, 16, 23, MD5_KP[17]);
    rp!(h, a, b, c, d, p, q, r, s, 1, 4, 4, 11, MD5_KP[18]);
    rp!(h, c, d, a, b, r, s, p, q, 7, 10, 16, 23, MD5_KP[19]);
    rp!(h, a, b, c, d, p, q, r, s, 13, 0, 4, 11, MD5_KP[20]);
    rp!(h, c, d, a, b, r, s, p, q, 3, 6, 16, 23, MD5_KP[21]);
    rp!(h, a, b, c, d, p, q, r, s, 9, 12, 4, 11, MD5_KP[22]);
    rp!(h, c, d, a, b, r, s, p, q, 15, 2, 16, 23, MD5_KP[23]);
    // Round 3 (I)
    rp!(i, a, b, c, d, p, q, r, s, 0, 7, 6, 10, MD5_KP[24]);
    rp!(i, c, d, a, b, r, s, p, q, 14, 5, 15, 21, MD5_KP[25]);
    rp!(i, a, b, c, d, p, q, r, s, 12, 3, 6, 10, MD5_KP[26]);
    rp!(i, c, d, a, b, r, s, p, q, 10, 1, 15, 21, MD5_KP[27]);
    rp!(i, a, b, c, d, p, q, r, s, 8, 15, 6, 10, MD5_KP[28]);
    rp!(i, c, d, a, b, r, s, p, q, 6, 13, 15, 21, MD5_KP[29]);
    rp!(i, a, b, c, d, p, q, r, s, 4, 11, 6, 10, MD5_KP[30]);
    rp!(i, c, d, a, b, r, s, p, q, 2, 9, 15, 21, MD5_KP[31]);
    state0[0] = oa.wrapping_add(a);
    state0[1] = ob.wrapping_add(b);
    state0[2] = oc.wrapping_add(c);
    state0[3] = od.wrapping_add(d);
    state1[0] = op.wrapping_add(p);
    state1[1] = oq.wrapping_add(q);
    state1[2] = or_.wrapping_add(r);
    state1[3] = os.wrapping_add(s);
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_rf4 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             mov {t2:e}, {d2:e}
             add {a1:e}, {K0}
             add {a2:e}, {K0}
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             and {t1:e}, {b1:e}
             and {t2:e}, {b2:e}
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             mov {t1:e}, {c1:e}
             mov {t2:e}, {c2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             and {t1:e}, {a1:e}
             and {t2:e}, {a2:e}
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             mov {t1:e}, {b1:e}
             mov {t2:e}, {b2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             and {t1:e}, {d1:e}
             and {t2:e}, {d2:e}
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             mov {t1:e}, {a1:e}
             mov {t2:e}, {a2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             and {t1:e}, {c1:e}
             and {t2:e}, {c2:e}
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {a2:e}, DWORD PTR [{q}+{OFF3}]
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_rg4 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             mov {t2:e}, {d2:e}
             add {a1:e}, {K0}
             add {a2:e}, {K0}
             not {t1:e}
             not {t2:e}
             and {t1:e}, {c1:e}
             and {t2:e}, {c2:e}
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             mov {t1:e}, {d1:e}
             mov {t2:e}, {d2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             and {t1:e}, {b1:e}
             and {t2:e}, {b2:e}
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             mov {t1:e}, {c1:e}
             mov {t2:e}, {c2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             not {t1:e}
             not {t2:e}
             and {t1:e}, {b1:e}
             and {t2:e}, {b2:e}
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             mov {t1:e}, {c1:e}
             mov {t2:e}, {c2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             and {t1:e}, {a1:e}
             and {t2:e}, {a2:e}
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             mov {t1:e}, {b1:e}
             mov {t2:e}, {b2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             not {t1:e}
             not {t2:e}
             and {t1:e}, {a1:e}
             and {t2:e}, {a2:e}
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             mov {t1:e}, {b1:e}
             mov {t2:e}, {b2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             and {t1:e}, {d1:e}
             and {t2:e}, {d2:e}
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             mov {t1:e}, {a1:e}
             mov {t2:e}, {a2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             not {t1:e}
             not {t2:e}
             and {t1:e}, {d1:e}
             and {t2:e}, {d2:e}
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             mov {t1:e}, {a1:e}
             mov {t2:e}, {a2:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {a2:e}, DWORD PTR [{q}+{OFF3}]
             and {t1:e}, {c1:e}
             and {t2:e}, {c2:e}
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_rh4 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             mov {t2:e}, {d2:e}
             add {a1:e}, {K0}
             add {a2:e}, {K0}
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             mov {t1:e}, {c1:e}
             mov {t2:e}, {c2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             mov {t1:e}, {b1:e}
             mov {t2:e}, {b2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             mov {t1:e}, {a1:e}
             mov {t2:e}, {a2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {a2:e}, DWORD PTR [{q}+{OFF3}]
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             mov {t2:e}, {d2:e}
             add {a1:e}, {K0}
             add {a2:e}, {K0}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {b1:e}
             or {t2:e}, {b2:e}
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             mov {t1:e}, {c1:e}
             mov {t2:e}, {c2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {a1:e}
             or {t2:e}, {a2:e}
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             mov {t1:e}, {b1:e}
             mov {t2:e}, {b2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {d1:e}
             or {t2:e}, {d2:e}
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             mov {t1:e}, {a1:e}
             mov {t2:e}, {a2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {c1:e}
             or {t2:e}, {c2:e}
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {a2:e}, DWORD PTR [{q}+{OFF3}]
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4_last {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    // last round has no message-word prefetch
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             mov {t2:e}, {d2:e}
             add {a1:e}, {K0}
             add {a2:e}, {K0}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {b1:e}
             or {t2:e}, {b2:e}
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             mov {t1:e}, {c1:e}
             mov {t2:e}, {c2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {a1:e}
             or {t2:e}, {a2:e}
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             mov {t1:e}, {b1:e}
             mov {t2:e}, {b2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {d1:e}
             or {t2:e}, {d2:e}
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             mov {t1:e}, {a1:e}
             mov {t2:e}, {a2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             not {t1:e}
             not {t2:e}
             or {t1:e}, {c1:e}
             or {t2:e}, {c2:e}
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}
#[cfg(target_arch = "x86_64")]
macro_rules! md5_rg4_bmi1 {
    // BMI1 variant: andn replaces mov+not+and in the first G term
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             add {a1:e}, {K0}
             add {a2:e}, {K0}
             andn {t1:e}, {d1:e}, {c1:e}
             andn {t2:e}, {d2:e}, {c2:e}
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             mov {t1:e}, {d1:e}
             mov {t2:e}, {d2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             and {t1:e}, {b1:e}
             and {t2:e}, {b2:e}
             add {a1:e}, {t1:e}
             add {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             andn {t1:e}, {c1:e}, {b1:e}
             andn {t2:e}, {c2:e}, {b2:e}
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             mov {t1:e}, {c1:e}
             mov {t2:e}, {c2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             and {t1:e}, {a1:e}
             and {t2:e}, {a2:e}
             add {d1:e}, {t1:e}
             add {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             andn {t1:e}, {b1:e}, {a1:e}
             andn {t2:e}, {b2:e}, {a2:e}
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             mov {t1:e}, {b1:e}
             mov {t2:e}, {b2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             and {t1:e}, {d1:e}
             and {t2:e}, {d2:e}
             add {c1:e}, {t1:e}
             add {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             andn {t1:e}, {a1:e}, {d1:e}
             andn {t2:e}, {a2:e}, {d2:e}
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             mov {t1:e}, {a1:e}
             mov {t2:e}, {a2:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {a2:e}, DWORD PTR [{q}+{OFF3}]
             and {t1:e}, {c1:e}
             and {t2:e}, {c2:e}
             add {b1:e}, {t1:e}
             add {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4_bmi1 {
    // BMI1 variant: I(x,y,z)=y^(x|~z) evaluated as -( z^(~x&y) )-1 via andn+sub
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             add {a1:e}, {K0}
             add {a2:e}, {K0}
             andn {t1:e}, {b1:e}, {d1:e}
             andn {t2:e}, {b2:e}, {d2:e}
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             sub {a1:e}, {t1:e}
             sub {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             andn {t1:e}, {a1:e}, {c1:e}
             andn {t2:e}, {a2:e}, {c2:e}
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             sub {d1:e}, {t1:e}
             sub {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             andn {t1:e}, {d1:e}, {b1:e}
             andn {t2:e}, {d2:e}, {b2:e}
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             sub {c1:e}, {t1:e}
             sub {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             andn {t1:e}, {c1:e}, {a1:e}
             andn {t2:e}, {c2:e}, {a2:e}
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {a2:e}, DWORD PTR [{q}+{OFF3}]
             sub {b1:e}, {t1:e}
             sub {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const ($k0 - 1), K1 = const ($k1 - 1), K2 = const ($k2 - 1), K3 = const ($k3 - 1),
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4_last_bmi1 {
    // BMI1 variant of md5_ri4_last - final round has no message prefetch
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident,
     $a2:ident, $b2:ident, $c2:ident, $d2:ident,
     $p:ident, $q:ident,
     $i0:expr, $i1:expr, $i2:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             add {a1:e}, {K0}
             add {a2:e}, {K0}
             andn {t1:e}, {b1:e}, {d1:e}
             andn {t2:e}, {b2:e}, {d2:e}
             xor {t1:e}, {c1:e}
             xor {t2:e}, {c2:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {d2:e}, DWORD PTR [{q}+{OFF0}]
             sub {a1:e}, {t1:e}
             sub {a2:e}, {t2:e}
             rol {a1:e}, {R0}
             rol {a2:e}, {R0}
             add {a1:e}, {b1:e}
             add {a2:e}, {b2:e}
             add {d1:e}, {K1}
             add {d2:e}, {K1}
             andn {t1:e}, {a1:e}, {c1:e}
             andn {t2:e}, {a2:e}, {c2:e}
             xor {t1:e}, {b1:e}
             xor {t2:e}, {b2:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {c2:e}, DWORD PTR [{q}+{OFF1}]
             sub {d1:e}, {t1:e}
             sub {d2:e}, {t2:e}
             rol {d1:e}, {R1}
             rol {d2:e}, {R1}
             add {d1:e}, {a1:e}
             add {d2:e}, {a2:e}
             add {c1:e}, {K2}
             add {c2:e}, {K2}
             andn {t1:e}, {d1:e}, {b1:e}
             andn {t2:e}, {d2:e}, {b2:e}
             xor {t1:e}, {a1:e}
             xor {t2:e}, {a2:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {b2:e}, DWORD PTR [{q}+{OFF2}]
             sub {c1:e}, {t1:e}
             sub {c2:e}, {t2:e}
             rol {c1:e}, {R2}
             rol {c2:e}, {R2}
             add {c1:e}, {d1:e}
             add {c2:e}, {d2:e}
             add {b1:e}, {K3}
             add {b2:e}, {K3}
             andn {t1:e}, {c1:e}, {a1:e}
             andn {t2:e}, {c2:e}, {a2:e}
             xor {t1:e}, {d1:e}
             xor {t2:e}, {d2:e}
             sub {b1:e}, {t1:e}
             sub {b2:e}, {t2:e}
             rol {b1:e}, {R3}
             rol {b2:e}, {R3}
             add {b1:e}, {c1:e}
             add {b2:e}, {c2:e}",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            a2 = inout(reg) $a2, b2 = inout(reg) $b2,
            c2 = inout(reg) $c2, d2 = inout(reg) $d2,
            t1 = out(reg) _, t2 = out(reg) _,
p = in(reg) $p, q = in(reg) $q,
             OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4),
             K0 = const ($k0 - 1), K1 = const ($k1 - 1), K2 = const ($k2 - 1), K3 = const ($k3 - 1),
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

/// x86 two-lane MD5 block - direct port of md5x2-x86-asm.h
/// (md5_process_block_x2_scalar with MD5_USE_ASM, classic ISA + BMI1 variants).
#[cfg(target_arch = "x86_64")]
macro_rules! md5_block_x2_impl {
    ($fname:ident, $rg4:ident, $ri4:ident, $ri4l:ident) => {
        unsafe fn $fname(state: *mut u32, src0: *const u8, src1: *const u8) {
            let p = src0;
            let q = src1;
            let mut a1 = state.add(0).read();
            let mut b1 = state.add(1).read();
            let mut c1 = state.add(2).read();
            let mut d1 = state.add(3).read();
            let mut a2 = state.add(4).read();
            let mut b2 = state.add(5).read();
            let mut c2 = state.add(6).read();
            let mut d2 = state.add(7).read();
            // first message words folded in ahead of round 0 (like `A1 += read32()`)
            a1 = a1.wrapping_add((p as *const u32).read_unaligned());
            a2 = a2.wrapping_add((q as *const u32).read_unaligned());

            // RF4/RG4/RH4/RI4 groups - rotation is baked into each template
            md5_rf4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                1,
                2,
                3,
                4,
                0xd76aa478u32,
                0xe8c7b756u32,
                0x242070dbu32,
                0xc1bdceeeu32,
                7,
                12,
                17,
                22
            );
            md5_rf4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                5,
                6,
                7,
                8,
                0xf57c0fafu32,
                0x4787c62au32,
                0xa8304613u32,
                0xfd469501u32,
                7,
                12,
                17,
                22
            );
            md5_rf4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                9,
                10,
                11,
                12,
                0x698098d8u32,
                0x8b44f7afu32,
                0xffff5bb1u32,
                0x895cd7beu32,
                7,
                12,
                17,
                22
            );
            md5_rf4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                13,
                14,
                15,
                1,
                0x6b901122u32,
                0xfd987193u32,
                0xa679438eu32,
                0x49b40821u32,
                7,
                12,
                17,
                22
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                6,
                11,
                0,
                5,
                0xf61e2562u32,
                0xc040b340u32,
                0x265e5a51u32,
                0xe9b6c7aau32,
                5,
                9,
                14,
                20
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                10,
                15,
                4,
                9,
                0xd62f105du32,
                0x02441453u32,
                0xd8a1e681u32,
                0xe7d3fbc8u32,
                5,
                9,
                14,
                20
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                14,
                3,
                8,
                13,
                0x21e1cde6u32,
                0xc33707d6u32,
                0xf4d50d87u32,
                0x455a14edu32,
                5,
                9,
                14,
                20
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                2,
                7,
                12,
                5,
                0xa9e3e905u32,
                0xfcefa3f8u32,
                0x676f02d9u32,
                0x8d2a4c8au32,
                5,
                9,
                14,
                20
            );
            md5_rh4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                8,
                11,
                14,
                1,
                0xfffa3942u32,
                0x8771f681u32,
                0x6d9d6122u32,
                0xfde5380cu32,
                4,
                11,
                16,
                23
            );
            md5_rh4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                4,
                7,
                10,
                13,
                0xa4beea44u32,
                0x4bdecfa9u32,
                0xf6bb4b60u32,
                0xbebfbc70u32,
                4,
                11,
                16,
                23
            );
            md5_rh4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                0,
                3,
                6,
                9,
                0x289b7ec6u32,
                0xeaa127fau32,
                0xd4ef3085u32,
                0x04881d05u32,
                4,
                11,
                16,
                23
            );
            md5_rh4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                12,
                15,
                2,
                0,
                0xd9d4d039u32,
                0xe6db99e5u32,
                0x1fa27cf8u32,
                0xc4ac5665u32,
                4,
                11,
                16,
                23
            );
            $ri4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                7,
                14,
                5,
                12,
                0xf4292244u32,
                0x432aff97u32,
                0xab9423a7u32,
                0xfc93a039u32,
                6,
                10,
                15,
                21
            );
            $ri4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                3,
                10,
                1,
                8,
                0x655b59c3u32,
                0x8f0ccc92u32,
                0xffeff47du32,
                0x85845dd1u32,
                6,
                10,
                15,
                21
            );
            $ri4!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                15,
                6,
                13,
                4,
                0x6fa87e4fu32,
                0xfe2ce6e0u32,
                0xa3014314u32,
                0x4e0811a1u32,
                6,
                10,
                15,
                21
            );
            $ri4l!(
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                p,
                q,
                11,
                2,
                9,
                0xf7537e82u32,
                0xbd3af235u32,
                0x2ad7d2bbu32,
                0xeb86d391u32,
                6,
                10,
                15,
                21
            );

            state.add(0).write(state.add(0).read().wrapping_add(a1));
            state.add(1).write(state.add(1).read().wrapping_add(b1));
            state.add(2).write(state.add(2).read().wrapping_add(c1));
            state.add(3).write(state.add(3).read().wrapping_add(d1));
            state.add(4).write(state.add(4).read().wrapping_add(a2));
            state.add(5).write(state.add(5).read().wrapping_add(b2));
            state.add(6).write(state.add(6).read().wrapping_add(c2));
            state.add(7).write(state.add(7).read().wrapping_add(d2));
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_use_bmi1() -> bool {
    std::arch::is_x86_feature_detected!("bmi1")
}

#[cfg(target_arch = "x86_64")]
md5_block_x2_impl!(md5_block_x2_asm, md5_rg4, md5_ri4, md5_ri4_last);
#[cfg(target_arch = "x86_64")]
md5_block_x2_impl!(
    md5_block_x2_asm_bmi1,
    md5_rg4_bmi1,
    md5_ri4_bmi1,
    md5_ri4_last_bmi1
);

/// Two-lane MD5 state: lane 0 = per-block hash, lane 1 = whole-file hash
/// (md5x2-scalar equivalent; lane 0 occupies state[0..4], lane 1 state[4..8]).
#[derive(Clone)]
pub struct Md5X2 {
    pub state: [u32; 8],
}

impl Md5X2 {
    pub fn new() -> Self {
        Md5X2 {
            state: [IV[0], IV[1], IV[2], IV[3], IV[0], IV[1], IV[2], IV[3]],
        }
    }
    /// md5_init_lane_x2_scalar(state, idx)
    pub fn init_lane(&mut self, lane: usize) {
        self.state[lane * 4..lane * 4 + 4].copy_from_slice(&IV);
    }
    /// md5_update_block_x2(state, block_data /*lane0*/, file_data /*lane1*/)
    pub fn update_block(&mut self, block_data: &[u8], file_data: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // same kernel the C++ build uses: two-lane scalar in inline asm
            if x86_use_bmi1() {
                return md5_block_x2_asm_bmi1(
                    self.state.as_mut_ptr(),
                    block_data.as_ptr(),
                    file_data.as_ptr(),
                );
            }
            return md5_block_x2_asm(
                self.state.as_mut_ptr(),
                block_data.as_ptr(),
                file_data.as_ptr(),
            );
        }
        #[allow(unreachable_code)]
        {
            // scalar on aarch64: the C++ port reaches the same conclusion
            // (dedicated arm64 GPR asm, no NEON) - a serial MD5 chain gains
            // nothing from 2-lane SIMD and pays for it in vector-op latency.
            // Interleaved lockstep version recovers most of the asm's ILP.
            let (l0, l1) = self.state.split_at_mut(4);
            let l0: &mut [u32; 4] = l0.try_into().unwrap();
            let l1: &mut [u32; 4] = l1.try_into().unwrap();
            md5_process_block_x2(l0, &block_data[..MD5_BLOCKSIZE], l1, &file_data[..MD5_BLOCKSIZE]);
        }
    }
    /// md5_extract_x2(dst, state, idx)
    pub fn extract(&self, lane: usize) -> [u32; 4] {
        self.state[lane * 4..lane * 4 + 4].try_into().unwrap()
    }
    /// Feed only lane 0 (block hash). Used when the file lane is computed
    /// on a separate thread (the read side) and this context is lane0-only.
    #[inline]
    pub fn update_lane0(&mut self, data: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if x86_use_bmi1() {
                return md5_block_x1_asm_bmi1(self.state.as_mut_ptr(), data.as_ptr());
            }
            return md5_block_x1_asm(self.state.as_mut_ptr(), data.as_ptr());
        }
        #[allow(unreachable_code)]
        {
            let l0: &mut [u32; 4] = (&mut self.state[..4]).try_into().unwrap();
            md5_process_block(l0, &data[..MD5_BLOCKSIZE]);
        }
    }
}

/// Single-lane block step shared by Md5Ctx::update - same dispatch as the
/// x2 kernels (BMI1 variant when the CPU supports it).
#[inline]
fn md5_block_x1(state: &mut [u32; 4], data: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if x86_use_bmi1() {
            return md5_block_x1_asm_bmi1(state.as_mut_ptr(), data.as_ptr());
        }
        return md5_block_x1_asm(state.as_mut_ptr(), data.as_ptr());
    }
    #[allow(unreachable_code)]
    md5_process_block(state, &data[..MD5_BLOCKSIZE]);
}

/// Streaming single-lane MD5 (for HasherOutput / MD5Multi regions).
pub struct Md5Ctx {
    pub state: [u32; 4],
    pub tmp: [u8; MD5_BLOCKSIZE],
    pub tmp_len: usize,
    pub data_len: u64,
}

impl Md5Ctx {
    pub fn new() -> Self {
        Md5Ctx {
            state: IV,
            tmp: [0; MD5_BLOCKSIZE],
            tmp_len: 0,
            data_len: 0,
        }
    }
    pub fn update(&mut self, mut data: &[u8]) {
        self.data_len = self.data_len.wrapping_add(data.len() as u64);
        if self.tmp_len > 0 {
            let want = MD5_BLOCKSIZE - self.tmp_len;
            let take = want.min(data.len());
            self.tmp[self.tmp_len..self.tmp_len + take].copy_from_slice(&data[..take]);
            self.tmp_len += take;
            data = &data[take..];
            if self.tmp_len == MD5_BLOCKSIZE {
                let tmp = self.tmp;
                md5_block_x1(&mut self.state, &tmp);
                self.tmp_len = 0;
            }
            if data.is_empty() {
                return;
            }
        }
        while data.len() >= MD5_BLOCKSIZE {
            md5_block_x1(&mut self.state, &data[..MD5_BLOCKSIZE]);
            data = &data[MD5_BLOCKSIZE..];
        }
        self.tmp[..data.len()].copy_from_slice(data);
        self.tmp_len = data.len();
    }
    /// Standard MD5 finalisation (zeroPad == 0 path of md5_final_block).
    pub fn finish(&mut self) -> [u8; 16] {
        let mut st = self.state;
        md5_final_block(&mut st, &self.tmp, self.data_len, 0);
        digest_bytes(&st)
    }
}

/// Write the 4-word state as a 16-byte little-endian digest.
pub fn digest_bytes(state: &[u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (j, w) in state.iter().enumerate() {
        out[j * 4..j * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Port of md5-final.c: finishes `state` where `data` holds the trailing
/// `totalLength & 63` bytes, with `zeroPad` additional *virtual* zero bytes
/// logically appended before MD5 padding.
pub fn md5_final_block(
    state: &mut [u32; 4],
    data: &[u8],
    mut total_length: u64,
    mut zero_pad: u64,
) {
    let mut block = [0u8; MD5_BLOCKSIZE];
    let mut remaining = (total_length & 63) as usize;
    block[..remaining].copy_from_slice(&data[..remaining]);
    total_length = total_length.wrapping_add(zero_pad);
    // loopState: 0 = consume tmp remainder + zeros, 1 = consume pure zero blocks,
    // 2 = place 0x80, 3 = spill to a second padding block, 4 = length block.
    // wrapping_add mirrors the C++ (remaining + zeroPad < 64) overflow
    // behaviour - zero_pad is effectively bounded, but keep parity cheaply
    let mut loop_state = if (remaining as u64).wrapping_add(zero_pad) < 64 {
        2
    } else {
        0
    };
    loop {
        if loop_state == 1 && zero_pad < 64 {
            loop_state = 2;
        }
        if loop_state == 2 {
            remaining = (total_length & 63) as usize;
            block[remaining] = 0x80;
            remaining += 1;
            if remaining <= 64 - 8 {
                loop_state = 4;
            } else {
                loop_state = 3;
                remaining = 0;
            }
        }
        if loop_state == 4 {
            for b in &mut block[remaining..64 - 8] {
                *b = 0;
            }
            let bits = total_length << 3;
            block[56..64].copy_from_slice(&bits.to_le_bytes());
        }
        md5_process_block(state, &block);
        if loop_state == 4 {
            break;
        } else if loop_state == 3 {
            loop_state = 4;
        } else if loop_state == 1 {
            zero_pad -= 64;
        } else if loop_state == 0 {
            block = [0u8; MD5_BLOCKSIZE];
            zero_pad -= 64 - remaining as u64;
            loop_state = 1;
        }
    }
}

#[cfg(test)]
mod sse_tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x2_sse_vs_scalar() {
        let mut rng = 0x99u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        for _ in 0..8 {
            let a: Vec<u8> = (0..64).map(|_| next()).collect();
            let b: Vec<u8> = (0..64).map(|_| next()).collect();
            let mut st = Md5X2::new();
            let mut want = st.state;
            let (l0, l1) = want.split_at_mut(4);
            md5_process_block(l0.try_into().unwrap(), &a);
            md5_process_block(l1.try_into().unwrap(), &b);
            unsafe {
                if x86_use_bmi1() {
                    md5_block_x2_asm_bmi1(st.state.as_mut_ptr(), a.as_ptr(), b.as_ptr());
                } else {
                    md5_block_x2_asm(st.state.as_mut_ptr(), a.as_ptr(), b.as_ptr());
                }
            }
            assert_eq!(st.state, want);
        }
    }
}

// ===== single-lane variants (lane-split hashing) =====

#[cfg(target_arch = "x86_64")]
macro_rules! md5_rf4_x1 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             add {a1:e}, {K0}
             xor {t1:e}, {c1:e}
             and {t1:e}, {b1:e}
             xor {t1:e}, {d1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             mov {t1:e}, {c1:e}
             add {d1:e}, {K1}
             xor {t1:e}, {b1:e}
             and {t1:e}, {a1:e}
             xor {t1:e}, {c1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             mov {t1:e}, {b1:e}
             add {c1:e}, {K2}
             xor {t1:e}, {a1:e}
             and {t1:e}, {d1:e}
             xor {t1:e}, {b1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             mov {t1:e}, {a1:e}
             add {b1:e}, {K3}
             xor {t1:e}, {d1:e}
             and {t1:e}, {c1:e}
             xor {t1:e}, {a1:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_rg4_x1 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             add {a1:e}, {K0}
             not {t1:e}
             and {t1:e}, {c1:e}
             add {a1:e}, {t1:e}
             mov {t1:e}, {d1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             and {t1:e}, {b1:e}
             add {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             mov {t1:e}, {c1:e}
             add {d1:e}, {K1}
             not {t1:e}
             and {t1:e}, {b1:e}
             add {d1:e}, {t1:e}
             mov {t1:e}, {c1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             and {t1:e}, {a1:e}
             add {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             mov {t1:e}, {b1:e}
             add {c1:e}, {K2}
             not {t1:e}
             and {t1:e}, {a1:e}
             add {c1:e}, {t1:e}
             mov {t1:e}, {b1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             and {t1:e}, {d1:e}
             add {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             mov {t1:e}, {a1:e}
             add {b1:e}, {K3}
             not {t1:e}
             and {t1:e}, {d1:e}
             add {b1:e}, {t1:e}
             mov {t1:e}, {a1:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             and {t1:e}, {c1:e}
             add {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_rh4_x1 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             add {a1:e}, {K0}
             xor {t1:e}, {c1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             xor {t1:e}, {b1:e}
             add {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             mov {t1:e}, {c1:e}
             add {d1:e}, {K1}
             xor {t1:e}, {b1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             xor {t1:e}, {a1:e}
             add {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             mov {t1:e}, {b1:e}
             add {c1:e}, {K2}
             xor {t1:e}, {a1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             xor {t1:e}, {d1:e}
             add {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             mov {t1:e}, {a1:e}
             add {b1:e}, {K3}
             xor {t1:e}, {d1:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             xor {t1:e}, {c1:e}
             add {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4_x1 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    //
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             add {a1:e}, {K0}
             not {t1:e}
             or {t1:e}, {b1:e}
             xor {t1:e}, {c1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             mov {t1:e}, {c1:e}
             add {d1:e}, {K1}
             not {t1:e}
             or {t1:e}, {a1:e}
             xor {t1:e}, {b1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             mov {t1:e}, {b1:e}
             add {c1:e}, {K2}
             not {t1:e}
             or {t1:e}, {d1:e}
             xor {t1:e}, {a1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             mov {t1:e}, {a1:e}
             add {b1:e}, {K3}
             not {t1:e}
             or {t1:e}, {c1:e}
             xor {t1:e}, {d1:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             add {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4_last_x1 {
    // 4 rounds, roles rotate (a,b,c,d)->(d,a,b,c)->(c,d,a,b)->(b,c,d,a)
    // last round has no message-word prefetch
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             mov {t1:e}, {d1:e}
             add {a1:e}, {K0}
             not {t1:e}
             or {t1:e}, {b1:e}
             xor {t1:e}, {c1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             add {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             mov {t1:e}, {c1:e}
             add {d1:e}, {K1}
             not {t1:e}
             or {t1:e}, {a1:e}
             xor {t1:e}, {b1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             add {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             mov {t1:e}, {b1:e}
             add {c1:e}, {K2}
             not {t1:e}
             or {t1:e}, {d1:e}
             xor {t1:e}, {a1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             add {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             mov {t1:e}, {a1:e}
             add {b1:e}, {K3}
             not {t1:e}
             or {t1:e}, {c1:e}
             xor {t1:e}, {d1:e}
             add {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_rg4_bmi1_x1 {
    // BMI1 variant: andn replaces mov+not+and in the first G term
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             add {a1:e}, {K0}
             andn {t1:e}, {d1:e}, {c1:e}
             add {a1:e}, {t1:e}
             mov {t1:e}, {d1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             and {t1:e}, {b1:e}
             add {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             add {d1:e}, {K1}
             andn {t1:e}, {c1:e}, {b1:e}
             add {d1:e}, {t1:e}
             mov {t1:e}, {c1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             and {t1:e}, {a1:e}
             add {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             add {c1:e}, {K2}
             andn {t1:e}, {b1:e}, {a1:e}
             add {c1:e}, {t1:e}
             mov {t1:e}, {b1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             and {t1:e}, {d1:e}
             add {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             add {b1:e}, {K3}
             andn {t1:e}, {a1:e}, {d1:e}
             add {b1:e}, {t1:e}
             mov {t1:e}, {a1:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             and {t1:e}, {c1:e}
             add {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const $k0, K1 = const $k1, K2 = const $k2, K3 = const $k3,
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4_bmi1_x1 {
    // BMI1 variant: I(x,y,z)=y^(x|~z) evaluated as -( z^(~x&y) )-1 via andn+sub
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             add {a1:e}, {K0}
             andn {t1:e}, {b1:e}, {d1:e}
             xor {t1:e}, {c1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             sub {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             add {d1:e}, {K1}
             andn {t1:e}, {a1:e}, {c1:e}
             xor {t1:e}, {b1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             sub {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             add {c1:e}, {K2}
             andn {t1:e}, {d1:e}, {b1:e}
             xor {t1:e}, {a1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             sub {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             add {b1:e}, {K3}
             andn {t1:e}, {c1:e}, {a1:e}
             xor {t1:e}, {d1:e}
             add {a1:e}, DWORD PTR [{p}+{OFF3}]
             sub {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4), OFF3 = const ($i3 * 4),
             K0 = const ($k0 - 1), K1 = const ($k1 - 1), K2 = const ($k2 - 1), K3 = const ($k3 - 1),
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_ri4_last_bmi1_x1 {
    // BMI1 variant of md5_ri4_last - final round has no message prefetch
    ($a1:ident, $b1:ident, $c1:ident, $d1:ident, $p:ident,
     $i0:expr, $i1:expr, $i2:expr,
     $k0:expr, $k1:expr, $k2:expr, $k3:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        std::arch::asm!(
"             add {a1:e}, {K0}
             andn {t1:e}, {b1:e}, {d1:e}
             xor {t1:e}, {c1:e}
             add {d1:e}, DWORD PTR [{p}+{OFF0}]
             sub {a1:e}, {t1:e}
             rol {a1:e}, {R0}
             add {a1:e}, {b1:e}
             add {d1:e}, {K1}
             andn {t1:e}, {a1:e}, {c1:e}
             xor {t1:e}, {b1:e}
             add {c1:e}, DWORD PTR [{p}+{OFF1}]
             sub {d1:e}, {t1:e}
             rol {d1:e}, {R1}
             add {d1:e}, {a1:e}
             add {c1:e}, {K2}
             andn {t1:e}, {d1:e}, {b1:e}
             xor {t1:e}, {a1:e}
             add {b1:e}, DWORD PTR [{p}+{OFF2}]
             sub {c1:e}, {t1:e}
             rol {c1:e}, {R2}
             add {c1:e}, {d1:e}
             add {b1:e}, {K3}
             andn {t1:e}, {c1:e}, {a1:e}
             xor {t1:e}, {d1:e}
             sub {b1:e}, {t1:e}
             rol {b1:e}, {R3}
             add {b1:e}, {c1:e}
",
            a1 = inout(reg) $a1, b1 = inout(reg) $b1,
            c1 = inout(reg) $c1, d1 = inout(reg) $d1,
            t1 = out(reg) _, p = in(reg) $p,              OFF0 = const ($i0 * 4), OFF1 = const ($i1 * 4), OFF2 = const ($i2 * 4),
             K0 = const ($k0 - 1), K1 = const ($k1 - 1), K2 = const ($k2 - 1), K3 = const ($k3 - 1),
             R0 = const $r0, R1 = const $r1, R2 = const $r2, R3 = const $r3,
            options(nostack, pure, readonly),
        )
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! md5_block_x1_impl {
    ($fname:ident, $rg4:ident, $ri4:ident, $ri4l:ident) => {
        unsafe fn $fname(state: *mut u32, src0: *const u8) {
            let p = src0;
            let mut a1 = state.add(0).read();
            let mut b1 = state.add(1).read();
            let mut c1 = state.add(2).read();
            let mut d1 = state.add(3).read();
            // first message words folded in ahead of round 0 (like `A1 += read32()`)
            a1 = a1.wrapping_add((p as *const u32).read_unaligned());

            // RF4/RG4/RH4/RI4 groups - rotation is baked into each template
            md5_rf4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                1,
                2,
                3,
                4,
                0xd76aa478u32,
                0xe8c7b756u32,
                0x242070dbu32,
                0xc1bdceeeu32,
                7,
                12,
                17,
                22
            );
            md5_rf4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                5,
                6,
                7,
                8,
                0xf57c0fafu32,
                0x4787c62au32,
                0xa8304613u32,
                0xfd469501u32,
                7,
                12,
                17,
                22
            );
            md5_rf4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                9,
                10,
                11,
                12,
                0x698098d8u32,
                0x8b44f7afu32,
                0xffff5bb1u32,
                0x895cd7beu32,
                7,
                12,
                17,
                22
            );
            md5_rf4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                13,
                14,
                15,
                1,
                0x6b901122u32,
                0xfd987193u32,
                0xa679438eu32,
                0x49b40821u32,
                7,
                12,
                17,
                22
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                p,
                6,
                11,
                0,
                5,
                0xf61e2562u32,
                0xc040b340u32,
                0x265e5a51u32,
                0xe9b6c7aau32,
                5,
                9,
                14,
                20
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                p,
                10,
                15,
                4,
                9,
                0xd62f105du32,
                0x02441453u32,
                0xd8a1e681u32,
                0xe7d3fbc8u32,
                5,
                9,
                14,
                20
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                p,
                14,
                3,
                8,
                13,
                0x21e1cde6u32,
                0xc33707d6u32,
                0xf4d50d87u32,
                0x455a14edu32,
                5,
                9,
                14,
                20
            );
            $rg4!(
                a1,
                b1,
                c1,
                d1,
                p,
                2,
                7,
                12,
                5,
                0xa9e3e905u32,
                0xfcefa3f8u32,
                0x676f02d9u32,
                0x8d2a4c8au32,
                5,
                9,
                14,
                20
            );
            md5_rh4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                8,
                11,
                14,
                1,
                0xfffa3942u32,
                0x8771f681u32,
                0x6d9d6122u32,
                0xfde5380cu32,
                4,
                11,
                16,
                23
            );
            md5_rh4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                4,
                7,
                10,
                13,
                0xa4beea44u32,
                0x4bdecfa9u32,
                0xf6bb4b60u32,
                0xbebfbc70u32,
                4,
                11,
                16,
                23
            );
            md5_rh4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                0,
                3,
                6,
                9,
                0x289b7ec6u32,
                0xeaa127fau32,
                0xd4ef3085u32,
                0x04881d05u32,
                4,
                11,
                16,
                23
            );
            md5_rh4_x1!(
                a1,
                b1,
                c1,
                d1,
                p,
                12,
                15,
                2,
                0,
                0xd9d4d039u32,
                0xe6db99e5u32,
                0x1fa27cf8u32,
                0xc4ac5665u32,
                4,
                11,
                16,
                23
            );
            $ri4!(
                a1,
                b1,
                c1,
                d1,
                p,
                7,
                14,
                5,
                12,
                0xf4292244u32,
                0x432aff97u32,
                0xab9423a7u32,
                0xfc93a039u32,
                6,
                10,
                15,
                21
            );
            $ri4!(
                a1,
                b1,
                c1,
                d1,
                p,
                3,
                10,
                1,
                8,
                0x655b59c3u32,
                0x8f0ccc92u32,
                0xffeff47du32,
                0x85845dd1u32,
                6,
                10,
                15,
                21
            );
            $ri4!(
                a1,
                b1,
                c1,
                d1,
                p,
                15,
                6,
                13,
                4,
                0x6fa87e4fu32,
                0xfe2ce6e0u32,
                0xa3014314u32,
                0x4e0811a1u32,
                6,
                10,
                15,
                21
            );
            $ri4l!(
                a1,
                b1,
                c1,
                d1,
                p,
                11,
                2,
                9,
                0xf7537e82u32,
                0xbd3af235u32,
                0x2ad7d2bbu32,
                0xeb86d391u32,
                6,
                10,
                15,
                21
            );

            state.add(0).write(state.add(0).read().wrapping_add(a1));
            state.add(1).write(state.add(1).read().wrapping_add(b1));
            state.add(2).write(state.add(2).read().wrapping_add(c1));
            state.add(3).write(state.add(3).read().wrapping_add(d1));
        }
    };
}

#[cfg(target_arch = "x86_64")]
md5_block_x1_impl!(md5_block_x1_asm, md5_rg4_x1, md5_ri4_x1, md5_ri4_last_x1);
#[cfg(target_arch = "x86_64")]
md5_block_x1_impl!(
    md5_block_x1_asm_bmi1,
    md5_rg4_bmi1_x1,
    md5_ri4_bmi1_x1,
    md5_ri4_last_bmi1_x1
);

#[cfg(test)]
mod x2_tests {
    use super::*;
    #[test]
    fn x2_matches_scalar() {
        let mut d0 = vec![0u8; 64 * 37];
        let mut d1 = vec![0u8; 64 * 37];
        for (i, b) in d0.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2654435761) as u8;
        }
        for (i, b) in d1.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2246822519) as u8;
        }
        let init = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
        let mut s0 = init;
        let mut s1 = [0x10325476u32, 0x98badcfe, 0xefcdab89, 0x67452301];
        for (b0, b1) in d0.chunks_exact(64).zip(d1.chunks_exact(64)) {
            md5_process_block_x2(&mut s0, b0, &mut s1, b1);
        }
        let mut r0 = init;
        let mut r1 = [0x10325476u32, 0x98badcfe, 0xefcdab89, 0x67452301];
        for (b0, b1) in d0.chunks_exact(64).zip(d1.chunks_exact(64)) {
            md5_process_block(&mut r0, b0);
            md5_process_block(&mut r1, b1);
        }
        assert_eq!(s0, r0);
        assert_eq!(s1, r1);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod x1_tests {
    use super::*;
    #[test]
    fn x1_matches_scalar() {
        let mut data = vec![0u8; 64 * 37];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2654435761) as u8;
        }
        let mut s_asm = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
        let mut s_ref = s_asm;
        unsafe {
            for blk in data.chunks_exact(64) {
                md5_block_x1_asm(s_asm.as_mut_ptr(), blk.as_ptr());
            }
        }
        for blk in data.chunks_exact(64) {
            md5_process_block(&mut s_ref, blk);
        }
        assert_eq!(s_asm, s_ref);
    }
    #[test]
    #[ignore]
    fn x1_bench() {
        let gib = 1usize << 30;
        let data = vec![0xabu8; gib];
        let mut st = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
        let t = std::time::Instant::now();
        unsafe {
            for blk in data.chunks_exact(64) {
                md5_block_x1_asm(st.as_mut_ptr(), blk.as_ptr());
            }
        }
        let el = t.elapsed().as_secs_f64();
        eprintln!("x1: {:.2}s {:.2} GB/s", el, gib as f64 / el / 1e9);
    }
    #[test]
    fn x1_bmi1_matches_scalar() {
        if !super::x86_use_bmi1() {
            return;
        }
        let mut data = vec![0u8; 64 * 41];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(40503) as u8;
        }
        let mut s_asm = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
        unsafe {
            for blk in data.chunks_exact(64) {
                md5_block_x1_asm_bmi1(s_asm.as_mut_ptr(), blk.as_ptr());
            }
        }
        let mut s_ref = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
        for blk in data.chunks_exact(64) {
            md5_process_block(&mut s_ref, blk);
        }
        assert_eq!(s_asm, s_ref);
    }
}

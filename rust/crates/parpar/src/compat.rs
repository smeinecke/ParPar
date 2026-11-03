//! JS/Node compatibility primitives used by the ported argument parser and
//! orchestration.
//!
//! The reference implementation runs on Node/V8, so a number of observable
//! behaviours are JS-specific: ToInt32 coercion, Math.round's tie direction,
//! Number(string) parsing, Node's fixed Buffer-encoding label set, and the
//! process.arch names ParPar embeds in PAR2 Creator packets.

/// JS ToInt32 (the `x|0` idiom): ToNumber, truncate toward zero, wrap to a
/// signed 32-bit range. NaN/Infinity map to 0.
pub fn to_int32(v: f64) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    const M: f64 = 4294967296.0; // 2^32
    let mut i = v % M;
    if i > 2147483647.0 {
        i -= M;
    } else if i < -2147483648.0 {
        i += M;
    }
    i as i32
}

/// `x|0` applied directly to a string argument, as the parser does.
pub fn str_to_int32(s: &str) -> i32 {
    to_int32(js_number(s))
}

/// JS `Math.round`: nearest integer, ties toward positive infinity
/// (`Math.round(-1.5) === -1`, unlike Rust's round which gives -2).
/// Preserves -0 for inputs in (-0.5, 0].
pub fn js_math_round(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() || v == 0.0 {
        return v;
    }
    let r = (v + 0.5).floor();
    // x+0.5 can land exactly on an integer boundary for x in (-0.5,0]; JS
    // still yields -0 there
    if r == 0.0 && v.is_sign_negative() {
        -0.0
    } else {
        r
    }
}

/// JS `Math.max`/`Math.min`: NaN propagates (Rust f64::max/min return the
/// non-NaN operand - divergent on solver spec expressions that yield NaN).
pub fn js_math_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}
pub fn js_math_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// `^\d+$` - JS \d is ASCII digits only.
pub fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `^-?\d+$`
pub fn is_signed_digits(s: &str) -> bool {
    let s = s.strip_prefix('-').unwrap_or(s);
    is_digits(s)
}

/// JS number-to-string for solver values: integral values (in positional
/// range) print without a fraction, matching V8's formatting.
pub fn fmt_num(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        // JS prints Infinity/-Infinity; Rust Display prints inf/-inf
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if v.fract() == 0.0 && v.abs() < 1e21 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// JS `Number(str)` / unary `+str`, restricted to what the CLI feeds it:
/// JS trims surrounding whitespace, maps "" to 0, accepts only the literal
/// "Infinity"/"inf"? No - JS Number() accepts "Infinity" and signed variants
/// but NOT "inf", and accepts 0x/0o/0b prefixes. Rust's f64::from_str accepts
/// "inf"/"nan" spellings JS rejects, so guard those.
pub fn js_number(s: &str) -> f64 {
    let t = s.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}');
    if t.is_empty() {
        return 0.0;
    }
    let (neg, body) = match t.as_bytes()[0] {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    if body == "Infinity" {
        return if neg {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if body.len() > 2 && body.as_bytes()[0] == b'0' {
        let (radix, rest) = match body.as_bytes()[1] {
            b'x' | b'X' => (16u32, &body[2..]),
            b'o' | b'O' => (8u32, &body[2..]),
            b'b' | b'B' => (2u32, &body[2..]),
            _ => (0u32, &body[0..]),
        };
        if radix != 0 {
            if rest.is_empty() || !rest.bytes().all(|b| (b as char).is_digit(radix)) {
                return f64::NAN;
            }
            let v = u128::from_str_radix(rest, radix).unwrap_or(u128::MAX) as f64;
            return if neg { -v } else { v };
        }
    }
    // Rust accepts "inf"/"NaN" spellings that JS rejects
    if body.eq_ignore_ascii_case("nan") || body.eq_ignore_ascii_case("inf") {
        return f64::NAN;
    }
    match body.parse::<f64>() {
        Ok(v) => {
            if neg {
                -v
            } else {
                v
            }
        }
        Err(_) => f64::NAN,
    }
}

/// JS `parseFloat`: parses the longest valid numeric prefix (sign, digits,
/// one decimal point, exponent) and ignores trailing junk. NaN when no
/// numeric prefix exists.
pub fn js_parse_float(s: &str) -> f64 {
    let t = s.trim_start_matches(|c: char| c.is_whitespace() || c == '\u{feff}');
    let b = t.as_bytes();
    let mut i = 0usize;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    // Infinity literal
    if t[i..].starts_with("Infinity") {
        return compat_sign(t, i);
    }
    let start_digits = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i == start_digits || (i == start_digits + 1 && t.as_bytes()[start_digits] == b'.') {
        return f64::NAN;
    }
    // optional exponent
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let d = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > d {
            i = j;
        }
    }
    t[..i].parse::<f64>().unwrap_or(f64::NAN)
}

fn compat_sign(t: &str, i: usize) -> f64 {
    if t.as_bytes().first() == Some(&b'-') && i >= 1 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

/// Node `process.arch` name for the build target, per the mapping table in
/// the migration plan: x86_64->x64, aarch64->arm64, arm->arm, i686->ia32,
/// riscv64->riscv64 (which the solver's 64-bit check intentionally excludes),
/// powerpc64->ppc64.
pub fn node_arch() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    return "x64";
    #[cfg(target_arch = "x86")]
    return "ia32";
    #[cfg(target_arch = "aarch64")]
    return "arm64";
    #[cfg(all(target_arch = "arm", not(target_arch = "aarch64")))]
    return "arm";
    #[cfg(target_arch = "riscv64")]
    return "riscv64";
    #[cfg(target_arch = "riscv32")]
    return "riscv32";
    #[cfg(target_arch = "powerpc64")]
    return "ppc64";
    #[cfg(target_arch = "powerpc")]
    return "ppc32";
    #[cfg(target_arch = "s390x")]
    return "s390x";
    #[cfg(target_arch = "mips64")]
    return "mips64el";
    #[cfg(target_arch = "mips")]
    return "mipsel";
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv64",
        target_arch = "riscv32",
        target_arch = "powerpc64",
        target_arch = "powerpc",
        target_arch = "s390x",
        target_arch = "mips64",
        target_arch = "mips"
    )))]
    return "unknown";
}

/// Node Buffer encodings are a fixed label set, not a charset registry.
/// Validation mirrors `Buffer.from('', enc)` - accepts case-insensitive
/// labels from this list.
const NODE_ENCODINGS: &[&str] = &[
    "utf8",
    "utf-8",
    "ascii",
    "latin1",
    "binary",
    "ucs2",
    "ucs-2",
    "utf16le",
    "utf-16le",
    "hex",
    "base64",
    "base64url",
];

/// Mirrors the `Buffer.from('', v)` validation used for `ascii-charset` and
/// `input-file-enc`.
pub fn is_node_encoding(enc: &str) -> bool {
    let e = enc.to_ascii_lowercase();
    NODE_ENCODINGS.contains(&e.as_str())
}

/// Node's `util.inspect` format for fs-style errors, as printed by
/// `console.error(err)`: `[Error: ENOENT: ...] { errno, code, syscall, path }`.
pub fn node_fs_error(
    errno: i32,
    code: &str,
    desc: &str,
    syscall: &str,
    path: Option<&str>,
) -> String {
    let msg = match path {
        Some(p) => format!("{}: {}, {} '{}'", code, desc, syscall, p),
        None => format!("{}: {}, {}", code, desc, syscall),
    };
    let mut s = format!(
        "[Error: {}] {{\n  errno: {},\n  code: '{}',\n  syscall: '{}'",
        msg, errno, code, syscall
    );
    if let Some(p) = path {
        s.push_str(&format!(",\n  path: '{}'", p));
    }
    s.push_str("\n}");
    s
}

/// errno -> Node's (code, description) - libuv `uv_strerror` texts.
/// `io::Error` -> Node inspect-format error for a syscall+path.
pub fn fs_io_error(e: &std::io::Error, syscall: &str, path: &str) -> String {
    let errno = e.raw_os_error().unwrap_or(5);
    let (code, desc) = errno_info(errno);
    node_fs_error(-errno, code, desc, syscall, Some(path))
}

pub fn errno_info(errno: i32) -> (&'static str, &'static str) {
    match errno {
        1 => ("EPERM", "operation not permitted"),
        2 => ("ENOENT", "no such file or directory"),
        5 => ("EIO", "i/o error"),
        9 => ("EBADF", "bad file descriptor"),
        11 => ("EAGAIN", "resource temporarily unavailable"),
        13 => ("EACCES", "permission denied"),
        17 => ("EEXIST", "file already exists"),
        18 => ("EXDEV", "cross-device link not permitted"),
        20 => ("ENOTDIR", "not a directory"),
        21 => ("EISDIR", "illegal operation on a directory"),
        22 => ("EINVAL", "invalid argument"),
        23 => ("ENFILE", "file table overflow"),
        24 => ("EMFILE", "too many open files"),
        27 => ("EFBIG", "file too large"),
        28 => ("ENOSPC", "no space left on device"),
        30 => ("EROFS", "read-only file system"),
        31 => ("EMLINK", "too many links"),
        36 => ("ENAMETOOLONG", "name too long"),
        39 => ("ENOTEMPTY", "directory not empty"),
        40 => ("ELOOP", "too many levels of symbolic links"),
        122 => ("EDQUOT", "disk quota exceeded"),
        _ => ("EIO", "i/o error"),
    }
}

/// `process.platform` name for the build target (used for proc:// shell
/// selection and any platform branching).
pub fn node_platform() -> &'static str {
    #[cfg(target_os = "windows")]
    return "win32";
    #[cfg(target_os = "linux")]
    return "linux";
    #[cfg(target_os = "macos")]
    return "darwin";
    #[cfg(target_os = "freebsd")]
    return "freebsd";
    #[cfg(target_os = "openbsd")]
    return "openbsd";
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "openbsd"
    )))]
    return "unknown";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_int32_matches_js() {
        assert_eq!(to_int32(0.0), 0);
        assert_eq!(to_int32(1.0), 1);
        assert_eq!(to_int32(-1.0), -1);
        assert_eq!(to_int32(1.9), 1);
        assert_eq!(to_int32(-1.9), -1);
        assert_eq!(to_int32(2147483647.0), 2147483647);
        assert_eq!(to_int32(2147483648.0), -2147483648);
        assert_eq!(to_int32(4294967296.0), 0);
        assert_eq!(to_int32(4294967297.0), 1);
        assert_eq!(to_int32(-4294967296.0), 0);
        assert_eq!(to_int32(f64::NAN), 0);
        assert_eq!(to_int32(f64::INFINITY), 0);
        assert_eq!(to_int32(-f64::INFINITY), 0);
        assert_eq!(to_int32(1e10), 1410065408); // 1e10 mod 2^32
    }

    #[test]
    fn str_int32() {
        assert_eq!(str_to_int32("4294967296"), 0);
        assert_eq!(str_to_int32("-5"), -5);
        assert_eq!(str_to_int32("9999999999999"), 1316134911);
    }

    #[test]
    fn js_round_negative_ties() {
        assert_eq!(js_math_round(-1.5), -1.0);
        assert_eq!(js_math_round(-2.5), -2.0);
        assert_eq!(js_math_round(1.5), 2.0);
        assert_eq!(js_math_round(2.5), 3.0);
        assert!(js_math_round(-0.4).is_sign_negative());
        assert_eq!(js_math_round(-0.4), 0.0);
        assert!(js_math_round(f64::NAN).is_nan());
        assert_eq!(js_math_round(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn js_number_parse() {
        assert_eq!(js_number(" 12 "), 12.0);
        assert_eq!(js_number(""), 0.0);
        assert_eq!(js_number("Infinity"), f64::INFINITY);
        assert_eq!(js_number("-Infinity"), f64::NEG_INFINITY);
        assert!(js_number("inf").is_nan());
        assert!(js_number("NaN").is_nan());
        assert_eq!(js_number("0x10"), 16.0);
        assert_eq!(js_number("0o17"), 15.0);
        assert_eq!(js_number("0b101"), 5.0);
        assert!(js_number("0x").is_nan());
        assert!(js_number("0xzz").is_nan());
        assert_eq!(js_number("1e3"), 1000.0);
        assert_eq!(js_number("1."), 1.0);
        assert_eq!(js_number(".5"), 0.5);
        assert!(js_number("1.2.3").is_nan());
        assert!(js_number("abc").is_nan());
    }

    #[test]
    fn encodings() {
        assert!(is_node_encoding("utf-8"));
        assert!(is_node_encoding("UTF8"));
        assert!(is_node_encoding("latin1"));
        assert!(is_node_encoding("binary"));
        assert!(is_node_encoding("base64url"));
        assert!(!is_node_encoding("shift-jis"));
        assert!(!is_node_encoding("windows-1252"));
    }

    #[test]
    fn arch_mapping() {
        // on this dev machine
        #[cfg(target_arch = "x86_64")]
        assert_eq!(node_arch(), "x64");
    }
}

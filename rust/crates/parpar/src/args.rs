//! Port of bin/parpar.js argument handling: the option spec table plus all
//! post-parse processing - spec expressions (`l*2`, `50%`, `+-` chains,
//! `<`/`>` rounding prefixes, size units), option conflicts and defaults
//! that mutate parsed args, OpenCL device specs, and construction of the
//! ParParOpts (ppo) object passed to PAR2Gen.
//!
//! `parseObject` from arg_parser.js is not ported: the CLI only uses the
//! array form (config-file parsing was an exported-library feature).

use crate::cliparse::{parse, ArgVal, OptFn, OptSpec, OptType, ParsedArgs, ThrottleSpec};
use crate::compat;
use std::collections::HashMap;

pub use crate::cliparse::{parse_size, parse_time, ThrottleMode};

// ---------------------------------------------------------------------------
// option spec table (bin/parpar.js `opts`)
// ---------------------------------------------------------------------------

fn str_spec() -> OptSpec {
    OptSpec::default()
}

pub fn opts_table() -> HashMap<&'static str, OptSpec> {
    use OptType::*;
    let mut o: HashMap<&'static str, OptSpec> = HashMap::new();
    let mut add = |k: &'static str, s: OptSpec| {
        o.insert(k, s);
    };
    add("input-slices", {
        let mut s = str_spec();
        s.alias = Some('s');
        s
    });
    add("min-input-slices", str_spec());
    add("max-input-slices", str_spec());
    add("slice-size-multiple", str_spec());
    add(
        "auto-slice-size",
        OptSpec {
            type_: Bool,
            alias: Some('S'),
            ..str_spec()
        },
    );
    add(
        "recovery-slices",
        OptSpec {
            type_: Str,
            alias: Some('r'),
            ..str_spec()
        },
    );
    add("min-recovery-slices", str_spec());
    add("max-recovery-slices", str_spec());
    add(
        "recovery-offset",
        OptSpec {
            type_: Int,
            alias: Some('e'),
            map: Some("recoveryOffset"),
            ..str_spec()
        },
    );
    add(
        "recovery-exponents",
        OptSpec {
            type_: List,
            map: Some("recoveryExponents"),
            ..str_spec()
        },
    );
    add(
        "comment",
        OptSpec {
            type_: Array,
            alias: Some('c'),
            map: Some("comments"),
            ..str_spec()
        },
    );
    add(
        "packet-redundancy",
        OptSpec {
            type_: Str,
            map: Some("criticalRedundancyScheme"),
            ..str_spec()
        },
    );
    add(
        "min-packet-redundancy",
        OptSpec {
            type_: Int,
            map: Some("minCriticalRedundancy"),
            ..str_spec()
        },
    );
    add(
        "max-packet-redundancy",
        OptSpec {
            type_: Int,
            map: Some("maxCriticalRedundancy"),
            ..str_spec()
        },
    );
    add(
        "filepath-format",
        OptSpec {
            type_: Enum(&["basename", "keep", "common", "outrel", "path"]),
            alias: Some('f'),
            map: Some("displayNameFormat"),
            ..str_spec()
        },
    );
    add(
        "filepath-base",
        OptSpec {
            type_: Str,
            alias: Some('B'),
            map: Some("displayNameBase"),
            ..str_spec()
        },
    );
    add(
        "unicode",
        OptSpec {
            type_: Bool,
            default: Some(ArgVal::Null),
            map: Some("unicode"),
            ..str_spec()
        },
    );
    add(
        "ascii-charset",
        OptSpec {
            type_: Str,
            default: Some(ArgVal::Str("utf-8".into())),
            func: Some(OptFn::AsciiCharset),
            ..str_spec()
        },
    );
    add(
        "out",
        OptSpec {
            type_: Str,
            alias: Some('o'),
            ..str_spec()
        },
    );
    add(
        "overwrite",
        OptSpec {
            type_: Bool,
            alias: Some('O'),
            map: Some("outputOverwrite"),
            ..str_spec()
        },
    );
    add(
        "write-sync",
        OptSpec {
            type_: Bool,
            map: Some("outputSync"),
            ..str_spec()
        },
    );
    add(
        "std-naming",
        OptSpec {
            type_: Bool,
            alias: Some('n'),
            map: Some("outputAltNamingScheme"),
            default: Some(ArgVal::Bool(true)),
            func: Some(OptFn::Invert),
            ..str_spec()
        },
    );
    add(
        "slice-dist",
        OptSpec {
            type_: Enum(&["equal", "uniform", "pow2"]),
            alias: Some('d'),
            map: Some("outputSizeScheme"),
            ..str_spec()
        },
    );
    add(
        "slices-per-file",
        OptSpec {
            type_: Str,
            alias: Some('p'),
            ..str_spec()
        },
    );
    add("slices-first-file", str_spec());
    add(
        "recovery-files",
        OptSpec {
            type_: Int,
            alias: Some('F'),
            map: Some("outputFileCount"),
            ..str_spec()
        },
    );
    add(
        "noindex",
        OptSpec {
            type_: Bool,
            map: Some("outputIndex"),
            default: Some(ArgVal::Bool(true)),
            func: Some(OptFn::Invert),
            ..str_spec()
        },
    );
    add(
        "memory",
        OptSpec {
            type_: Size0,
            alias: Some('m'),
            map: Some("memoryLimit"),
            ..str_spec()
        },
    );
    add(
        "threads",
        OptSpec {
            type_: Int,
            alias: Some('t'),
            default: Some(ArgVal::Null),
            map: Some("numThreads"),
            ..str_spec()
        },
    );
    add(
        "min-chunk-size",
        OptSpec {
            type_: Size0,
            map: Some("minChunkSize"),
            ..str_spec()
        },
    );
    add(
        "seq-read-size",
        OptSpec {
            type_: Size,
            map: Some("seqReadSize"),
            ..str_spec()
        },
    );
    add(
        "chunk-read-threads",
        OptSpec {
            type_: Int,
            map: Some("chunkReadThreads"),
            ..str_spec()
        },
    );
    add(
        "seq-read-throttle",
        OptSpec {
            type_: Str,
            map: Some("seqReadThrottle"),
            func: Some(OptFn::Throttle("seq")),
            ..str_spec()
        },
    );
    add(
        "chunk-read-throttle",
        OptSpec {
            type_: Str,
            map: Some("chunkReadThrottle"),
            func: Some(OptFn::Throttle("chunk")),
            ..str_spec()
        },
    );
    add(
        "read-buffers",
        OptSpec {
            type_: Int,
            map: Some("readBuffers"),
            ..str_spec()
        },
    );
    add(
        "read-hash-queue",
        OptSpec {
            type_: Int,
            map: Some("readHashQueue"),
            ..str_spec()
        },
    );
    add(
        "proc-batch-size",
        OptSpec {
            type_: Int,
            map: Some("processBatchSize"),
            ..str_spec()
        },
    );
    add(
        "md5-batch-size",
        OptSpec {
            type_: Int,
            map: Some("hashBatchSize"),
            ..str_spec()
        },
    );
    add(
        "recovery-buffers",
        OptSpec {
            type_: Int,
            map: Some("recDataSize"),
            ..str_spec()
        },
    );
    add(
        "method",
        OptSpec {
            type_: Str,
            default: Some(ArgVal::Str("".into())),
            map: Some("gfMethod"),
            ..str_spec()
        },
    );
    add(
        "loop-tile-size",
        OptSpec {
            type_: Size,
            map: Some("loopTileSize"),
            ..str_spec()
        },
    );
    add("hash-method", str_spec());
    add("md5-method", str_spec());
    add(
        "opencl",
        OptSpec {
            type_: Array,
            ..str_spec()
        },
    );
    add("opencl-process", str_spec());
    add(
        "opencl-device",
        OptSpec {
            type_: Str,
            default: Some(ArgVal::Str(":".into())),
            ..str_spec()
        },
    );
    add("opencl-memory", str_spec());
    add("opencl-method", str_spec());
    add(
        "opencl-batch-size",
        OptSpec {
            type_: Int,
            ..str_spec()
        },
    );
    add(
        "opencl-iter-count",
        OptSpec {
            type_: Int,
            ..str_spec()
        },
    );
    add(
        "opencl-grouping",
        OptSpec {
            type_: Int,
            ..str_spec()
        },
    );
    add(
        "opencl-minchunk",
        OptSpec {
            type_: Size0,
            ..str_spec()
        },
    );
    add(
        "opencl-list",
        OptSpec {
            type_: Str,
            if_set_default: Some(ArgVal::Str("gpu".into())),
            ..str_spec()
        },
    );
    add(
        "cpu-minchunk",
        OptSpec {
            type_: Size0,
            map: Some("cpuMinChunkSize"),
            ..str_spec()
        },
    );
    add(
        "recurse",
        OptSpec {
            type_: Bool,
            alias: Some('R'),
            default: Some(ArgVal::Num(1.0)), // traverse folders explicitly specified
            ..str_spec()
        },
    );
    add(
        "skip-symlinks",
        OptSpec {
            type_: Bool,
            alias: Some('L'),
            ..str_spec()
        },
    );
    add(
        "input-file",
        OptSpec {
            type_: Array,
            alias: Some('i'),
            ..str_spec()
        },
    );
    add(
        "input-file0",
        OptSpec {
            type_: Array,
            alias: Some('0'),
            ..str_spec()
        },
    );
    add(
        "input-file-enc",
        OptSpec {
            type_: Str,
            default: Some(ArgVal::Str("utf8".into())),
            ..str_spec()
        },
    );
    add(
        "help",
        OptSpec {
            type_: Bool,
            alias: Some('?'),
            ..str_spec()
        },
    );
    add(
        "help-full",
        OptSpec {
            type_: Bool,
            ..str_spec()
        },
    );
    add(
        "quiet",
        OptSpec {
            type_: Bool,
            alias: Some('q'),
            ..str_spec()
        },
    );
    add(
        "progress",
        OptSpec {
            type_: Enum(&["none", "stderr", "stdout"]),
            ..str_spec()
        },
    );
    add(
        "version",
        OptSpec {
            type_: Bool,
            ..str_spec()
        },
    );
    add(
        "json",
        OptSpec {
            type_: Bool,
            ..str_spec()
        },
    );
    add(
        "skip-self-check",
        OptSpec {
            type_: Bool,
            ..str_spec()
        },
    );
    o
}

// ---------------------------------------------------------------------------
// spec-expression language (parseSizeOrNum + the expr chain in parpar.js)
// ---------------------------------------------------------------------------

/// `getMethodNum` input - a method name or a numeric index into the table
/// (numbers use the `'index N' in table` existence check).
#[derive(Debug, Clone, PartialEq)]
pub enum MethodRef {
    Name(String),
    Index(f64),
}

/// Term units produced by `parseSizeOrNum`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecUnit {
    /// JS 'slices' - raw slice count as a value term (distinct from
    /// 'count', which calcNumInputSlices negates into a count request).
    Slices,
    Count,
    Bytes,
    Ratio,
    LargestFiles,
    SmallestFiles,
    Power,
    Log,
    ILog,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpecTerm {
    pub unit: SpecUnit,
    pub value: f64,
    pub scale: f64,
}

/// An element of `ppo[k]` - either a single term (when the JS stored a bare
/// object) or the term list from `+-` expressions.
pub type SpecExpr = Vec<SpecTerm>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecRounding {
    Floor,
    Ceil,
}

fn is_rec_arg(arg: &str) -> bool {
    arg.ends_with("recovery-slices")
        || arg == "slices-per-file"
        || arg == "slices-first-file"
        || arg == "packet-redundancy"
}

/// The `input` argument to `parseSizeOrNum`: JS accepts a number or a
/// string (numbers occur when post-parse processing injects one, e.g.
/// `--auto-slice-size` sets `max-input-slices` to the number 32768).
#[derive(Debug, Clone, Copy)]
pub enum SpecInput<'a> {
    Num(f64),
    Str(&'a str),
}

impl<'a> From<&'a ArgVal> for SpecInput<'a> {
    fn from(v: &'a ArgVal) -> Self {
        match v {
            ArgVal::Num(n) => SpecInput::Num(*n),
            ArgVal::Str(s) => SpecInput::Str(s),
            _ => SpecInput::Str(""),
        }
    }
}

/// Port of `parseSizeOrNum` in bin/parpar.js.
/// `arg` is the option name (for error text + isRec classification),
/// `input` the value to parse, `multiple` the slice-size multiple used for
/// the bytes-unit snap rule.
pub fn parse_size_or_num(arg: &str, input: SpecInput, multiple: f64) -> Result<SpecTerm, String> {
    let is_rec = is_rec_arg(arg);
    let invalid = || format!("Invalid value specified for `{}`", arg);

    // `typeof input == 'number' || /^-?\d+$/.test(input)` -> `input|0`
    let int_input: Option<i32> = match input {
        SpecInput::Num(n) => Some(compat::to_int32(n)),
        SpecInput::Str(s) if compat::is_signed_digits(s) => Some(compat::str_to_int32(s)),
        _ => None,
    };
    if let Some(v) = int_input {
        if !is_rec && v < 0 {
            return Err(invalid());
        }
        return Ok(SpecTerm {
            unit: SpecUnit::Count,
            value: v as f64,
            scale: 1.0,
        });
    }
    let SpecInput::Str(input) = input else {
        // a non-integer number can't match the unit regex either
        return Err(invalid());
    };

    // ^(-?[0-9.]+)([%bkmgtpelswon]|[lswn][*\/][0-9.]+)$  (case-insensitive)
    let Some((num_part, unit_part)) = split_spec_term(input) else {
        return Err(invalid());
    };
    let n = compat::js_number(num_part);
    if n.is_nan() || !n.is_finite() || (!is_rec && n < 0.0) {
        return Err(invalid());
    }

    if unit_part == "%" {
        if !is_rec {
            return Err(invalid());
        }
        return Ok(SpecTerm {
            unit: SpecUnit::Ratio,
            value: n / 100.0,
            scale: 1.0,
        });
    }

    let first = unit_part.as_bytes()[0].to_ascii_lowercase();
    if matches!(first, b'l' | b's' | b'w' | b'o' | b'n') {
        let mut scale = 1.0;
        if unit_part.len() > 2 {
            scale = compat::js_number(&unit_part[2..]);
            if scale.is_nan() || !scale.is_finite() {
                return Err(invalid());
            }
            if unit_part.as_bytes()[1] == b'/' {
                scale = 1.0 / scale;
                if scale.is_nan() || !scale.is_finite() {
                    return Err(invalid());
                }
            }
        }
        let unit = match first {
            b'l' => SpecUnit::LargestFiles,
            b's' => SpecUnit::SmallestFiles,
            b'w' => SpecUnit::Power,
            b'o' => SpecUnit::Log,
            _ => SpecUnit::ILog,
        };
        return Ok(SpecTerm {
            unit,
            value: n,
            scale,
        });
    }

    // size unit
    let unit_pow = match unit_part.to_ascii_uppercase().as_str() {
        "E" => 6,
        "P" => 5,
        "T" => 4,
        "G" => 3,
        "M" => 2,
        "K" => 1,
        "B" => 0,
        _ => return Err(invalid()),
    };
    let unit = 1024f64.powi(unit_pow);
    let mut n = n * unit;
    if multiple != 0.0 && n % multiple != 0.0 {
        // if unit specified is too coarse, round to desired multiple
        let target = multiple.max(compat::js_math_round(n / multiple) * multiple);
        if n > multiple * 25.0 && (target - n).abs() < unit * 0.05 {
            n = target;
        }
    }
    Ok(SpecTerm {
        unit: SpecUnit::Bytes,
        value: n.floor(),
        scale: 1.0,
    })
}

/// Split `(-?[0-9.]+)(unit)` where unit is `%`, `[bkmgtpelswon]` or
/// `[lswn][*\/][0-9.]+`. Returns (num, unit); None if the shape is wrong.
fn split_spec_term(input: &str) -> Option<(&str, &str)> {
    let bytes = input.as_bytes();
    let mut i = 0usize;
    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    let num_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i == num_start {
        return None;
    }
    let unit = &input[i..];
    if unit.is_empty() {
        return None;
    }
    let ub = unit.as_bytes();
    let valid = if unit == "%" {
        true
    } else if ub.len() == 1 {
        matches!(
            ub[0].to_ascii_lowercase(),
            b'b' | b'k' | b'm' | b'g' | b't' | b'p' | b'e' | b'l' | b's' | b'w' | b'o' | b'n'
        )
    } else {
        // [lswn][*/][0-9.]+ - 'o' is NOT in the scale branch of the JS regex
        matches!(ub[0].to_ascii_lowercase(), b'l' | b's' | b'w' | b'n')
            && (ub[1] == b'*' || ub[1] == b'/')
            && ub.len() > 2
            && ub[2..].iter().all(|c| c.is_ascii_digit() || *c == b'.')
    };
    if !valid {
        return None;
    }
    Some((&input[..i], unit))
}

/// Port of the sign-normalisation + split in parpar.js:
/// `-x` -> `0-x`, `++`/`--` -> `+`, `+-`/`-+` -> `-`, then `-` -> `+-`
/// separators, split on `+`. `raw` is the original `argv[arg]` value, used
/// by JS's `input || argv[arg]` fallback when an expr term is empty.
pub fn parse_spec_expr(
    arg: &str,
    val: &str,
    raw: SpecInput,
    multiple: f64,
) -> Result<SpecExpr, String> {
    let mut expr = val.to_string();
    // leading sign
    if let Some(first) = expr.chars().next() {
        if first == '-' {
            expr = format!("0-{}", &expr[1..]);
        } else if first == '+' {
            expr = expr[1..].to_string();
        }
    }
    // collapse ++ -- +- -+
    let mut collapsed = String::new();
    let mut chars = expr.chars().peekable();
    while let Some(c) = chars.next() {
        if (c == '+' || c == '-')
            && chars
                .peek()
                .map(|n| *n == '+' || *n == '-')
                .unwrap_or(false)
        {
            let n = chars.next().unwrap();
            if (c == '+' && n == '+') || (c == '-' && n == '-') {
                collapsed.push('+');
            } else {
                collapsed.push('-');
            }
        } else {
            collapsed.push(c);
        }
    }
    let expr = collapsed.replace('-', "+-");
    let mut terms = Vec::new();
    for term in expr.split('+') {
        // JS: `input = input || argv[arg]` - an empty term parses the raw value
        let input = if term.is_empty() {
            raw
        } else {
            SpecInput::Str(term)
        };
        terms.push(parse_size_or_num(arg, input, multiple)?);
    }
    Ok(terms)
}

/// Port of the per-option post-processing loop: strips whitespace, handles
/// the `<`/`>` rounding prefix for `slices-*` options and returns the expr
/// terms. `raw` is the unstripped `argv[arg]` value for the empty-term
/// fallback.
pub fn parse_cli_spec(
    arg: &str,
    val: &str,
    multiple: f64,
) -> Result<(SpecExpr, Option<SpecRounding>), String> {
    // JS strips \s - JS regex whitespace, close enough to Rust is_whitespace
    // for these inputs
    let stripped: String = val.chars().filter(|c| !c.is_whitespace()).collect();
    let mut rounding = None;
    let mut v = stripped.as_str();
    if arg.starts_with("slices-") && (v.starts_with('<') || v.starts_with('>')) {
        rounding = Some(if v.starts_with('<') {
            SpecRounding::Floor
        } else {
            SpecRounding::Ceil
        });
        v = &v[1..];
    }
    Ok((
        parse_spec_expr(arg, v, SpecInput::Str(val), multiple)?,
        rounding,
    ))
}

// ---------------------------------------------------------------------------
// OpenCL specs
// ---------------------------------------------------------------------------

/// `ppo.openclDevices` entry (openclMap result).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OclDeviceSpec {
    pub ratio: Option<f64>,
    pub platform: Option<i32>,
    pub device: Option<i32>,
    /// The raw `device` value verbatim. For `platform:device` form the ids
    /// above are already set; otherwise it holds the device name for a
    /// deferred lookup (JS does it lazily inside openclMap) - resolution
    /// also needs it for the `openclDevicesSelected` dedup key.
    pub device_raw: Option<String>,
    /// `device_raw` came from the merged global `--opencl-device` rather
    /// than this spec's own `device=` entry. JS resolves the global spec
    /// once and `_extend` copies its already-resolved ids into entries
    /// lacking their own device, so an inherited name must NOT be
    /// re-resolved (or newly marked) per entry.
    pub device_inherited: bool,
    pub memory_limit: Option<f64>,
    pub method: Option<String>,
    pub input_batchsize: Option<i32>,
    pub target_iters: Option<i32>,
    pub target_grouping: Option<i32>,
    pub min_chunk_size: Option<f64>,
}

impl OclDeviceSpec {
    /// `_extend` merge: fields set in `other` override.
    pub fn extend(&self, other: &OclDeviceSpec) -> OclDeviceSpec {
        OclDeviceSpec {
            ratio: other.ratio.or(self.ratio),
            platform: other.platform.or(self.platform),
            device: other.device.or(self.device),
            device_raw: other.device_raw.clone().or(self.device_raw.clone()),
            device_inherited: other.device_raw.is_none() && self.device_raw.is_some(),
            memory_limit: other.memory_limit.or(self.memory_limit),
            method: other.method.clone().or(self.method.clone()),
            input_batchsize: other.input_batchsize.or(self.input_batchsize),
            target_iters: other.target_iters.or(self.target_iters),
            target_grouping: other.target_grouping.or(self.target_grouping),
            min_chunk_size: other.min_chunk_size.or(self.min_chunk_size),
        }
    }
}

/// Backend-neutral platform entry for device-name resolution.
#[derive(Debug, Clone)]
pub struct OclPlatformInfo {
    pub id: i32,
    pub devices: Vec<OclDeviceInfo>,
}

#[derive(Debug, Clone)]
pub struct OclDeviceInfo {
    pub id: i32,
    pub name: String,
}

/// True when `s` matches JS `/^\d*:\d*$/`.
pub fn is_numeric_device_spec(s: &str) -> bool {
    let mut parts = s.split(':');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(a), Some(b), None) => {
            (a.is_empty() || compat::is_digits(a)) && (b.is_empty() || compat::is_digits(b))
        }
        _ => false,
    }
}

/// Port of `openclMap` in bin/parpar.js for the fields that don't need a
/// live platform list. Device-name lookups are resolved by the caller (the
/// JS searches opencl_devices() lazily on first name).
pub fn opencl_map(data: &HashMap<String, ArgVal>) -> Result<OclDeviceSpec, String> {
    let mut ret = OclDeviceSpec::default();
    if let Some(process) = data.get("process") {
        if let ArgVal::Str(process) = process {
            let mut ratio = compat::js_parse_float(process);
            if process.ends_with('%') {
                ratio /= 100.0;
            }
            ret.ratio = Some(ratio);
        }
    }
    if let Some(ArgVal::Str(device)) = data.get("device") {
        ret.device_raw = Some(device.clone());
        if is_numeric_device_spec(device) {
            let mut spec = device.split(':');
            let p = spec.next().unwrap();
            let d = spec.next().unwrap();
            if !p.is_empty() {
                ret.platform = Some(compat::str_to_int32(p));
            }
            if !d.is_empty() {
                ret.device = Some(compat::str_to_int32(d));
            }
        }
        // non-numeric: resolved later by resolve_opencl_names
    }
    if let Some(memory) = data.get("memory") {
        ret.memory_limit = ocl_parse_size(memory);
    }
    if let Some(ArgVal::Str(method)) = data.get("method") {
        ret.method = Some(method.clone());
    }
    if let Some(v) = data.get("batch-size") {
        ret.input_batchsize = Some(ocl_to_int32(v));
    }
    if let Some(v) = data.get("iter-count") {
        ret.target_iters = Some(ocl_to_int32(v));
    }
    if let Some(v) = data.get("grouping") {
        ret.target_grouping = Some(ocl_to_int32(v));
    }
    if let Some(v) = data.get("minchunk") {
        ret.min_chunk_size = ocl_parse_size(v);
    }
    Ok(ret)
}

/// `arg_parser.parseSize(v)` where v may be a number (already-parsed
/// int/size option) or a string (top-level string options and --opencl
/// k=v spec entries).
fn ocl_parse_size(v: &ArgVal) -> Option<f64> {
    match v {
        ArgVal::Num(n) => Some(n.floor().max(0.0)),
        ArgVal::Str(s) => parse_size(s),
        _ => None,
    }
}

/// `v|0` for a number or string.
fn ocl_to_int32(v: &ArgVal) -> i32 {
    match v {
        ArgVal::Num(n) => compat::to_int32(*n),
        ArgVal::Str(s) => compat::str_to_int32(s),
        _ => 0,
    }
}

/// Deferred half of `openclMap`: resolves device names against the live
/// platform list and applies the `openclDevicesSelected` dedup. Replicates
/// the JS loops including the falsy-platform-id outer-`break` quirk (a
/// match on platform 0 keeps scanning and a later same-named device on
/// another platform overwrites it).
///
/// JS order: the global `--opencl-*` spec is processed first (its device
/// is resolved/marked once), then each `--opencl` entry's own `device=`.
/// Entries that inherited the global device get its resolved ids without
/// a fresh lookup.
pub fn resolve_opencl_devices(
    global: &mut OclDeviceSpec,
    devs: &mut [OclDeviceSpec],
    platforms: &[OclPlatformInfo],
) -> Result<(), String> {
    let mut selected: std::collections::HashSet<String> = std::collections::HashSet::new();
    resolve_one(global, platforms, &mut selected)?;
    for spec in devs.iter_mut() {
        if spec.device_inherited {
            spec.platform = global.platform;
            spec.device = global.device;
            continue;
        }
        resolve_one(spec, platforms, &mut selected)?;
    }
    Ok(())
}

fn resolve_one(
    spec: &mut OclDeviceSpec,
    platforms: &[OclPlatformInfo],
    selected: &mut std::collections::HashSet<String>,
) -> Result<(), String> {
    let Some(raw) = spec.device_raw.clone() else {
        return Ok(());
    };
    if is_numeric_device_spec(&raw) {
        selected.insert(raw);
        return Ok(());
    }
    let target = raw.replace(',', "").to_lowercase();
    let mut found: Option<(i32, i32)> = None;
    'outer: for plat in platforms {
        for dev in &plat.devices {
            let selkey = format!("{}:{}", plat.id, dev.id);
            if dev.name.replace(',', "").to_lowercase() == target && !selected.contains(&selkey) {
                found = Some((plat.id, dev.id));
                selected.insert(selkey);
                break;
            }
        }
        // JS: `if(ret.platform) break` - platform id 0 is falsy, so a
        // match there does not stop the outer scan
        if found.map(|(p, _)| p != 0).unwrap_or(false) {
            break 'outer;
        }
    }
    match found {
        Some((p, d)) => {
            spec.platform = Some(p);
            spec.device = Some(d);
            Ok(())
        }
        None => Err(format!("Couldn't find OpenCL device \"{}\"", raw)),
    }
}

/// Port of the `--opencl k=v,k=v` spec parsing: `^(.+?)=(.+)$` - non-empty
/// key AND value required.
pub fn parse_opencl_list_entry(spec: &str) -> Result<HashMap<String, ArgVal>, String> {
    let mut m = HashMap::new();
    for s in spec.split(',') {
        match s.find('=') {
            Some(eq) if eq > 0 && eq + 1 < s.len() => {
                m.insert(
                    s[..eq].trim().to_string(),
                    ArgVal::Str(s[eq + 1..].trim().to_string()),
                );
            }
            _ => return Err("Invalid specification for `--opencl`".into()),
        }
    }
    Ok(m)
}

// ---------------------------------------------------------------------------
// full CLI parse + post-processing
// ---------------------------------------------------------------------------

/// `ppo` - the options object handed to PAR2Gen (mapped names preserved as
/// Rust fields).
#[derive(Debug, Default)]
pub struct ParParOpts {
    pub output_base: String,
    pub creator: String,
    pub recovery_slices_unit: String,
    // mapped opts (map: keys); Option = "key present"
    pub recovery_offset: Option<i32>,
    pub recovery_exponents: Option<Vec<String>>,
    pub comments: Option<Vec<String>>,
    pub critical_redundancy_scheme: Option<CriticalRedundancy>,
    pub min_critical_redundancy: Option<i32>,
    pub max_critical_redundancy: Option<i32>,
    pub display_name_format: Option<String>,
    pub display_name_base: Option<String>,
    pub unicode: Option<bool>,
    pub output_overwrite: Option<bool>,
    pub output_sync: Option<bool>,
    pub output_alt_naming_scheme: Option<bool>,
    pub output_size_scheme: Option<String>,
    pub output_file_count: Option<i32>,
    pub output_index: Option<bool>,
    pub memory_limit: Option<f64>,
    pub num_threads: Option<i32>,
    pub min_chunk_size: Option<f64>,
    pub seq_read_size: Option<f64>,
    pub chunk_read_threads: Option<i32>,
    pub seq_read_throttle: Option<Option<ThrottleSpec>>,
    pub chunk_read_throttle: Option<Option<ThrottleSpec>>,
    pub read_buffers: Option<i32>,
    pub read_hash_queue: Option<i32>,
    pub process_batch_size: Option<i32>,
    pub hash_batch_size: Option<i32>,
    pub rec_data_size: Option<i32>,
    pub gf_method: Option<String>,
    pub loop_tile_size: Option<f64>,
    pub cpu_min_chunk_size: Option<f64>,
    // computed
    /// JS can store `false` here (unparseable `--slice-size-multiple`);
    /// `None` represents that (falsy for every consumer).
    pub slice_size_multiple: Option<f64>,
    pub opencl_devices: Option<Vec<OclDeviceSpec>>,
}

/// `criticalRedundancyScheme`: the string 'pow2'/'none' stays verbatim;
/// anything else is parsed into a spec expression.
#[derive(Debug, Clone, PartialEq)]
pub enum CriticalRedundancy {
    Named(String),
    Expr(SpecExpr),
}

/// Early-exit paths handled before the required-options check in JS
/// (`process.exit(0)` after printing).
#[derive(Debug, Clone, PartialEq)]
pub enum EarlyExit {
    /// `--help` / `--help-full` (`full` when help-full was given; if both,
    /// full wins - `argv['help-full'] ? 'full':'short'`).
    Help { full: bool },
    /// `--version`
    Version,
    /// `--opencl-list[=filter]` - value is the resolved string ('gpu' when
    /// bare).
    OpenclList(String),
}

/// Resolved CLI configuration after all post-parse processing.
#[derive(Debug)]
pub struct CliConfig {
    /// Set when the invocation short-circuits (help/version/opencl-list);
    /// all other fields are then unset/defaults.
    pub early: Option<EarlyExit>,
    pub argv: ParsedArgs,
    pub ppo: ParParOpts,
    pub input_slice_def: Option<SpecTerm>,
    pub min_slice_size: Option<SpecTerm>,
    pub max_slice_size: Option<SpecTerm>,
    pub recovery_slices: Option<SpecExpr>,
    pub min_recovery_slices: Option<SpecExpr>,
    pub max_recovery_slices: Option<SpecExpr>,
    pub output_file_max_slices: Option<SpecExpr>,
    pub output_file_max_slices_rounding: Option<SpecRounding>,
    pub output_first_file_slices: Option<SpecExpr>,
    pub output_first_file_slices_rounding: Option<SpecRounding>,
    pub input_files: Vec<String>,
    /// input-file list sources: (path, true=LF separated / false=NUL separated)
    pub input_file_lists: Vec<(String, bool)>,
    pub input_file_enc: String,
    /// `recurse`: JS accepts bool or an integer depth (default 1 =
    /// "traverse folders explicitly specified" but no deeper).
    pub recurse: Recurse,
    pub skip_symlinks: bool,
    pub out: String,
    pub json: bool,
    pub quiet: bool,
    pub progress: String,
    pub help: bool,
    pub help_full: bool,
    pub version: bool,
    pub opencl_list: Option<String>,
    pub hash_method: Option<String>,
    pub md5_method: Option<String>,
    pub ascii_charset: String,
    /// The global `--opencl-*` spec pre-merge. Kept separately because JS
    /// resolves its device name first (the resolved ids are what `--opencl`
    /// entries inherit).
    pub opencl_global: OclDeviceSpec,
}

/// `recurse` tri-state: par2gen treats `true` as unlimited recursion,
/// `false` as none, and a number as a depth limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recurse {
    No,
    Depth(i32),
    All,
}

impl Recurse {
    fn from_val(v: &ArgVal) -> Recurse {
        match v {
            ArgVal::Bool(b) => {
                if *b {
                    Recurse::All
                } else {
                    Recurse::No
                }
            }
            ArgVal::Num(n) => Recurse::Depth(*n as i32),
            ArgVal::Null => Recurse::No,
            _ => Recurse::No,
        }
    }
}

pub const PARPAR_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn creator_string() -> String {
    format!(
        "ParPar v{} {} [https://github.com/animetosho/parpar]",
        PARPAR_VERSION,
        compat::node_arch()
    )
}

/// Port of bin/parpar.js from `arg_parser(...)` through `ppo` construction.
/// Input-file list *contents* are not read here (see resolve_input_files).
/// Help/version/opencl-list short-circuit BEFORE the required check, as in
/// JS (`EarlyExit`), so callers must handle `cfg.early` first.
pub fn parse_cli(argv: &[String]) -> Result<CliConfig, String> {
    let opts = opts_table();
    let mut argv_p = parse(argv, &opts)?;
    let mut a = argv_p.map.clone();

    let truthy_of =
        |m: &HashMap<String, ArgVal>, k: &str| m.get(k).map(|v| v.truthy()).unwrap_or(false);

    // early exits, in JS order: help/help-full -> version -> opencl-list
    if truthy_of(&a, "help") || truthy_of(&a, "help-full") {
        return Ok(CliConfig {
            early: Some(EarlyExit::Help {
                full: truthy_of(&a, "help-full"),
            }),
            argv: argv_p,
            ..CliConfig::empty()
        });
    }
    if truthy_of(&a, "version") {
        return Ok(CliConfig {
            early: Some(EarlyExit::Version),
            argv: argv_p,
            ..CliConfig::empty()
        });
    }
    if let Some(v) = a.get("opencl-list") {
        if v.truthy() {
            let filter = match v {
                ArgVal::Str(s) => s.clone(),
                _ => String::new(),
            };
            return Ok(CliConfig {
                early: Some(EarlyExit::OpenclList(filter)),
                argv: argv_p,
                ..CliConfig::empty()
            });
        }
    }

    // JS: `if(!argv.out || !argv['input-slices'])` - falsy, not absent
    if !truthy_of(&a, "out") || !truthy_of(&a, "input-slices") {
        return Err("Values for `out` and `input-slices` are required".into());
    }

    // alias for 'auto-slice-size'
    if a.get("auto-slice-size")
        .map(|v| v.truthy())
        .unwrap_or(false)
        && !a.contains_key("max-input-slices")
    {
        a.insert("max-input-slices".into(), ArgVal::Num(32768.0));
    }

    // JS mutates argv here: `if(!argv.progress)
    //   argv.progress = argv.quiet ? 'none' : 'stderr';`
    let quiet = a.get("quiet").map(|v| v.truthy()).unwrap_or(false);
    let json = a.get("json").map(|v| v.truthy()).unwrap_or(false);
    if !a.get("progress").map(|v| v.truthy()).unwrap_or(false) {
        a.insert(
            "progress".into(),
            ArgVal::Str(if quiet { "none" } else { "stderr" }.into()),
        );
    }
    let progress = match a.get("progress") {
        Some(ArgVal::Str(s)) => s.clone(),
        _ => String::new(),
    };

    // recovery-exponents conflicts
    if a.get("recovery-exponents")
        .map(|v| v.truthy())
        .unwrap_or(false)
    {
        for conflict in [
            "recovery-slices",
            "min-recovery-slices",
            "max-recovery-slices",
            "recovery-offset",
            "slice-dist",
            "slices-per-file",
            "slices-first-file",
            "recovery-files",
        ] {
            if a.get(conflict).map(|v| v.truthy()).unwrap_or(false) {
                return Err(format!(
                    "`--recovery-exponents` cannot be used with `--{}`",
                    conflict
                ));
            }
        }
        if !a.get("noindex").map(|v| v.truthy()).unwrap_or(false) {
            return Err("`--recovery-exponents` cannot be used with `--noindex`".into());
        }
    }

    // JS: `if(argv['filepath-base'] && !argv['filepath-format'])` - truthy
    let (fp_base, fp_fmt) = (
        truthy_of(&a, "filepath-base"),
        truthy_of(&a, "filepath-format"),
    );
    if fp_base && !fp_fmt {
        a.insert("filepath-format".into(), ArgVal::Str("path".into()));
    }

    // ppo construction
    let mut ppo = ParParOpts {
        output_base: a
            .get("out")
            .map(|v| match v {
                ArgVal::Str(s) => s.clone(),
                _ => String::new(),
            })
            .unwrap_or_default(),
        creator: creator_string(),
        recovery_slices_unit: "slices".into(),
        ..Default::default()
    };
    if ppo.output_base.to_lowercase().ends_with(".par2") {
        ppo.output_base.truncate(ppo.output_base.len() - 5);
    }

    // slice-size-multiple: parseSize can yield JS `false` (-> None) or 0;
    // both falsy for consumers
    let mut slice_size_multiple = Some(4.0f64);
    if let Some(ArgVal::Str(ssm)) = a.get("slice-size-multiple") {
        slice_size_multiple = parse_size(ssm);
        if let Some(v) = slice_size_multiple {
            if (v as i64) % 4 != 0 && v > 100.0 && ends_with_size_unit(ssm) {
                // invalid multiple, but size may have been specified without
                // enough precision
                slice_size_multiple = Some(compat::js_math_round(v / 4.0).max(4.0) * 4.0);
            }
        }
    }
    ppo.slice_size_multiple = slice_size_multiple;
    let mult = slice_size_multiple.unwrap_or(0.0);

    // spec expressions for the five (optionally six) options
    let mut recovery_slices = None;
    let mut min_recovery_slices = None;
    let mut max_recovery_slices = None;
    let mut output_file_max_slices = None;
    let mut output_file_max_slices_rounding = None;
    let mut output_first_file_slices = None;
    let mut output_first_file_slices_rounding = None;
    let mut critical_redundancy: Option<CriticalRedundancy> = None;

    for (key, dst, rounding) in [
        ("recovery-slices", &mut recovery_slices, &mut None),
        ("min-recovery-slices", &mut min_recovery_slices, &mut None),
        ("max-recovery-slices", &mut max_recovery_slices, &mut None),
        (
            "slices-per-file",
            &mut output_file_max_slices,
            &mut output_file_max_slices_rounding,
        ),
        (
            "slices-first-file",
            &mut output_first_file_slices,
            &mut output_first_file_slices_rounding,
        ),
    ] {
        if let Some(ArgVal::Str(val)) = a.get(key) {
            let (expr, r) = parse_cli_spec(key, val, mult)?;
            *dst = Some(expr);
            *rounding = r;
        }
    }
    // packet-redundancy: 'pow2'/'none' stay strings; else expr
    if let Some(ArgVal::Str(val)) = a.get("packet-redundancy") {
        if val == "pow2" || val == "none" {
            critical_redundancy = Some(CriticalRedundancy::Named(val.clone()));
        } else {
            let (expr, _) = parse_cli_spec("packet-redundancy", val, mult)?;
            critical_redundancy = Some(CriticalRedundancy::Expr(expr));
        }
    }

    // input-slices (required, checked above)
    let input_slice_def = match a.get("input-slices") {
        Some(v) => parse_size_or_num("input-slices", SpecInput::from(v), mult)?,
        _ => unreachable!(),
    };
    if input_slice_def.unit == SpecUnit::Count && input_slice_def.value > 32768.0 {
        return Err(format!(
            "Invalid number (>32768) of input slices requested. Perhaps you meant `--input-slices={}b` instead?",
            compat::fmt_num(input_slice_def.value)
        ));
    }

    let mut min_slice_size = None;
    let mut max_slice_size = None;
    for (e, dst) in [
        ("min-input-slices", &mut min_slice_size),
        ("max-input-slices", &mut max_slice_size),
    ] {
        if let Some(v) = a.get(e) {
            *dst = Some(parse_size_or_num(e, SpecInput::from(v), mult)?);
        }
    }
    if input_slice_def.unit == SpecUnit::Bytes
        && !a.contains_key("slice-size-multiple")
        && (!a.contains_key("min-input-slices")
            || (min_slice_size
                .as_ref()
                .map(|m| m.unit == SpecUnit::Bytes && m.value == input_slice_def.value)
                .unwrap_or(false)))
    {
        ppo.slice_size_multiple = Some(input_slice_def.value);
    }

    // mapped fields (the `for k in opts: if opts[k].map && k in argv` loop).
    // `int`-typed options store i32; `size`/`size0`/`time` store f64.
    macro_rules! map_i32 {
        ($k:expr, $field:ident) => {
            if let Some(ArgVal::Num(n)) = a.get($k) {
                ppo.$field = Some(*n as i32);
            }
        };
    }
    map_i32!("recovery-offset", recovery_offset);
    map_i32!("min-packet-redundancy", min_critical_redundancy);
    map_i32!("max-packet-redundancy", max_critical_redundancy);
    map_i32!("recovery-files", output_file_count);
    map_i32!("chunk-read-threads", chunk_read_threads);
    map_i32!("read-buffers", read_buffers);
    map_i32!("read-hash-queue", read_hash_queue);
    map_i32!("proc-batch-size", process_batch_size);
    map_i32!("md5-batch-size", hash_batch_size);
    map_i32!("recovery-buffers", rec_data_size);

    if let Some(v) = a.get("recovery-exponents") {
        if let ArgVal::Arr(l) = v {
            ppo.recovery_exponents = Some(l.clone());
        }
    }
    if let Some(ArgVal::Arr(c)) = a.get("comment") {
        ppo.comments = Some(c.clone());
    }
    ppo.critical_redundancy_scheme = critical_redundancy;
    if let Some(ArgVal::Str(s)) = a.get("filepath-format") {
        ppo.display_name_format = Some(s.clone());
    }
    if let Some(ArgVal::Str(s)) = a.get("filepath-base") {
        ppo.display_name_base = Some(s.clone());
    }
    // unicode/threads use JS `default: null` - a present-but-null value is
    // "auto", not false/0, so map Null to None rather than truthy()/0
    ppo.unicode = a.get("unicode").and_then(|v| match v {
        ArgVal::Null => None,
        other => Some(other.truthy()),
    });
    ppo.output_overwrite = a.get("overwrite").map(|v| v.truthy());
    ppo.output_sync = a.get("write-sync").map(|v| v.truthy());
    ppo.output_alt_naming_scheme = a.get("std-naming").map(|v| v.truthy());
    if let Some(ArgVal::Str(s)) = a.get("slice-dist") {
        ppo.output_size_scheme = Some(s.clone());
    }
    ppo.output_index = a.get("noindex").map(|v| v.truthy());
    ppo.num_threads = a.get("threads").and_then(|v| match v {
        ArgVal::Num(n) => Some(*n as i32),
        _ => None,
    });
    if let Some(t) = a.get("seq-read-throttle") {
        if let ArgVal::Throttle(ts) = t {
            ppo.seq_read_throttle = Some(ts.clone());
        }
    }
    if let Some(t) = a.get("chunk-read-throttle") {
        if let ArgVal::Throttle(ts) = t {
            ppo.chunk_read_throttle = Some(ts.clone());
        }
    }
    if let Some(ArgVal::Str(s)) = a.get("method") {
        ppo.gf_method = Some(s.clone());
    }

    // numeric (f64) mapped fields
    macro_rules! map_f64 {
        ($k:expr, $field:ident) => {
            if let Some(ArgVal::Num(n)) = a.get($k) {
                ppo.$field = Some(*n);
            }
        };
    }
    map_f64!("memory", memory_limit);
    map_f64!("min-chunk-size", min_chunk_size);
    map_f64!("seq-read-size", seq_read_size);
    map_f64!("loop-tile-size", loop_tile_size);
    map_f64!("cpu-minchunk", cpu_min_chunk_size);

    // opencl opts - every `opencl-*` argv entry (JS `for k in argv`,
    // substring(0,7)=='opencl-'); values may be numbers (int/size0 opts)
    // as well as strings
    let mut opencl_opts_raw: HashMap<String, ArgVal> = HashMap::new();
    for (k, v) in a.iter() {
        if let Some(rest) = k.strip_prefix("opencl-") {
            opencl_opts_raw.insert(rest.to_string(), v.clone());
        }
    }
    let opencl_opts = opencl_map(&opencl_opts_raw)?;
    if let Some(ArgVal::Arr(list)) = a.get("opencl") {
        let mut devs = Vec::new();
        for spec in list {
            let m = parse_opencl_list_entry(spec)?;
            let e = opencl_map(&m)?;
            devs.push(opencl_opts.extend(&e));
        }
        ppo.opencl_devices = Some(devs);
    } else if opencl_opts.ratio.is_some() {
        ppo.opencl_devices = Some(vec![opencl_opts.clone()]);
    }

    // input files: positional + list-file sources
    let mut input_file_lists: Vec<(String, bool)> = Vec::new();
    if let Some(ArgVal::Arr(v)) = a.get("input-file") {
        input_file_lists.extend(v.iter().map(|f| (f.clone(), true)));
    }
    if let Some(ArgVal::Arr(v)) = a.get("input-file0") {
        input_file_lists.extend(v.iter().map(|f| (f.clone(), false)));
    }

    let input_file_enc = a
        .get("input-file-enc")
        .and_then(|v| v.str_opt())
        .unwrap_or_else(|| "utf8".into());
    let recurse = a
        .get("recurse")
        .map(Recurse::from_val)
        .unwrap_or(Recurse::No);
    let skip_symlinks = a.get("skip-symlinks").map(|v| v.truthy()).unwrap_or(false);
    let out = a.get("out").and_then(|v| v.str_opt()).unwrap_or_default();
    let help = a.get("help").map(|v| v.truthy()).unwrap_or(false);
    let help_full = a.get("help-full").map(|v| v.truthy()).unwrap_or(false);
    let version = a.get("version").map(|v| v.truthy()).unwrap_or(false);
    let opencl_list = a.get("opencl-list").and_then(|v| v.str_opt());
    let hash_method = a.get("hash-method").and_then(|v| v.str_opt());
    let md5_method = a.get("md5-method").and_then(|v| v.str_opt());
    let ascii_charset = a
        .get("ascii-charset")
        .and_then(|v| v.str_opt())
        .unwrap_or_else(|| "utf-8".into());
    let input_files = argv_p.positional.clone();

    argv_p.map = a;
    Ok(CliConfig {
        early: None,
        input_slice_def: Some(input_slice_def),
        min_slice_size,
        max_slice_size,
        recovery_slices,
        min_recovery_slices,
        max_recovery_slices,
        output_file_max_slices,
        output_file_max_slices_rounding,
        output_first_file_slices,
        output_first_file_slices_rounding,
        input_files,
        input_file_lists,
        input_file_enc,
        recurse,
        skip_symlinks,
        out,
        json,
        quiet,
        progress,
        help,
        help_full,
        version,
        opencl_list,
        hash_method,
        md5_method,
        ascii_charset,
        opencl_global: opencl_opts,
        argv: argv_p,
        ppo,
    })
}

impl CliConfig {
    /// Empty config used for the early-exit paths (JS never builds ppo on
    /// those).
    fn empty() -> CliConfig {
        CliConfig {
            early: None,
            argv: ParsedArgs::default(),
            ppo: ParParOpts::default(),
            input_slice_def: None,
            min_slice_size: None,
            max_slice_size: None,
            recovery_slices: None,
            min_recovery_slices: None,
            max_recovery_slices: None,
            output_file_max_slices: None,
            output_file_max_slices_rounding: None,
            output_first_file_slices: None,
            output_first_file_slices_rounding: None,
            input_files: Vec::new(),
            input_file_lists: Vec::new(),
            input_file_enc: "utf8".into(),
            recurse: Recurse::No,
            skip_symlinks: false,
            out: String::new(),
            json: false,
            quiet: false,
            progress: "stderr".into(),
            help: false,
            help_full: false,
            version: false,
            opencl_list: None,
            hash_method: None,
            md5_method: None,
            ascii_charset: "utf-8".into(),
            opencl_global: OclDeviceSpec::default(),
        }
    }

    /// Read the `--input-file`/`--input-file0` list sources and append their
    /// contents to `input_files` (JS does this via async.map preserving
    /// source order: all `--input-file` entries, then all `--input-file0`).
    /// `read_stdin` is invoked at most once for `-` sources (a second `-`
    /// is a JS error); `fd://N` and `proc://cmd` sources are also handled.
    /// Afterwards enforces `At least one input file must be supplied`.
    pub fn resolve_input_files(
        &mut self,
        mut read_stdin: impl FnMut() -> Result<Vec<u8>, String>,
    ) -> Result<(), String> {
        let enc = self.input_file_enc.clone();
        let mut stdin_used = false;
        for (src, newline_sep) in self.input_file_lists.clone() {
            let data: Vec<u8> = if src == "-" {
                if stdin_used {
                    return Err("stdin was specified as input for multiple sources".into());
                }
                stdin_used = true;
                read_stdin()?
            } else if src.len() > 5
                && src.as_bytes()[..5].eq_ignore_ascii_case(b"fd://")
                && src[5..].chars().all(|c| c.is_ascii_digit())
            {
                // JS /^fd:\/\/\d+$/i
                let fd = compat::str_to_int32(&src[5..]);
                read_fd(fd)?
            } else if src.len() >= 7 && src.as_bytes()[..7].eq_ignore_ascii_case(b"proc://") {
                exec_out(&src[7..])?
            } else {
                std::fs::read(&src).map_err(|e| fs_err_msg(&e, &src))?
            };
            let text = decode_list(&enc, &data)?;
            if newline_sep {
                // .replace(/\r/g,'').split('\n').filter(l => l !== '')
                for line in text.replace('\r', "").split('\n') {
                    if !line.is_empty() {
                        self.input_files.push(line.to_string());
                    }
                }
            } else {
                // .replace(/\0$/,'').split('\0') - a single trailing NUL is
                // stripped; interior NULs can produce empty entries
                let t = text.strip_suffix('\0').unwrap_or(&text);
                for line in t.split('\0') {
                    self.input_files.push(line.to_string());
                }
            }
        }
        if self.input_files.is_empty() {
            return Err("At least one input file must be supplied".into());
        }
        Ok(())
    }
}

/// Node system Error rendered the way `console.error(err)` shows it:
/// `[Error: <msg>] {\n  errno: <n>,\n  code: '<code>',\n  syscall: '<s>'(,
///   path: '<p>')?\n}`. `path` appears only when the failing syscall took
/// one (open/stat), and is also embedded in the message.
use crate::compat::{errno_info, node_fs_error};

/// `fs.readFile` error mapping (readFile opens then reads; directory reads
/// fail at the read step, carrying no path).
fn fs_err_msg(e: &std::io::Error, path: &str) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => node_fs_error(
            -2,
            "ENOENT",
            "no such file or directory",
            "open",
            Some(path),
        ),
        std::io::ErrorKind::PermissionDenied => {
            node_fs_error(-13, "EACCES", "permission denied", "open", Some(path))
        }
        _ if e.raw_os_error() == Some(21) => node_fs_error(
            -21,
            "EISDIR",
            "illegal operation on a directory",
            "read",
            None,
        ),
        _ => {
            let errno = e.raw_os_error().unwrap_or(5);
            let (code, desc) = errno_info(errno);
            node_fs_error(-errno, code, desc, "open", Some(path))
        }
    }
}

/// Read an `fd://N` source (JS `fs.createReadStream(null, {fd})`). The fd
/// is dup'd before reading: taking ownership of an arbitrary fd is unsafe
/// (a bad fd aborts on Rust IO-safety checks) and Node only borrows it.
#[cfg(unix)]
fn read_fd(fd: i32) -> Result<Vec<u8>, String> {
    use std::io::Read;
    use std::os::unix::io::FromRawFd;
    let dup = unsafe { libc::fcntl(fd, libc::F_DUPFD, 0) };
    if dup < 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(9);
        let (code, desc) = errno_info(errno);
        // Node's createReadStream reports the failed autoClose() here
        return Err(node_fs_error(-errno, code, desc, "close", None));
    }
    let mut f = unsafe { std::fs::File::from_raw_fd(dup) };
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| {
        let errno = e.raw_os_error().unwrap_or(5);
        let (code, desc) = errno_info(errno);
        node_fs_error(-errno, code, desc, "read", None)
    })?;
    Ok(buf)
}

#[cfg(not(unix))]
fn read_fd(fd: i32) -> Result<Vec<u8>, String> {
    let _ = fd;
    Err("fd:// input lists are not supported on this platform".into())
}

/// `child_process.exec` for `proc://cmd` - platform shell, 32 MiB stdout cap.
fn exec_out(cmd: &str) -> Result<Vec<u8>, String> {
    let out = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/d", "/s", "/c", cmd])
            .output()
    } else {
        std::process::Command::new("sh").arg("-c").arg(cmd).output()
    };
    match out {
        Ok(o) => {
            // Node caps exec stdout at maxBuffer (32 MiB here)
            if o.stdout.len() > 32 * 1048576 {
                return Err("stdout maxBuffer length exceeded".into());
            }
            if !o.status.success() {
                // Node reports "Command failed: <cmd>\n<stderr>"
                return Err(format!(
                    "Command failed: {}\n{}",
                    cmd,
                    String::from_utf8_lossy(&o.stderr).trim_end()
                ));
            }
            Ok(o.stdout)
        }
        Err(e) => Err(format!("{}", e)),
    }
}

/// `Buffer#toString(enc)` for input lists. Whole-buffer decode - note that
/// JS decodes per stream *chunk* for '-'/fd:// sources, so a multi-byte
/// char split across a 64 KiB chunk boundary differs (edge case).
fn decode_list(enc: &str, data: &[u8]) -> Result<String, String> {
    let e = enc.to_lowercase();
    let s = match e.as_str() {
        "utf8" | "utf-8" => String::from_utf8_lossy(data).into_owned(),
        "latin1" | "binary" => data.iter().map(|&b| b as char).collect(),
        "ascii" => data.iter().map(|&b| (b & 0x7f) as char).collect(),
        "ucs2" | "ucs-2" | "utf16le" | "utf-16le" => {
            let units: Vec<u16> = data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            String::from_utf16_lossy(&units)
        }
        "hex" => data.iter().map(|b| format!("{:02x}", b)).collect(),
        "base64" | "base64url" => base64_encode(data, e == "base64url"),
        _ => return Err(format!("Unknown encoding: {}", enc)),
    };
    Ok(s)
}

fn base64_encode(data: &[u8], url: bool) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        for i in 0..4 {
            let idx = (n >> (18 - 6 * i)) & 63;
            let pad = c.len() == 1 && i >= 2 || c.len() == 2 && i == 3;
            let ch = if pad { '=' } else { A[idx as usize] as char };
            out.push(match (url, ch) {
                (true, '+') => '-',
                (true, '/') => '_',
                (_, c) => c,
            });
        }
    }
    out
}

fn ends_with_size_unit(s: &str) -> bool {
    s.chars()
        .last()
        .map(|c| matches!(c.to_ascii_lowercase(), 'k' | 'm' | 'g' | 't' | 'p' | 'e'))
        .unwrap_or(false)
}

impl ArgVal {
    fn str_opt(&self) -> Option<String> {
        match self {
            ArgVal::Str(s) => Some(s.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn size_or_num_units() {
        let t = parse_size_or_num("input-slices", SpecInput::Str("100"), 4.0).unwrap();
        assert_eq!(t.unit, SpecUnit::Count);
        assert_eq!(t.value, 100.0);
        let t = parse_size_or_num("input-slices", SpecInput::Str("100k"), 4.0).unwrap();
        assert_eq!(t.unit, SpecUnit::Bytes);
        assert_eq!(t.value, 102400.0);
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("50%"), 0.0).unwrap();
        assert_eq!(t.unit, SpecUnit::Ratio);
        assert_eq!(t.value, 0.5);
        // % not allowed for non-recovery args
        assert!(parse_size_or_num("input-slices", SpecInput::Str("50%"), 0.0).is_err());
        // largest/smallest/power/log/ilog; bare 'l*2' is invalid (JS regex requires a leading number)
        assert!(parse_size_or_num("recovery-slices", SpecInput::Str("l*2"), 0.0).is_err());
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("1l*2"), 0.0).unwrap();
        assert_eq!(t.unit, SpecUnit::LargestFiles);
        assert_eq!(t.value, 1.0);
        assert_eq!(t.scale, 2.0);
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("2l*3"), 0.0).unwrap();
        assert_eq!(t.unit, SpecUnit::LargestFiles);
        assert_eq!(t.value, 2.0);
        assert_eq!(t.scale, 3.0);
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("5s"), 0.0).unwrap();
        assert_eq!(t.unit, SpecUnit::SmallestFiles);
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("3w"), 0.0).unwrap();
        assert_eq!(t.unit, SpecUnit::Power);
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("2o"), 0.0).unwrap();
        assert_eq!(t.unit, SpecUnit::Log);
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("7n"), 0.0).unwrap();
        assert_eq!(t.unit, SpecUnit::ILog);
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("4l/2"), 0.0).unwrap();
        assert_eq!(t.scale, 0.5);
        // negative count ok for recovery args, rejected for others
        let t = parse_size_or_num("recovery-slices", SpecInput::Str("-5"), 0.0).unwrap();
        assert_eq!(t.value, -5.0);
        assert!(parse_size_or_num("input-slices", SpecInput::Str("-5"), 0.0).is_err());
        // invalid
        assert!(parse_size_or_num("input-slices", SpecInput::Str("abc"), 0.0).is_err());
        assert!(parse_size_or_num("input-slices", SpecInput::Str("1.5"), 0.0).is_err());
        assert!(parse_size_or_num("input-slices", SpecInput::Str(""), 0.0).is_err());
    }

    #[test]
    fn spec_expr_chains() {
        // '+-' chains produce multiple terms
        let e =
            parse_spec_expr("recovery-slices", "100+20", SpecInput::Str("100+20"), 0.0).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[1].value, 20.0);
        let e =
            parse_spec_expr("recovery-slices", "100-20", SpecInput::Str("100-20"), 0.0).unwrap();
        assert_eq!(e[1].value, -20.0);
        // leading negative: '-5' -> 0-5
        let e = parse_spec_expr("recovery-slices", "-5", SpecInput::Str("-5"), 0.0).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].value, 0.0);
        assert_eq!(e[1].value, -5.0);
        // '++'/'--' collapse to '+'
        let e = parse_spec_expr("recovery-slices", "10++5", SpecInput::Str("10++5"), 0.0).unwrap();
        assert_eq!(e[1].value, 5.0);
        let e = parse_spec_expr("recovery-slices", "10--5", SpecInput::Str("10--5"), 0.0).unwrap();
        assert_eq!(e[1].value, 5.0);
        // '+-'/'-+' collapse to '-'
        let e = parse_spec_expr("recovery-slices", "10+-5", SpecInput::Str("10+-5"), 0.0).unwrap();
        assert_eq!(e[1].value, -5.0);
        let e = parse_spec_expr("recovery-slices", "10-+5", SpecInput::Str("10-+5"), 0.0).unwrap();
        assert_eq!(e[1].value, -5.0);
        // mixed units
        let e =
            parse_spec_expr("recovery-slices", "1l*2+5%", SpecInput::Str("1l*2+5%"), 0.0).unwrap();
        assert_eq!(e[0].unit, SpecUnit::LargestFiles);
        assert_eq!(e[1].unit, SpecUnit::Ratio);
    }

    #[test]
    fn rounding_prefix() {
        let (e, r) = parse_cli_spec("slices-per-file", "<10", 0.0).unwrap();
        assert_eq!(r, Some(SpecRounding::Floor));
        assert_eq!(e[0].value, 10.0);
        let (_, r) = parse_cli_spec("slices-first-file", ">5", 0.0).unwrap();
        assert_eq!(r, Some(SpecRounding::Ceil));
        // non-slices opt doesn't take the prefix
        let e = parse_cli_spec("recovery-slices", "<10", 0.0);
        assert!(e.is_err());
    }

    #[test]
    fn bytes_multiple_snap() {
        // >25*multiple and within 5% of the unit snaps to the multiple
        let t = parse_size_or_num("input-slices", SpecInput::Str("101k"), 4.0).unwrap();
        assert_eq!(t.value, 103424.0); // not snapped (small value)
        let t = parse_size_or_num("input-slices", SpecInput::Str("12.5m"), 4.0).unwrap();
        assert_eq!(t.value, (12.5f64 * 1024.0 * 1024.0).floor());
    }

    #[test]
    fn opencl_parsing() {
        let mut m = HashMap::new();
        m.insert("process".into(), ArgVal::Str("50%".to_string()));
        m.insert("device".into(), ArgVal::Str("0:1".to_string()));
        m.insert("memory".into(), ArgVal::Str("2G".to_string()));
        m.insert("batch-size".into(), ArgVal::Str("4".to_string()));
        let s = opencl_map(&m).unwrap();
        assert_eq!(s.ratio, Some(0.5));
        assert_eq!(s.platform, Some(0));
        assert_eq!(s.device, Some(1));
        assert_eq!(s.memory_limit, Some(2147483648.0));
        assert_eq!(s.input_batchsize, Some(4));

        let mut m = HashMap::new();
        m.insert("device".into(), ArgVal::Str(":".to_string()));
        let s = opencl_map(&m).unwrap();
        assert_eq!(s.platform, None);
        assert_eq!(s.device, None);
        assert_eq!(s.device_raw, Some(":".into()));

        let mut m = HashMap::new();
        m.insert("device".into(), ArgVal::Str("GeForce RTX 2080".to_string()));
        let s = opencl_map(&m).unwrap();
        assert_eq!(s.device_raw, Some("GeForce RTX 2080".into()));

        let e = parse_opencl_list_entry("device=0:1,memory=1G");
        assert!(e.is_ok());
        let e = parse_opencl_list_entry("nokv");
        assert!(e.is_err());
    }

    #[test]
    fn full_cli_parse() {
        let cfg = parse_cli(&argv(&[
            "-o", "out.par2", "-s", "10k", "-r", "20%", "file1", "file2",
        ]))
        .unwrap();
        assert_eq!(cfg.out, "out.par2");
        assert_eq!(cfg.ppo.output_base, "out"); // .par2 stripped
        assert_eq!(cfg.input_files, vec!["file1", "file2"]);
        assert_eq!(cfg.input_slice_def.as_ref().unwrap().unit, SpecUnit::Bytes);
        assert_eq!(cfg.input_slice_def.as_ref().unwrap().value, 10240.0);
        assert_eq!(
            cfg.recovery_slices.as_ref().unwrap()[0].unit,
            SpecUnit::Ratio
        );
        assert_eq!(cfg.progress, "stderr");
        assert_eq!(cfg.ppo.creator, creator_string());
        // sliceSizeMultiple follows input-slices when s-s-m absent
        assert_eq!(cfg.ppo.slice_size_multiple, Some(10240.0));
    }

    #[test]
    fn required_and_conflicts() {
        let e = parse_cli(&argv(&["-o", "x"])).unwrap_err();
        assert_eq!(e, "Values for `out` and `input-slices` are required");
        let e = parse_cli(&argv(&[
            "-o",
            "x",
            "-s",
            "10",
            "--recovery-exponents",
            "1,2",
            "--recovery-slices",
            "10",
        ]))
        .unwrap_err();
        assert_eq!(
            e,
            "`--recovery-exponents` cannot be used with `--recovery-slices`"
        );
        // exponents + explicit --noindex
        let e = parse_cli(&argv(&[
            "-o",
            "x",
            "-s",
            "10",
            "--recovery-exponents",
            "1,2",
            "--noindex",
        ]))
        .unwrap_err();
        assert_eq!(e, "`--recovery-exponents` cannot be used with `--noindex`");
        // exponents alone OK
        let cfg = parse_cli(&argv(&[
            "-o",
            "x",
            "-s",
            "10",
            "--recovery-exponents",
            "1,2,3",
        ]))
        .unwrap();
        assert_eq!(
            cfg.ppo.recovery_exponents,
            Some(vec!["1".into(), "2".into(), "3".into()])
        );
    }

    #[test]
    fn auto_slice_size_alias() {
        let cfg = parse_cli(&argv(&["-o", "x", "-s", "10", "-S"])).unwrap();
        assert_eq!(
            cfg.argv.get("max-input-slices"),
            Some(&ArgVal::Num(32768.0))
        );
    }

    #[test]
    fn input_slices_over_32768() {
        let e = parse_cli(&argv(&["-o", "x", "-s", "40000"])).unwrap_err();
        assert!(e.contains(">32768"));
        assert!(e.contains("--input-slices=40000b"));
    }

    #[test]
    fn filepath_base_implies_format() {
        let cfg = parse_cli(&argv(&["-o", "x", "-s", "10", "-B", "base/dir"])).unwrap();
        assert_eq!(cfg.ppo.display_name_format.as_deref(), Some("path"));
    }
}

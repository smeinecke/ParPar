//! Port of lib/arg_parser.js: the generic option-spec parser.
//!
//! Value model: JS uses dynamic types; the parser's observable states are
//! bools, JS numbers (f64), strings, arrays (JS list and array both store
//! `[]`), maps (insertion-ordered, last-wins string maps with optional
//! values) and the `null`/`false` sentinels multi-value options use before
//! the defaults pass resolves them. Error strings are byte-identical to
//! the JS.

use crate::compat;
use std::collections::HashMap;

/// JS `parseSize`. Returns `Option<f64>`: `None` where JS returns `false`,
/// `Some(n)` for `Math.floor` results (0 included - callers distinguish
/// "falsy 0" for the `size` type).
pub fn parse_size(s: &str) -> Option<f64> {
    if compat::is_digits(s) {
        return Some(compat::js_number(s).floor().max(0.0));
    }
    let up = s.to_uppercase();
    let b = up.as_bytes();
    if b.len() < 2 {
        return None;
    }
    let unit = b[b.len() - 1];
    let digits = &up[..b.len() - 1];
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit() || c == b'.') {
        return None;
    }
    let mult = match unit {
        b'E' => 1024f64.powi(6),
        b'P' => 1024f64.powi(5),
        b'T' => 1024f64.powi(4),
        b'G' => 1024f64.powi(3),
        b'M' => 1024f64.powi(2),
        b'K' => 1024f64,
        b'B' => 1.0,
        _ => return None,
    };
    let num = compat::js_number(digits) * mult;
    if num.is_nan() {
        return None;
    }
    Some(num.floor())
}

/// JS `parseTime` - same shape, result in milliseconds.
pub fn parse_time(s: &str) -> Option<f64> {
    if compat::is_digits(s) {
        return Some((compat::js_number(s) * 1000.0).floor().max(0.0));
    }
    let lo = s.to_lowercase();
    let (digits, unit_len) = if lo.ends_with("ms") {
        (&lo[..lo.len() - 2], 2usize)
    } else {
        let b = lo.as_bytes();
        match b.last() {
            Some(b'm') | Some(b'h') | Some(b'd') | Some(b'w') | Some(b's') => {
                (&lo[..b.len() - 1], 1usize)
            }
            _ => return None,
        }
    };
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit() || c == b'.') {
        return None;
    }
    let unit = &lo[lo.len() - unit_len..];
    // JS quirk: the regex accepts 'ms' but the switch has no case for it, so
    // the number passes through unscaled (i.e. treated as already-ms)
    let mult = match unit {
        "w" => 7.0 * 24.0 * 60.0 * 60.0 * 1000.0,
        "d" => 24.0 * 60.0 * 60.0 * 1000.0,
        "h" => 60.0 * 60.0 * 1000.0,
        "m" => 60.0 * 1000.0,
        "s" => 1000.0,
        "ms" => 1.0,
        _ => return None,
    };
    let num = compat::js_number(digits) * mult;
    if num.is_nan() {
        return None;
    }
    Some(num.floor())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptType {
    Bool,
    Int,
    NInt, // '-int' - signed integer
    Size,
    Size0,
    Time,
    Enum(&'static [&'static str]),
    Str,
    List,
    Array,
    Map,
    Map2,
}

impl OptType {
    fn is_multiple(self) -> bool {
        matches!(
            self,
            OptType::List | OptType::Array | OptType::Map | OptType::Map2
        )
    }
}

/// Post-set transform attached to an option, mirroring the JS `fn` hook.
/// Scalar options get the value; multi options get the whole collection at
/// the end (the applyFn pass).
#[derive(Debug, Clone)]
pub enum OptFn {
    Invert,
    AsciiCharset,
    Throttle(&'static str),
    Identity,
}

impl OptFn {
    pub fn apply(&self, _key: &str, v: ArgVal) -> Result<ArgVal, String> {
        match self {
            OptFn::Invert => Ok(ArgVal::Bool(!v.truthy())),
            OptFn::Identity => Ok(v),
            OptFn::AsciiCharset => {
                if let ArgVal::Str(ref s) = v {
                    if !compat::is_node_encoding(s) {
                        return Err("Unknown encoding for `ascii-charset`".into());
                    }
                }
                Ok(v)
            }
            OptFn::Throttle(prop) => match v {
                ArgVal::Str(ref s) => match parse_throttle(prop, s) {
                    Ok(t) => Ok(ArgVal::Throttle(t)),
                    Err(e) => Err(e),
                },
                _ => Ok(v),
            },
        }
    }
}

/// A parsed `--*-read-throttle` spec.
#[derive(Debug, Clone, PartialEq)]
pub struct ThrottleSpec {
    pub mode: ThrottleMode,
    /// `rate` mode only: bytes per interval (JS `size`).
    pub size: f64,
    /// `reqrate` mode only: the raw count string (JS `count` - kept verbatim,
    /// JS never converts it here).
    pub count: String,
    /// Milliseconds per interval (`delay` mode only uses time).
    pub time: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleMode {
    Rate,
    ReqRate,
    Delay,
}

/// Port of `throttleParseFunc` in bin/parpar.js.
/// Format: `[<num><unit>/]<num><time-unit>` - a `r/R` unit selects request
/// rate, size units select byte rate, absent prefix selects delay mode.
/// A falsy/'0' value maps to "no throttle" (JS returns null).
pub fn parse_throttle(prop: &str, v: &str) -> Result<Option<ThrottleSpec>, String> {
    let err = || {
        Err(format!(
            "Invalid format for `--{}-read-throttle`: {}",
            prop, v
        ))
    };
    if v.is_empty() || v == "0" {
        return Ok(None);
    }
    let (prefix, rest) = match v.find('/') {
        Some(i) => (Some(&v[..i]), &v[i + 1..]),
        None => (None, v),
    };
    // rest = `[0-9.]*(m?s|[mhdw])` (case-insensitive); JS defaults an empty
    // numeric group to '1'
    let (time_num, time_unit) = split_num_unit(rest);
    if !valid_time_unit(time_unit) {
        return err();
    }
    let time_num = if time_num.is_empty() { "1" } else { time_num };
    if !digits_dots(time_num) {
        return err();
    }
    let time = match parse_time(&format!("{}{}", time_num, time_unit)) {
        Some(t) => t,
        None => return err(),
    };
    match prefix {
        None => Ok(Some(ThrottleSpec {
            mode: ThrottleMode::Delay,
            size: 0.0,
            count: String::new(),
            time,
        })),
        Some(p) => {
            // prefix = `[0-9.]*[bkmgtper]` (case-insensitive); the numeric
            // group may be empty (JS defaults it to '1')
            let (n, u) = split_num_unit(p);
            if u.len() != 1 {
                return err();
            }
            let uc = u.as_bytes()[0].to_ascii_lowercase();
            if !matches!(uc, b'b' | b'k' | b'm' | b'g' | b't' | b'p' | b'e' | b'r') {
                return err();
            }
            let n = if n.is_empty() { "1" } else { n };
            match uc {
                b'r' => Ok(Some(ThrottleSpec {
                    mode: ThrottleMode::ReqRate,
                    size: 0.0,
                    count: n.to_string(),
                    time,
                })),
                _ => {
                    let size = match parse_size(&format!("{}{}", n, u)) {
                        Some(s) => s,
                        None => return err(),
                    };
                    Ok(Some(ThrottleSpec {
                        mode: ThrottleMode::Rate,
                        size,
                        count: String::new(),
                        time,
                    }))
                }
            }
        }
    }
}

fn valid_time_unit(u: &str) -> bool {
    let l = u.to_lowercase();
    matches!(l.as_str(), "ms" | "s" | "m" | "h" | "d" | "w")
}

/// Split a string into a leading `[0-9.]*` numeric part and the remaining
/// suffix (mirrors the regex groups in the throttle parser).
fn split_num_unit(s: &str) -> (&str, &str) {
    let i = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(s.len());
    (&s[..i], &s[i..])
}

fn digits_dots(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit() || b == b'.')
}

/// Dynamic value mirroring the JS parser's states.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    /// JS `[]` - shared storage for `list` and `array` types.
    Arr(Vec<String>),
    /// Insertion-ordered map; map2 values may be absent (JS undefined).
    Map(Vec<(String, Option<String>)>),
    /// Explicit blank for multi types (`--no-x`), resolves to empty at the
    /// defaults pass. Only ever appears pre-resolution.
    BlankMulti,
    /// Result of a `fn` hook producing a non-primitive (throttle spec).
    Throttle(Option<ThrottleSpec>),
}

impl ArgVal {
    /// JS truthiness. Arrays and objects are always truthy in JS, even
    /// when empty - important for the `!ret[key]` sentinel checks.
    pub fn truthy(&self) -> bool {
        match self {
            ArgVal::Null | ArgVal::BlankMulti => false,
            ArgVal::Bool(b) => *b,
            ArgVal::Num(n) => *n != 0.0 && !n.is_nan(),
            ArgVal::Str(s) => !s.is_empty(),
            ArgVal::Arr(_) | ArgVal::Map(_) => true,
            // throttle fn returns null ('0' value) or an object
            ArgVal::Throttle(t) => t.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptSpec {
    pub type_: OptType,
    pub alias: Option<char>,
    pub default: Option<ArgVal>,
    pub if_set_default: Option<ArgVal>,
    pub required: bool,
    pub map: Option<&'static str>,
    pub func: Option<OptFn>,
}

impl Default for OptSpec {
    fn default() -> Self {
        OptSpec {
            type_: OptType::Str,
            alias: None,
            default: None,
            if_set_default: None,
            required: false,
            map: None,
            func: None,
        }
    }
}

/// Parse result: `positional` is the JS `_` array.
#[derive(Debug, Default)]
pub struct ParsedArgs {
    pub map: HashMap<String, ArgVal>,
    pub positional: Vec<String>,
}

impl ParsedArgs {
    pub fn get(&self, k: &str) -> Option<&ArgVal> {
        self.map.get(k)
    }
    pub fn has(&self, k: &str) -> bool {
        self.map.contains_key(k)
    }
    pub fn str_val(&self, k: &str) -> Option<&str> {
        match self.map.get(k) {
            Some(ArgVal::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn bool_val(&self, k: &str) -> Option<bool> {
        match self.map.get(k) {
            Some(ArgVal::Bool(b)) => Some(*b),
            _ => None,
        }
    }
    pub fn num_val(&self, k: &str) -> Option<f64> {
        match self.map.get(k) {
            Some(ArgVal::Num(n)) => Some(*n),
            _ => None,
        }
    }
}

/// Port of `module.exports(argv, opts)` for the array form.
/// `opts` keys must already be lowercase.
pub fn parse(argv: &[String], opts: &HashMap<&'static str, OptSpec>) -> Result<ParsedArgs, String> {
    let mut alias_map: HashMap<char, &'static str> = HashMap::new();
    let mut ret: HashMap<String, ArgVal> = HashMap::new();
    let mut positional: Vec<String> = Vec::new();
    let mut apply_fn: Vec<String> = Vec::new();

    for (k, o) in opts.iter() {
        if let Some(a) = o.alias {
            alias_map.insert(a, *k);
        }
    }

    macro_rules! set_key {
        ($key:expr, $val:expr, $explicit:expr) => {
            set_key(&mut ret, &mut apply_fn, opts, $key, $val, $explicit)?
        };
    }

    let mut i = 0usize;
    while i < argv.len() {
        let arg = &argv[i];
        if arg.starts_with('-') {
            if arg.starts_with("--") {
                // long opt
                if arg.len() == 2 {
                    positional.extend_from_slice(&argv[i + 1..]);
                    break;
                }
                let eq = arg.find('=');
                if arg.len() >= 5 && arg.as_bytes()[2..5].eq_ignore_ascii_case(b"no-") {
                    if eq.is_some() {
                        return Err(format!("Unexpected value specified in `{}`", arg));
                    }
                    let k = arg[5..].to_lowercase();
                    if let Some(opt) = opts.get(k.as_str()) {
                        if !opt.type_.is_multiple() && opt.type_ != OptType::Bool {
                            // for multi-value types, --no-opt sets a blank
                            return Err(format!("Cannot specify `{}`", arg));
                        }
                    }
                    set_key!(k, ArgVal::Bool(false), true);
                } else if let Some(eq) = eq {
                    let k = arg[2..eq].to_lowercase();
                    let v = arg[eq + 1..].to_string();
                    set_key!(k, ArgVal::Str(v), true);
                } else {
                    let k = arg[2..].to_lowercase();
                    let opt = opts.get(k.as_str());
                    if opt.map(|o| o.type_) == Some(OptType::Bool) {
                        set_key!(k, ArgVal::Bool(true), true);
                    } else {
                        let next = argv.get(i + 1);
                        match next {
                            Some(n) if !(n.starts_with('-') && n.len() > 1) => {
                                set_key!(k, ArgVal::Str(n.clone()), false);
                                i += 1;
                            }
                            _ => set_key!(k, ArgVal::Null, false),
                        }
                    }
                }
            } else {
                // short opt - iterate bundled characters
                let chars: Vec<char> = arg.chars().collect();
                let mut j = 1usize;
                while j < chars.len() {
                    let c = chars[j];
                    let Some(k) = alias_map.get(&c) else {
                        return Err(format!("Unknown option specified in `{}`", arg));
                    };
                    let opt = &opts[k];
                    let next_char = chars.get(j + 1);
                    if opt.type_ == OptType::Bool && next_char != Some(&'=') {
                        set_key!(k.to_string(), ArgVal::Bool(true), true);
                        j += 1;
                        continue;
                    }
                    // everything else is the value
                    j += 1;
                    match chars.get(j) {
                        None => {
                            let next = argv.get(i + 1);
                            match next {
                                Some(n) if !(n.starts_with('-') && n.len() > 1) => {
                                    set_key!(k.to_string(), ArgVal::Str(n.clone()), false);
                                    i += 1;
                                }
                                _ => set_key!(k.to_string(), ArgVal::Null, false),
                            }
                        }
                        Some(&ch) => {
                            let explicit = ch == '=';
                            if !explicit && j > 2 {
                                // `-bkval` where `-b` is a bool and `-k`
                                // expects a value - ambiguous, reject
                                return Err(format!("Ambiguous option `{}` supplied, as `{}` (`{}`) expects a value; please check usage", arg, ch, k));
                            }
                            let v: String = chars[j + explicit as usize..].iter().collect();
                            set_key!(k.to_string(), ArgVal::Str(v), explicit);
                        }
                    }
                    break;
                }
            }
        } else {
            positional.push(arg.clone());
        }
        i += 1;
    }

    // apply functions to multi-valued items
    for k in apply_fn {
        if let Some(f) = &opts[k.as_str()].func {
            let v = ret.remove(&k).unwrap();
            ret.insert(k.clone(), f.apply(&k, v)?);
        }
    }

    // defaults + multi-value resolution
    for (k, o) in opts.iter() {
        if o.default.is_some() && !ret.contains_key(*k) {
            ret.insert(k.to_string(), o.default.clone().unwrap());
        } else if ret.contains_key(*k) && o.type_.is_multiple() && !ret[*k].truthy() {
            let v = if matches!(ret[*k], ArgVal::Null) {
                o.if_set_default.clone().unwrap_or(ArgVal::Null)
            } else {
                match o.type_ {
                    OptType::Map | OptType::Map2 => ArgVal::Map(Vec::new()),
                    _ => ArgVal::Arr(Vec::new()),
                }
            };
            ret.insert(k.to_string(), v);
        }
        if !ret.contains_key(*k) && o.required {
            return Err(format!("Missing value for `{}`", k));
        }
    }

    Ok(ParsedArgs {
        map: ret,
        positional,
    })
}

/// The `setKey` closure from arg_parser.js. `key` must be lowercase.
fn set_key(
    ret: &mut HashMap<String, ArgVal>,
    apply_fn: &mut Vec<String>,
    opts: &HashMap<&'static str, OptSpec>,
    key: String,
    val: ArgVal,
    explicit: bool,
) -> Result<(), String> {
    let Some(o) = opts.get(key.as_str()) else {
        return Err(format!("Unknown option `{}`", key));
    };
    let is_multiple = o.type_.is_multiple();
    if ret.contains_key(&key) && !is_multiple {
        return Err(format!("Option `{}` specified more than once", key));
    }

    // booleans
    if o.type_ == OptType::Bool {
        let b = match val {
            ArgVal::Bool(b) => b,
            ArgVal::Str(ref s) => match s.to_lowercase().as_str() {
                "true" => true,
                "false" | "0" => false,
                _ => return Err(format!("Unexpected value for `{}`", key)),
            },
            _ => return Err(format!("Unexpected value for `{}`", key)),
        };
        let mut v = ArgVal::Bool(b);
        if let Some(f) = &o.func {
            v = f.apply(&key, v)?;
        }
        ret.insert(key, v);
        return Ok(());
    }

    if !explicit {
        if let ArgVal::Str(ref s) = val {
            if s.starts_with('-') && s.len() > 1 {
                return Err(format!("Potentially incorrect usage - trying to set `{}` to `{}`; if you intend this, please specify `--{}={}` instead", key, s, key, s));
            }
        }
    }

    if is_multiple {
        // no-value / explicit-blank sentinels
        let no_value = matches!(val, ArgVal::Null)
            || matches!(&val, ArgVal::Str(s) if s.is_empty() && !explicit);
        if no_value {
            if o.if_set_default.is_none() || ret.contains_key(&key) {
                return Err(format!("No value specified for `{}`", key));
            }
            ret.insert(key, ArgVal::Null); // mark as set-but-blank
            return Ok(());
        }
        if matches!(val, ArgVal::Bool(false)) {
            // explicit blank via --no-x
            if ret.contains_key(&key) {
                if matches!(ret[&key], ArgVal::BlankMulti) {
                    return Err(format!("Option `{}` specified more than once", key));
                } else {
                    return Err(format!("Conflicting values passed for `{}`", key));
                }
            }
            ret.insert(key, ArgVal::BlankMulti);
            return Ok(());
        }

        if !ret.contains_key(&key) {
            let empty = match o.type_ {
                OptType::Map | OptType::Map2 => ArgVal::Map(Vec::new()),
                _ => ArgVal::Arr(Vec::new()),
            };
            ret.insert(key.clone(), empty);
        } else if !ret[&key].truthy() {
            if matches!(ret[&key], ArgVal::Null) {
                return Err(format!("No value specified for `{}`", key));
            } else {
                return Err(format!("Conflicting values passed for `{}`", key));
            }
        }

        let s = match val {
            ArgVal::Str(s) => s,
            _ => return Err(format!("Invalid value specified for `{}`", key)),
        };
        match o.type_ {
            OptType::List => {
                if let ArgVal::Arr(v) = ret.get_mut(&key).unwrap() {
                    v.extend(s.split(',').map(|x| x.trim().to_lowercase()));
                }
            }
            OptType::Array => {
                if let ArgVal::Arr(v) = ret.get_mut(&key).unwrap() {
                    v.push(s);
                }
            }
            OptType::Map | OptType::Map2 => {
                // split at the first = or : - the JS regex /^(.+?)[=:](.*)$/
                // requires at least one key char, so an empty key fails
                // the match and falls through to the Invalid/Map2 path
                let pos = s.find(|c| c == '=' || c == ':').filter(|&p| p > 0);
                if let ArgVal::Map(m) = ret.get_mut(&key).unwrap() {
                    if let Some(p) = pos {
                        let (k_part, v_part) = s.split_at(p);
                        map_insert(
                            m,
                            k_part.trim().to_string(),
                            Some(v_part[1..].trim().to_string()),
                        );
                    } else if o.type_ == OptType::Map2 {
                        map_insert(m, s.trim().to_string(), None);
                    } else {
                        return Err(format!("Invalid format for `{}`", key));
                    }
                }
            }
            _ => unreachable!(),
        }
        if o.func.is_some() && !apply_fn.contains(&key) {
            apply_fn.push(key.clone());
        }
    } else {
        let mut v = match val {
            ArgVal::Null => {
                if let Some(d) = &o.if_set_default {
                    d.clone()
                } else {
                    return Err(format!("No value specified for `{}`", key));
                }
            }
            ArgVal::Str(s) => {
                if s.is_empty() && !explicit {
                    if let Some(d) = &o.if_set_default {
                        d.clone()
                    } else {
                        return Err(format!("No value specified for `{}`", key));
                    }
                } else {
                    ArgVal::Str(s)
                }
            }
            other => other,
        };
        if let ArgVal::Str(ref sv) = v {
            v = match o.type_ {
                OptType::Int => {
                    if !compat::is_digits(sv) {
                        return Err(format!("Invalid number specified for `{}`", key));
                    }
                    let n = compat::str_to_int32(sv);
                    if n < 0 {
                        return Err(format!("Invalid number specified for `{}`", key));
                    }
                    ArgVal::Num(n as f64)
                }
                OptType::NInt => {
                    if !compat::is_signed_digits(sv) {
                        return Err(format!("Invalid number specified for `{}`", key));
                    }
                    ArgVal::Num(compat::str_to_int32(sv) as f64)
                }
                OptType::Size => match parse_size(sv) {
                    Some(n) if n != 0.0 => ArgVal::Num(n),
                    _ => return Err(format!("Invalid size specified for `{}`", key)),
                },
                OptType::Size0 => match parse_size(sv) {
                    Some(n) => ArgVal::Num(n),
                    None => return Err(format!("Invalid size specified for `{}`", key)),
                },
                OptType::Time => match parse_time(sv) {
                    Some(n) => ArgVal::Num(n),
                    None => return Err(format!("Invalid time specified for `{}`", key)),
                },
                OptType::Enum(vals) => {
                    if !vals.contains(&sv.to_lowercase().as_str()) {
                        return Err(format!("Invalid value specified for `{}`", key));
                    }
                    ArgVal::Str(sv.clone())
                }
                _ => ArgVal::Str(sv.clone()),
            };
        }
        if let Some(f) = &o.func {
            v = f.apply(&key, v)?;
        }
        ret.insert(key, v);
    }
    Ok(())
}

fn map_insert(m: &mut Vec<(String, Option<String>)>, k: String, v: Option<String>) {
    // JS objects: overwrite keeps the original insertion position
    for e in m.iter_mut() {
        if e.0 == k {
            e.1 = v;
            return;
        }
    }
    m.push((k, v));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(type_: OptType) -> OptSpec {
        OptSpec {
            type_,
            ..Default::default()
        }
    }

    fn opts(entries: &[(&'static str, OptSpec)]) -> HashMap<&'static str, OptSpec> {
        entries.iter().cloned().collect()
    }

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("123"), Some(123.0));
        assert_eq!(parse_size("0"), Some(0.0));
        assert_eq!(parse_size("1K"), Some(1024.0));
        assert_eq!(parse_size("1.5M"), Some(1572864.0));
        assert_eq!(parse_size("2g"), Some(2147483648.0));
        assert_eq!(parse_size("1B"), Some(1.0));
        assert_eq!(parse_size("1.2.3K"), None);
        assert_eq!(parse_size("1.5"), None); // digits+dot, no unit
        assert_eq!(parse_size("K"), None);
        assert_eq!(parse_size("abc"), None);
        assert_eq!(parse_size("-5"), None); // '-' not in [0-9.]
        assert_eq!(parse_size("1E"), Some(1152921504606846976.0));
    }

    #[test]
    fn times() {
        assert_eq!(parse_time("5"), Some(5000.0));
        assert_eq!(parse_time("500ms"), Some(500.0));
        assert_eq!(parse_time("1.5s"), Some(1500.0));
        assert_eq!(parse_time("2m"), Some(120000.0));
        assert_eq!(parse_time("1h"), Some(3600000.0));
        assert_eq!(parse_time("1d"), Some(86400000.0));
        assert_eq!(parse_time("1w"), Some(604800000.0));
        assert_eq!(parse_time("1MS"), Some(1.0)); // lowercased
        assert_eq!(parse_time("1x"), None);
        assert_eq!(parse_time("m"), None);
    }

    #[test]
    fn throttle() {
        assert_eq!(parse_throttle("seq", "0").unwrap(), None);
        let t = parse_throttle("seq", "1k/2s").unwrap().unwrap();
        assert_eq!(t.mode, ThrottleMode::Rate);
        assert_eq!(t.size, 1024.0);
        assert_eq!(t.time, 2000.0);
        let t = parse_throttle("seq", "10r/1m").unwrap().unwrap();
        assert_eq!(t.mode, ThrottleMode::ReqRate);
        assert_eq!(t.count, "10");
        assert_eq!(t.time, 60000.0);
        let t = parse_throttle("seq", "250ms").unwrap().unwrap();
        assert_eq!(t.mode, ThrottleMode::Delay);
        assert_eq!(t.time, 250.0);
        // JS defaults empty numeric groups to '1'
        let t = parse_throttle("seq", "k/2s").unwrap().unwrap();
        assert_eq!(t.size, 1024.0);
        let t = parse_throttle("chunk", "r/1s").unwrap().unwrap();
        assert_eq!(t.count, "1");
        let t = parse_throttle("seq", "s").unwrap().unwrap(); // '' -> '1'
        assert_eq!(t.time, 1000.0);
        assert!(parse_throttle("seq", "/2s").is_err());
        assert!(parse_throttle("seq", "k/").is_err());
        assert!(parse_throttle("seq", "junk").is_err());
    }

    #[test]
    fn basic_parse() {
        let o = opts(&[
            ("out", spec(OptType::Str)),
            ("quiet", {
                let mut s = spec(OptType::Bool);
                s.alias = Some('q');
                s
            }),
            ("recovery-slices", {
                let mut s = spec(OptType::Str);
                s.alias = Some('r');
                s
            }),
            ("n", spec(OptType::Str)),
        ]);
        let r = parse(
            &argv(&["-q", "--out", "x.par2", "-r", "100", "file1", "file2"]),
            &o,
        )
        .unwrap();
        assert_eq!(r.bool_val("quiet"), Some(true));
        assert_eq!(r.str_val("out"), Some("x.par2"));
        assert_eq!(r.str_val("recovery-slices"), Some("100"));
        assert_eq!(r.positional, vec!["file1", "file2"]);
    }

    #[test]
    fn errors_match_js() {
        let o = opts(&[
            ("out", spec(OptType::Str)),
            ("n", {
                let mut s = spec(OptType::Int);
                s.alias = Some('n');
                s
            }),
        ]);
        let e = parse(&argv(&["--bogus"]), &o).unwrap_err();
        assert_eq!(e, "Unknown option `bogus`");
        let e = parse(&argv(&["--out", "a", "--out", "b"]), &o).unwrap_err();
        assert_eq!(e, "Option `out` specified more than once");
        let e = parse(&argv(&["--n", "-5"]), &o).unwrap_err();
        assert_eq!(e, "No value specified for `n`");
        let e = parse(&argv(&["-n-5"]), &o).unwrap_err();
        assert_eq!(e, "Potentially incorrect usage - trying to set `n` to `-5`; if you intend this, please specify `--n=-5` instead");
        let e = parse(&argv(&["--n=-5"]), &o).unwrap_err();
        assert_eq!(e, "Invalid number specified for `n`");
        let e = parse(&argv(&["--out"]), &o).unwrap_err();
        assert_eq!(e, "No value specified for `out`");
        let e = parse(&argv(&["-z"]), &o).unwrap_err();
        assert_eq!(e, "Unknown option specified in `-z`");
        let e = parse(&argv(&["--no-out"]), &o).unwrap_err();
        assert_eq!(e, "Cannot specify `--no-out`");
        let r = parse(&argv(&["--out="]), &o).unwrap();
        // explicit blank on scalar string is allowed
        assert_eq!(r.str_val("out"), Some(""));
    }

    #[test]
    fn bool_handling() {
        let o = opts(&[("q", spec(OptType::Bool))]);
        assert_eq!(
            parse(&argv(&["--q"]), &o).unwrap().bool_val("q"),
            Some(true)
        );
        assert_eq!(
            parse(&argv(&["--no-q"]), &o).unwrap().bool_val("q"),
            Some(false)
        );
        assert_eq!(
            parse(&argv(&["--q=false"]), &o).unwrap().bool_val("q"),
            Some(false)
        );
        assert_eq!(
            parse(&argv(&["--q=0"]), &o).unwrap().bool_val("q"),
            Some(false)
        );
        assert!(parse(&argv(&["--q=yes"]), &o).is_err());
        assert!(parse(&argv(&["--no-q=x"]), &o).is_err());
        // long-opt case-insensitivity on the key, --no- prefix too
        assert_eq!(
            parse(&argv(&["--Q"]), &o).unwrap().bool_val("q"),
            Some(true)
        );
        assert_eq!(
            parse(&argv(&["--NO-Q"]), &o).unwrap().bool_val("q"),
            Some(false)
        );
    }

    #[test]
    fn multi_types() {
        let o = opts(&[
            ("c", {
                let mut s = spec(OptType::Array);
                s.alias = Some('c');
                s
            }),
            ("l", spec(OptType::List)),
            ("m", spec(OptType::Map)),
            ("m2", spec(OptType::Map2)),
        ]);
        let r = parse(&argv(&["-c", "a", "-c", "b"]), &o).unwrap();
        assert_eq!(r.get("c"), Some(&ArgVal::Arr(vec!["a".into(), "b".into()])));
        let r = parse(&argv(&["--l", " A,b ,C "]), &o).unwrap();
        assert_eq!(
            r.get("l"),
            Some(&ArgVal::Arr(vec!["a".into(), "b".into(), "c".into()]))
        );
        let r = parse(&argv(&["--m", "k1=v1", "--m", "k2:v2"]), &o).unwrap();
        assert_eq!(
            r.get("m"),
            Some(&ArgVal::Map(vec![
                ("k1".into(), Some("v1".into())),
                ("k2".into(), Some("v2".into()))
            ]))
        );
        let r = parse(&argv(&["--m2", "flag"]), &o).unwrap();
        assert_eq!(r.get("m2"), Some(&ArgVal::Map(vec![("flag".into(), None)])));
        assert!(parse(&argv(&["--m", "nokv"]), &o).is_err());
        // --no- on multi gives explicit blank
        let r = parse(&argv(&["--no-l"]), &o).unwrap();
        assert_eq!(r.get("l"), Some(&ArgVal::Arr(vec![])));
        // map overwrite keeps position
        let r = parse(&argv(&["--m", "a=1", "--m", "a=2"]), &o).unwrap();
        assert_eq!(
            r.get("m"),
            Some(&ArgVal::Map(vec![("a".into(), Some("2".into()))]))
        );
        // --no-x twice -> "specified more than once"; --no-x then --x=v -> conflicting
        let e = parse(&argv(&["--no-l", "--no-l"]), &o).unwrap_err();
        assert_eq!(e, "Option `l` specified more than once");
        let e = parse(&argv(&["--no-l", "--l", "a"]), &o).unwrap_err();
        assert_eq!(e, "Conflicting values passed for `l`");
    }

    #[test]
    fn short_bundling_and_terminator() {
        let o = opts(&[
            ("a", {
                let mut s = spec(OptType::Bool);
                s.alias = Some('a');
                s
            }),
            ("b", {
                let mut s = spec(OptType::Bool);
                s.alias = Some('b');
                s
            }),
            ("k", {
                let mut s = spec(OptType::Str);
                s.alias = Some('k');
                s
            }),
        ]);
        let r = parse(&argv(&["-ab"]), &o).unwrap();
        assert_eq!(r.bool_val("a"), Some(true));
        assert_eq!(r.bool_val("b"), Some(true));
        // -abkval: a,b bool then k sees 'v' at index 4 (>2) -> ambiguous
        let e = parse(&argv(&["-abkval"]), &o).unwrap_err();
        assert_eq!(
            e,
            "Ambiguous option `-abkval` supplied, as `v` (`k`) expects a value; please check usage"
        );
        let r = parse(&argv(&["--", "-a", "--x"]), &o).unwrap();
        assert_eq!(r.positional, vec!["-a", "--x"]);
        let r = parse(&argv(&["-k=eq"]), &o).unwrap();
        assert_eq!(r.str_val("k"), Some("eq"));
        // -kb: k consumes 'b' as its value at index 2 - allowed
        let r = parse(&argv(&["-kb"]), &o).unwrap();
        assert_eq!(r.str_val("k"), Some("b"));
        // -k at end consumes the next arg; '-' alone is consumable
        let r = parse(&argv(&["-k", "-"]), &o).unwrap();
        assert_eq!(r.str_val("k"), Some("-"));
        // but a dash-prefixed arg is not
        let r = parse(&argv(&["-k", "-a"]), &o);
        assert!(r.is_err()); // No value specified for `k`
    }

    #[test]
    fn ifsetdefault_and_defaults() {
        let o = opts(&[
            ("ol", {
                let mut s = spec(OptType::Str);
                s.if_set_default = Some(ArgVal::Str("gpu".into()));
                s
            }),
            ("enc", {
                let mut s = spec(OptType::Str);
                s.default = Some(ArgVal::Str("utf8".into()));
                s
            }),
            ("arr", {
                let mut s = spec(OptType::Array);
                s.if_set_default = Some(ArgVal::Arr(vec!["d".into()]));
                s
            }),
        ]);
        let r = parse(&argv(&["--ol"]), &o).unwrap();
        assert_eq!(r.str_val("ol"), Some("gpu"));
        let r = parse(&argv(&[]), &o).unwrap();
        assert_eq!(r.str_val("enc"), Some("utf8"));
        let r = parse(&argv(&["--arr"]), &o).unwrap();
        assert_eq!(r.get("arr"), Some(&ArgVal::Arr(vec!["d".into()])));
    }

    #[test]
    fn size_types() {
        let o = opts(&[
            ("s", spec(OptType::Size)),
            ("s0", spec(OptType::Size0)),
            ("t", spec(OptType::Time)),
        ]);
        let e = parse(&argv(&["--s", "0"]), &o).unwrap_err();
        assert_eq!(e, "Invalid size specified for `s`");
        let r = parse(&argv(&["--s0", "0"]), &o).unwrap();
        assert_eq!(r.num_val("s0"), Some(0.0));
        let r = parse(&argv(&["--s", "2M", "--t", "3m"]), &o).unwrap();
        assert_eq!(r.num_val("s"), Some(2097152.0));
        assert_eq!(r.num_val("t"), Some(180000.0));
    }

    #[test]
    fn int_wraparound() {
        // val|0 wraps: "4294967296" -> 0 (0>=0 and matches \d+ -> passes)
        let o = opts(&[("n", spec(OptType::Int))]);
        let r = parse(&argv(&["--n", "4294967296"]), &o).unwrap();
        assert_eq!(r.num_val("n"), Some(0.0));
        let o2 = opts(&[("e", spec(OptType::NInt))]);
        let r = parse(&argv(&["--e=-5"]), &o2).unwrap();
        assert_eq!(r.num_val("e"), Some(-5.0));
    }
}

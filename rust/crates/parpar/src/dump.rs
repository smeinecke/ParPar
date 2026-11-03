//! Dev/test aid: emit the `{argv, ppo, inputFiles}` state exactly as the
//! JS `test/dump_args.js` harness captures it right before generation.
//! Enabled via `PARPAR_DUMP_ARGS=1` so the CLI surface stays identical.

use crate::args::{
    opts_table, CliConfig, CriticalRedundancy, OclDeviceSpec, SpecExpr, SpecTerm, SpecUnit,
};
use crate::cliparse::{ArgVal, ThrottleMode, ThrottleSpec};
use serde_json::{json, Map, Value};

/// JSON.stringify drops keys whose value is `undefined` (map2 entries).
fn argval_json(v: &ArgVal) -> Option<Value> {
    Some(match v {
        ArgVal::Null => Value::Null,
        ArgVal::Bool(b) => json!(b),
        ArgVal::Num(n) => js_num(*n),
        ArgVal::Str(s) => json!(s),
        ArgVal::Arr(a) => json!(a),
        ArgVal::Map(m) => {
            let mut o = Map::new();
            for (k, val) in m {
                if let Some(val) = val {
                    o.insert(k.clone(), json!(val));
                }
            }
            Value::Object(o)
        }
        // BlankMulti never survives the defaults pass into a dump
        ArgVal::BlankMulti => return None,
        ArgVal::Throttle(t) => throttle_json(t),
    })
}

/// V8's Number::toString is the shortest roundtrip decimal, positional up
/// to 21 digits then exponential. Rust's `{}` produces the same shortest
/// digits but never uses exponential - fine for the magnitudes that occur
/// here (|x| < 1e21, |x| >= 1e-6). Parsing the text back into a Value
/// yields integer JSON where the digits are integral (2^60 ->
/// '1152921504606847000').
fn js_num(n: f64) -> Value {
    if !n.is_finite() {
        return Value::Null; // NaN/Infinity stringify to null
    }
    let s = format!("{}", n);
    serde_json::from_str::<Value>(&s).unwrap_or(Value::Null)
}

/// `{mode, count|size, time}` / null - field order from throttleParseFunc.
fn throttle_json(t: &Option<ThrottleSpec>) -> Value {
    let Some(t) = t else { return Value::Null };
    match t.mode {
        ThrottleMode::ReqRate => {
            json!({"mode": "reqrate", "count": t.count, "time": js_num(t.time)})
        }
        ThrottleMode::Rate => {
            json!({"mode": "rate", "size": js_num(t.size), "time": js_num(t.time)})
        }
        ThrottleMode::Delay => json!({"mode": "delay", "time": js_num(t.time)}),
    }
}

fn spec_unit_name(u: &SpecUnit) -> &'static str {
    match u {
        SpecUnit::Slices => "slices",
        SpecUnit::Count => "count",
        SpecUnit::Bytes => "bytes",
        SpecUnit::Ratio => "ratio",
        SpecUnit::LargestFiles => "largest_files",
        SpecUnit::SmallestFiles => "smallest_files",
        SpecUnit::Power => "power",
        SpecUnit::Log => "log",
        SpecUnit::ILog => "ilog",
    }
}

fn spec_term_json(t: &SpecTerm) -> Value {
    match t.unit {
        SpecUnit::LargestFiles
        | SpecUnit::SmallestFiles
        | SpecUnit::Power
        | SpecUnit::Log
        | SpecUnit::ILog => json!({
            "unit": spec_unit_name(&t.unit),
            "value": js_num(t.value),
            "scale": js_num(t.scale),
        }),
        _ => json!({"unit": spec_unit_name(&t.unit), "value": js_num(t.value)}),
    }
}

fn spec_expr_json(e: &SpecExpr) -> Value {
    Value::Array(e.iter().map(spec_term_json).collect())
}

/// The merged `{ratio?, platform?, device?, memoryLimit?, method?,
/// input_batchsize?, target_iters?, target_grouping?, minChunkSize?}` in
/// openclMap insertion order; only set fields appear.
fn ocl_spec_json(s: &OclDeviceSpec) -> Value {
    let mut o = Map::new();
    let mut put = |k: &str, v: Value| {
        o.insert(k.to_string(), v);
    };
    if let Some(r) = s.ratio {
        put("ratio", js_num(r));
    }
    if let Some(p) = s.platform {
        put("platform", json!(p));
    }
    if let Some(d) = s.device {
        put("device", json!(d));
    }
    if let Some(m) = s.memory_limit {
        put("memoryLimit", js_num(m));
    }
    if let Some(m) = &s.method {
        put("method", json!(m));
    }
    if let Some(v) = s.input_batchsize {
        put("input_batchsize", json!(v));
    }
    if let Some(v) = s.target_iters {
        put("target_iters", json!(v));
    }
    if let Some(v) = s.target_grouping {
        put("target_grouping", json!(v));
    }
    if let Some(v) = s.min_chunk_size {
        put("minChunkSize", js_num(v));
    }
    Value::Object(o)
}

/// `{argv, ppo, inputFiles}` - JSON.stringify single line, matching the
/// dump_args.js output contract. `argv` key order can't fully match JS
/// insertion order (HashMap) - `_` goes first, the rest is sorted; the
/// differential comparator normalizes anyway.
pub fn dump_json(cfg: &CliConfig) -> String {
    let mut argv = Map::new();
    argv.insert("_".into(), json!(cfg.argv.positional));
    let mut keys: Vec<&String> = cfg.argv.map.keys().collect();
    keys.sort();
    for k in keys {
        if let Some(v) = argval_json(&cfg.argv.map[k]) {
            argv.insert(k.clone(), v);
        }
    }

    let ppo = &cfg.ppo;
    let mut p = Map::new();
    let mut put = |k: &str, v: Value| {
        p.insert(k.to_string(), v);
    };
    // ppo = {outputBase, recoverySlicesUnit: 'slices', creator}
    let mut base = cfg.out.clone();
    // byte-level compare: str indexing would panic if len-5 lands mid-codepoint
    if base.len() >= 5 && base.as_bytes()[base.len() - 5..].eq_ignore_ascii_case(b".par2") {
        base.truncate(base.len() - 5);
    }
    put("outputBase", json!(base));
    put("recoverySlicesUnit", json!("slices"));
    put("creator", json!(ppo.creator));
    // ppo[map] = argv[k] for every mapped key present in argv
    for (k, spec) in opts_table() {
        if let Some(mapped) = spec.map {
            if let Some(v) = cfg.argv.map.get(k) {
                if let Some(j) = argval_json(v) {
                    put(mapped, j);
                }
            }
        }
    }
    // sliceSizeMultiple: JS can leave `false` here (unparseable input) -
    // Option None serializes as false
    put(
        "sliceSizeMultiple",
        ppo.slice_size_multiple.map(js_num).unwrap_or(json!(false)),
    );
    // spec-expression arrays (rounding keys are set before the array in JS)
    if let Some(e) = &cfg.recovery_slices {
        put("recoverySlices", spec_expr_json(e));
    }
    if let Some(e) = &cfg.min_recovery_slices {
        put("minRecoverySlices", spec_expr_json(e));
    }
    if let Some(e) = &cfg.max_recovery_slices {
        put("maxRecoverySlices", spec_expr_json(e));
    }
    if let Some(r) = &cfg.output_file_max_slices_rounding {
        put(
            "outputFileMaxSlicesRounding",
            json!(match r {
                crate::args::SpecRounding::Floor => "floor",
                crate::args::SpecRounding::Ceil => "ceil",
            }),
        );
    }
    if let Some(e) = &cfg.output_file_max_slices {
        put("outputFileMaxSlices", spec_expr_json(e));
    }
    if let Some(r) = &cfg.output_first_file_slices_rounding {
        put(
            "outputFirstFileSlicesRounding",
            json!(match r {
                crate::args::SpecRounding::Floor => "floor",
                crate::args::SpecRounding::Ceil => "ceil",
            }),
        );
    }
    if let Some(e) = &cfg.output_first_file_slices {
        put("outputFirstFileSlices", spec_expr_json(e));
    }
    // criticalRedundancyScheme: pow2/none stays the mapped string; other
    // values were overwritten by the spec array
    if let Some(c) = &ppo.critical_redundancy_scheme {
        match c {
            CriticalRedundancy::Named(s) => put("criticalRedundancyScheme", json!(s)),
            CriticalRedundancy::Expr(e) => put("criticalRedundancyScheme", spec_expr_json(e)),
        }
    }
    if let Some(t) = &cfg.min_slice_size {
        put("minSliceSize", spec_term_json(t));
    }
    if let Some(t) = &cfg.max_slice_size {
        put("maxSliceSize", spec_term_json(t));
    }
    if let Some(devs) = &ppo.opencl_devices {
        put(
            "openclDevices",
            Value::Array(devs.iter().map(ocl_spec_json).collect()),
        );
    }

    let out = json!({
        "argv": Value::Object(argv),
        "ppo": Value::Object(p),
        "inputFiles": cfg.input_files,
    });
    serde_json::to_string(&out).unwrap()
}

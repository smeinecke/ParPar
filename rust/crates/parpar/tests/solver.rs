//! Differential test: PAR2Gen solver/layout vs the JS oracle.
//! Cases: fixtures/gen/cases.json. Oracles: fixtures/gen/<i>_<id>.json
//! produced by test/dump_gen.js (real binding + real OpenCL hardware).

#![cfg(target_env = "gnu")]
use parpar::args::{SpecExpr, SpecTerm, SpecUnit};
use parpar::cliparse::{ThrottleMode, ThrottleSpec};
use parpar::fileinfo::FileInfo;
use parpar::generate::{CritScheme, GenOpts, OclDevOpt, Par2Gen, SolverInfo, SysEnv};
use parpar::par2::{method_num_ref, SharedBackend, GF_METHODS};
use parpar_ffi::FfiBackend;
use serde_json::Value;
use std::collections::BTreeMap;

fn fixture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gen")
}

fn spec_unit(s: &str) -> SpecUnit {
    match s {
        "slices" => SpecUnit::Slices,
        "count" => SpecUnit::Count,
        "bytes" => SpecUnit::Bytes,
        "ratio" => SpecUnit::Ratio,
        "largest_files" => SpecUnit::LargestFiles,
        "smallest_files" => SpecUnit::SmallestFiles,
        "power" => SpecUnit::Power,
        "log" => SpecUnit::Log,
        "ilog" => SpecUnit::ILog,
        other => panic!("bad spec unit {other}"),
    }
}

fn term_from_json(v: &Value) -> SpecTerm {
    if v.is_number() {
        // bare number = slices/count spec
        return SpecTerm {
            unit: SpecUnit::Count,
            value: v.as_f64().unwrap(),
            scale: 1.0,
        };
    }
    SpecTerm {
        unit: spec_unit(v["unit"].as_str().unwrap()),
        value: v["value"].as_f64().unwrap_or(0.0),
        scale: v["scale"].as_f64().unwrap_or(1.0),
    }
}

fn expr_from_json(v: &Value) -> SpecExpr {
    if let Value::Array(a) = v {
        a.iter().map(term_from_json).collect()
    } else {
        vec![term_from_json(v)]
    }
}

fn opt_expr(v: &Value) -> Option<SpecExpr> {
    if v.is_null() {
        None
    } else {
        Some(expr_from_json(v))
    }
}

fn rounding(v: &Value) -> Option<parpar::args::SpecRounding> {
    match v.as_str() {
        Some("floor") => Some(parpar::args::SpecRounding::Floor),
        Some("ceil") => Some(parpar::args::SpecRounding::Ceil),
        _ => None, // 'round'/absent -> None (JS default)
    }
}

fn throttle(v: &Value) -> Option<ThrottleSpec> {
    if v.is_null() {
        return None;
    }
    let mode = match v["mode"].as_str() {
        Some("delay") => ThrottleMode::Delay,
        Some("rate") => ThrottleMode::Rate,
        _ => ThrottleMode::ReqRate,
    };
    Some(ThrottleSpec {
        mode,
        time: v["time"].as_f64().unwrap_or(0.0),
        size: v["size"].as_f64().unwrap_or(0.0),
        count: v["count"]
            .as_f64()
            .map(|c| c.to_string())
            .unwrap_or_default(),
    })
}

fn opts_from_json(v: &Value) -> GenOpts {
    let mut o = GenOpts::default();
    let g = |k: &str| &v[k];
    if !g("gfMethod").is_null() {
        o.gf_method = Some(match g("gfMethod") {
            Value::String(s) => parpar::args::MethodRef::Name(s.clone()),
            Value::Number(n) => parpar::args::MethodRef::Index(n.as_f64().unwrap_or(f64::NAN)),
            _ => parpar::args::MethodRef::Name("null".into()),
        });
    }
    // null gfMethod stays None => JS null error path
    if !g("recoverySlices").is_null() {
        o.recovery_slices = expr_from_json(g("recoverySlices"));
    }
    o.min_recovery_slices = opt_expr(g("minRecoverySlices"));
    o.max_recovery_slices = opt_expr(g("maxRecoverySlices")).or(o.max_recovery_slices);
    if let Some(x) = g("recoveryOffset").as_i64() {
        o.recovery_offset = x as i32;
    }
    if let Some(a) = g("recoveryExponents").as_array() {
        o.recovery_exponents = Some(
            a.iter()
                .map(|e| e.as_str().unwrap_or("").to_string())
                .collect(),
        );
    }
    if let Some(x) = g("memoryLimit").as_f64() {
        o.memory_limit = Some(x);
    } else if !g("memoryLimit").is_null() {
        o.memory_limit = Some(f64::NAN); // JS truthy non-number -> NaN math
    }
    if let Some(x) = g("minChunkSize").as_f64() {
        o.min_chunk_size = x;
    }
    if let Some(x) = g("processBatchSize").as_f64() {
        o.process_batch_size = Some(x);
    }
    if let Some(x) = g("hashBatchSize").as_f64() {
        o.hash_batch_size = x;
    }
    if let Some(x) = g("recDataSize").as_f64() {
        o.rec_data_size = Some(x);
    }
    if let Some(a) = g("comments").as_array() {
        o.comments = a
            .iter()
            .map(|e| e.as_str().unwrap_or("").to_string())
            .collect();
    }
    if let Some(s) = g("creator").as_str() {
        o.creator = s.to_string();
    }
    if let Some(x) = g("unicode").as_bool() {
        o.unicode = Some(x);
    }
    if let Some(x) = g("outputOverwrite").as_bool() {
        o.output_overwrite = x;
    }
    if let Some(x) = g("outputSync").as_bool() {
        o.output_sync = x;
    }
    if let Some(x) = g("outputIndex").as_bool() {
        o.output_index = x;
    }
    if let Some(s) = g("outputSizeScheme").as_str() {
        o.output_size_scheme = s.to_string();
    }
    if let Some(x) = opt_expr(g("outputFirstFileSlices")) {
        o.output_first_file_slices = Some(x);
        o.output_first_file_slices_rounding = rounding(g("outputFirstFileSlicesRounding"));
    }
    if let Some(x) = opt_expr(g("outputFileMaxSlices")) {
        o.output_file_max_slices = x;
        o.output_file_max_slices_rounding = rounding(g("outputFileMaxSlicesRounding"));
    }
    if let Some(x) = g("outputFileCount").as_f64() {
        o.output_file_count = x;
    }
    match g("criticalRedundancyScheme") {
        Value::String(s) if s == "none" => o.critical_redundancy_scheme = CritScheme::None,
        Value::String(s) if s == "pow2" => {
            o.critical_redundancy_scheme = CritScheme::Spec(vec![SpecTerm {
                unit: SpecUnit::Log,
                value: 2.0,
                scale: 1.0,
            }])
        }
        Value::Null => {}
        x => o.critical_redundancy_scheme = CritScheme::Spec(expr_from_json(x)),
    }
    if let Some(x) = g("minCriticalRedundancy").as_f64() {
        o.min_critical_redundancy = x;
    }
    if let Some(x) = g("maxCriticalRedundancy").as_f64() {
        o.max_critical_redundancy = x;
    }
    if let Some(x) = g("outputAltNamingScheme").as_bool() {
        o.output_alt_naming_scheme = x;
    }
    if let Some(s) = g("displayNameFormat").as_str() {
        o.display_name_format = s.to_string();
    }
    if let Some(s) = g("displayNameBase").as_str() {
        o.display_name_base = s.to_string();
    }
    if let Some(x) = g("seqReadSize").as_f64() {
        o.seq_read_size = x;
    }
    if let Some(x) = g("chunkReadThreads").as_f64() {
        o.chunk_read_threads = x;
    }
    if !g("seqReadThrottle").is_null() {
        o.seq_read_throttle = throttle(g("seqReadThrottle"));
    }
    if !g("chunkReadThrottle").is_null() {
        o.chunk_read_throttle = throttle(g("chunkReadThrottle"));
    }
    if let Some(x) = g("readBuffers").as_f64() {
        o.read_buffers = x;
    }
    if let Some(x) = g("readHashQueue").as_f64() {
        o.read_hash_queue = x;
    }
    if let Some(x) = g("numThreads").as_i64() {
        o.num_threads = Some(x as i32);
    }
    if let Some(x) = g("loopTileSize").as_f64() {
        o.loop_tile_size = x;
    }
    if let Some(x) = g("cpuMinChunkSize").as_f64() {
        o.cpu_min_chunk_size = x;
    }
    if let Some(x) = g("sliceSizeMultiple").as_f64() {
        o.slice_size_multiple = x;
    }
    if let Some(s) = g("outputBase").as_str() {
        o.output_base = s.to_string();
    }
    o.min_slice_size = opt_term(g("minSliceSize"));
    o.max_slice_size = opt_term(g("maxSliceSize"));
    if let Some(devs) = g("openclDevices").as_array() {
        o.opencl_devices = devs
            .iter()
            .map(|d| OclDevOpt {
                platform: d["platform"].as_i64().map(|x| x as i32).unwrap_or(-1),
                device: d["device"].as_i64().map(|x| x as i32).unwrap_or(-1),
                ratio: d["ratio"].as_f64(),
                memory_limit: d["memoryLimit"].as_f64(),
                method: match &d["method"] {
                    Value::String(s) => Some(parpar::args::MethodRef::Name(s.clone())),
                    Value::Number(n) => Some(parpar::args::MethodRef::Index(
                        n.as_f64().unwrap_or(f64::NAN),
                    )),
                    _ => None,
                },
                input_batchsize: d["input_batchsize"].as_i64().unwrap_or(0) as i32,
                target_iters: d["target_iters"].as_i64().unwrap_or(0) as i32,
                target_grouping: d["target_grouping"].as_i64().unwrap_or(0) as i32,
                min_chunk_size: d["minChunkSize"].as_f64().unwrap_or(32768.0),
                ..Default::default()
            })
            .collect();
    }
    o
}

fn opt_term(v: &Value) -> Option<SpecTerm> {
    if v.is_null() {
        None
    } else {
        Some(term_from_json(v))
    }
}

/// Normalize a JSON value: JS numbers -> fmt_num strings (matching the
/// Rust dump's f64 rendering), undefined->null already handled upstream.
fn norm(v: &Value) -> Value {
    match v {
        Value::Number(n) => {
            let f = n.as_f64().unwrap();
            Value::String(parpar::compat::fmt_num(f))
        }
        Value::Array(a) => Value::Array(a.iter().map(norm).collect()),
        Value::Object(m) => {
            let m: BTreeMap<String, Value> = m.iter().map(|(k, v)| (k.clone(), norm(v))).collect();
            Value::Object(m.into_iter().collect())
        }
        x => x.clone(),
    }
}

/// Deep-compare; only keys present in `expect` are checked (Rust may emit
/// extra diagnostic fields).
fn cmp(path: &str, expect: &Value, got: &Value, errs: &mut Vec<String>) {
    let e = norm(expect);
    let g = norm(got);
    match (&e, &g) {
        (Value::Object(em), Value::Object(gm)) => {
            for (k, ev) in em {
                match gm.get(k) {
                    Some(gv) => cmp(&format!("{path}.{k}"), ev, gv, errs),
                    None => errs.push(format!("{path}.{k}: missing in Rust dump")),
                }
            }
        }
        (Value::Array(ea), Value::Array(ga)) => {
            if ea.len() != ga.len() {
                errs.push(format!("{path}: array len {} != {}", ea.len(), ga.len()));
                return;
            }
            for (i, (ev, gv)) in ea.iter().zip(ga.iter()).enumerate() {
                cmp(&format!("{path}[{i}]"), ev, gv, errs);
            }
        }
        _ => {
            if e != g {
                errs.push(format!("{path}: expected {e} got {g}"));
            }
        }
    }
}

// `auto` gfMethod resolves to whatever kernel the host CPU picks, so the
// oracle's resolved method ids are machine-dependent - scrub them
fn scrub_auto_method(v: &mut Value) {
    if let Some(c) = v.get_mut("procCpu").and_then(|c| c.as_object_mut()) {
        c.insert("method".into(), 0.into());
    }
    for key in ["openclDevices", "procOcl"] {
        if let Some(arr) = v.get_mut(key).and_then(|a| a.as_array_mut()) {
            for d in arr.iter_mut() {
                if let Some(o) = d.as_object_mut() {
                    o.insert("cksum_method".into(), 0.into());
                }
            }
        }
    }
}

#[test]
fn solver_layout_differential() {
    let dir = fixture_dir();
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let backend = FfiBackend::new().unwrap();
    let mut failures = Vec::new();
    let mut ran = 0;
    let mut skipped = 0;
    for (i, case) in cases.iter().enumerate() {
        let id = case["id"].as_str().unwrap().to_string();
        let fx_path = dir.join(format!("{i}_{id}.json"));
        let fx: Value = serde_json::from_str(&std::fs::read_to_string(&fx_path).unwrap()).unwrap();
        let sys = &fx["sys"];
        let env = SysEnv {
            total_mem: sys["total"].as_u64().unwrap_or(0),
            free_mem: sys["free"].as_u64().unwrap_or(0),
            constrained_mem: sys["constrained"].as_u64().unwrap_or(0),
            is_64b: sys["is64"].as_bool().unwrap_or(true),
            k_max_length: sys["kmax"].as_u64().unwrap_or(0),
        };
        let mut files: Vec<FileInfo> = case["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| FileInfo {
                name: f["name"].as_str().unwrap().to_string(),
                display_name: f
                    .get("displayName")
                    .and_then(|d| d.as_str())
                    .map(String::from),
                full_path: f.get("fullPath").and_then(|d| d.as_str()).map(String::from),
                size: f["size"].as_u64().unwrap(),
                md5_16k: {
                    let hex = f["md5_16k"].as_str().unwrap_or("");
                    let mut b = [0u8; 16];
                    if hex.len() == 32 {
                        for j in 0..16 {
                            b[j] = u8::from_str_radix(&hex[j * 2..j * 2 + 2], 16).unwrap();
                        }
                    }
                    b
                },
                md5: None,
                ..Default::default()
            })
            .collect();
        let slice_def = term_from_json(&case["slice_def"]);
        let opts = opts_from_json(&case["opts"]);
        let expect = &fx["out"];

        let mut shared = SharedBackend::new(backend.clone());
        eprintln!("CASE {i} {id}");
        // OCL fixtures encode the generator machine's resolved device - where
        // one can't be resolved the case can't be reproduced (JS itself throws
        // 'Unable to obtain OpenCL device info'); skip unless that failure is
        // exactly what the oracle expects
        let ocl_ok = opts
            .opencl_devices
            .iter()
            .all(|d| shared.ocl_device_info(d.platform, d.device).is_some());
        if !ocl_ok && expect["error"].as_str() != Some("Unable to obtain OpenCL device info") {
            skipped += 1;
            continue;
        }
        let auto_method = match &opts.gf_method {
            Some(m) => method_num_ref(GF_METHODS, m).ok() == Some(0),
            None => false,
        };
        let result = Par2Gen::new(
            &mut files,
            &slice_def,
            opts,
            shared.clone(),
            "utf-8",
            &mut shared,
            &env,
        );
        ran += 1;
        match (expect.get("error"), result) {
            (Some(errv), Err(e)) => {
                let want = errv.as_str().unwrap_or("");
                if e != want {
                    failures.push(format!(
                        "{id}: error mismatch:\n  want: {want}\n  got:  {e}"
                    ));
                }
            }
            (Some(errv), Ok(_)) => failures.push(format!(
                "{id}: JS errored ({}) but Rust succeeded",
                errv.as_str().unwrap_or("?")
            )),
            (None, Err(e)) => failures.push(format!("{id}: Rust errored: {e}")),
            (None, Ok(gen)) => {
                let mut expect_v = expect.clone();
                let mut got = gen.dump();
                if auto_method {
                    scrub_auto_method(&mut expect_v);
                    scrub_auto_method(&mut got);
                }
                let mut errs = Vec::new();
                cmp("out", &expect_v, &got, &mut errs);
                for e in errs {
                    failures.push(format!("{id}: {e}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures (of {ran}, {skipped} skipped):\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

//! Integration tests beyond oracle-equality:
//!
//! - method-invariance: every host-supported GF16 kernel must produce
//!   byte-identical .par2 output (kernels change speed, never bytes)
//! - determinism: identical inputs+opts twice must give identical bytes
//! - independent repair: `par2` (par2cmdline) verifies our output and
//!   recovers missing/damaged inputs - proof against a shared spec
//!   misreading in both our code and the JS oracle
//!
//! All run through the in-process native pipeline (native hasher + native
//! GF16 proc), so no built CLI binary or JS runtime is required.

#![cfg(target_env = "gnu")]
use parpar::args::{MethodRef, SpecTerm, SpecUnit};
use parpar::fileinfo::FileInfo;
use parpar::generate::{GenOpts, Par2Gen, SysEnv};
use parpar::par2::{SharedBackend, GF_METHODS};
use parpar::pipeline::GenDriver;
use parpar_ffi::FfiBackend;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn sys_env() -> SysEnv {
    SysEnv {
        total_mem: u64::MAX,
        free_mem: u64::MAX,
        constrained_mem: u64::MAX,
        is_64b: true,
        k_max_length: u64::MAX,
    }
}

fn file_info_for(dir: &Path, names: &[&str]) -> Vec<FileInfo> {
    let paths: Vec<String> = names
        .iter()
        .map(|n| dir.join(n).to_string_lossy().into_owned())
        .collect();
    parpar::fileinfo::file_info(&paths, parpar::args::Recurse::No, false, 0, None).unwrap()
}

/// Run one generation; returns produced files keyed by basename.
fn generate(
    files: &mut Vec<FileInfo>,
    out_base: &Path,
    gf_method: MethodRef,
    rec_slices: f64,
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    generate_opts(
        files,
        out_base,
        4096.0,
        None,
        rec_slices,
        gf_method,
        true,
        true,
        |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_opts(
    files: &mut Vec<FileInfo>,
    out_base: &Path,
    slice: f64,
    memory_limit: Option<f64>,
    rec_slices: f64,
    gf_method: MethodRef,
    native_hash: bool,
    native_gf: bool,
    extra: impl FnOnce(&mut GenOpts),
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut o = GenOpts::default();
    o.gf_method = Some(gf_method);
    o.output_base = out_base.to_string_lossy().into_owned();
    o.output_overwrite = true;
    o.creator = "ParPar v0.5.0 x64 [https://github.com/animetosho/parpar]".into();
    o.memory_limit = memory_limit;
    o.recovery_slices = vec![SpecTerm {
        unit: SpecUnit::Slices,
        value: rec_slices,
        scale: 1.0,
    }];
    extra(&mut o);

    let env = sys_env();
    let backend = FfiBackend::new().unwrap();
    let mut shared = SharedBackend::new(backend);
    shared.set_native_hashers(native_hash);
    shared.set_native_proc(native_gf);
    let mut gen = Par2Gen::new(
        files,
        &SpecTerm {
            unit: SpecUnit::Bytes,
            value: slice,
            scale: 1.0,
        },
        o,
        shared.clone(),
        "utf-8",
        &mut shared,
        &env,
    )?;
    let mut drv = GenDriver::new(&mut gen).map_err(|e| e.to_string())?;
    drv.run(&mut |_| {}).map_err(|e| e.to_string())?;

    let mut out = BTreeMap::new();
    for rf in &drv.out_files {
        let base = Path::new(&rf.name)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        out.insert(base, std::fs::read(&rf.name).map_err(|e| e.to_string())?);
    }
    Ok(out)
}

fn write_inputs(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    // deterministic content; sizes cross slice boundaries (4 KiB slices)
    let mut rng = 0x9e3779b9u32;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng >> 16) as u8
    };
    std::fs::write(
        dir.join("f1.bin"),
        (0..70000).map(|_| next()).collect::<Vec<u8>>(),
    )
    .unwrap();
    std::fs::write(
        dir.join("f2.bin"),
        (0..20000).map(|_| next()).collect::<Vec<u8>>(),
    )
    .unwrap();
}

#[test]
fn method_invariance() {
    let dir = std::env::temp_dir().join("e2e_method_inv");
    let _ = std::fs::remove_dir_all(&dir);
    write_inputs(&dir.join("in"));

    let mut reference: Option<BTreeMap<String, Vec<u8>>> = None;
    let mut methods_run = 0;
    for id in 0..GF_METHODS.len() as i32 {
        // only kernels usable on this host
        if parpar_gf16::kernel::pick_explicit(id).is_none() {
            continue;
        }
        let out_dir = dir.join(format!("m{id}"));
        std::fs::create_dir_all(&out_dir).unwrap();
        let mut files = file_info_for(&dir.join("in"), &["f1.bin", "f2.bin"]);
        let got = match generate(
            &mut files,
            &out_dir.join("o"),
            MethodRef::Index(id as f64),
            4.0,
        ) {
            Ok(g) => g,
            Err(e) => panic!("method {id} ({}) failed: {e}", GF_METHODS[id as usize]),
        };
        methods_run += 1;
        match &reference {
            None => reference = Some(got),
            Some(r) => assert_eq!(
                *r, got,
                "method {id} ({}) output differs",
                GF_METHODS[id as usize]
            ),
        }
    }
    assert!(
        methods_run >= 2,
        "need >= 2 host-supported kernels for invariance (got {methods_run})"
    );
}

#[test]
fn determinism() {
    let dir = std::env::temp_dir().join("e2e_determinism");
    let _ = std::fs::remove_dir_all(&dir);
    write_inputs(&dir.join("in"));

    let mut runs = Vec::new();
    for run in 0..2 {
        let out_dir = dir.join(format!("r{run}"));
        std::fs::create_dir_all(&out_dir).unwrap();
        let mut files = file_info_for(&dir.join("in"), &["f1.bin", "f2.bin"]);
        runs.push(
            generate(
                &mut files,
                &out_dir.join("o"),
                MethodRef::Name("".into()),
                4.0,
            )
            .unwrap(),
        );
    }
    assert_eq!(runs[0], runs[1], "identical runs produced different bytes");
}

/// Path to the JS CLI entry point: `$PARPAR_JS` override, else repo-
/// relative (crates/parpar -> repo root bin/parpar.js).
fn js_cli() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("PARPAR_JS") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // CARGO_MANIFEST_DIR = rust/crates/parpar -> repo root is ../../..
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../bin/parpar.js");
    p.exists().then_some(p)
}

/// `par2` binary in PATH (par2cmdline) - the independent verifier.
fn par2_bin() -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_string_lossy()
        .split(if cfg!(windows) { ';' } else { ':' })
        .map(|d| Path::new(d).join(if cfg!(windows) { "par2.exe" } else { "par2" }))
        .find(|p| p.is_file())
}

#[test]
fn par2cmdline_repair() {
    let Some(par2) = par2_bin() else {
        eprintln!("par2cmdline not in PATH - skipping independent repair test");
        return;
    };
    let dir = std::env::temp_dir().join("e2e_par2cmd");
    let _ = std::fs::remove_dir_all(&dir);
    write_inputs(&dir);
    // keep pristine copies for post-repair comparison
    let orig1 = std::fs::read(dir.join("f1.bin")).unwrap();
    let orig2 = std::fs::read(dir.join("f2.bin")).unwrap();

    let mut files = file_info_for(&dir, &["f1.bin", "f2.bin"]);
    generate(
        &mut files,
        &dir.join("rep"),
        MethodRef::Name("".into()),
        8.0,
    )
    .unwrap();

    // clean verify must pass
    let st = std::process::Command::new(&par2)
        .arg("v")
        .arg("rep.par2")
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "par2 verify failed on clean set:\n{}",
        String::from_utf8_lossy(&st.stdout)
    );

    // delete f2 (5 slices), damage one slice of f1 - 6 lost vs 8 recovery
    std::fs::remove_file(dir.join("f2.bin")).unwrap();
    std::fs::write(dir.join("f1.bin"), {
        let mut v = orig1.clone();
        for (i, b) in v.iter_mut().take(4000).enumerate() {
            *b = (i * 31 & 0xff) as u8;
        }
        v
    })
    .unwrap();
    let st = std::process::Command::new(&par2)
        .arg("r")
        .arg("rep.par2")
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "par2 repair failed:\n{}\n{}",
        String::from_utf8_lossy(&st.stdout),
        String::from_utf8_lossy(&st.stderr)
    );
    assert_eq!(
        std::fs::read(dir.join("f1.bin")).unwrap(),
        orig1,
        "f1.bin damaged"
    );
    assert_eq!(
        std::fs::read(dir.join("f2.bin")).unwrap(),
        orig2,
        "f2.bin damaged"
    );
}

/// Forced multi-pass recovery (256 slices over a 512 KiB limit, the
/// mem_limit_min oracle shape): FFI pipeline and native pipeline must
/// produce byte-identical output.
#[test]
fn multi_pass_differential() {
    let dir = std::env::temp_dir().join("e2e_multipass");
    let _ = std::fs::remove_dir_all(&dir);
    write_inputs(&dir.join("in"));

    let mut results = Vec::new();
    for (i, (nh, ng)) in [(false, false), (true, true)].iter().enumerate() {
        let out_dir = dir.join(format!("p{i}"));
        std::fs::create_dir_all(&out_dir).unwrap();
        let mut files = file_info_for(&dir.join("in"), &["f1.bin", "f2.bin"]);
        results.push(
            generate_opts(
                &mut files,
                &out_dir.join("mp"),
                8192.0,
                Some(524288.0),
                256.0,
                MethodRef::Name("".into()),
                *nh,
                *ng,
                |_| {},
            )
            .unwrap(),
        );
    }
    assert_eq!(
        results[0], results[1],
        "multi-pass output differs between FFI and native pipelines"
    );
}

/// Edge-input tree through the native pipeline, verified independently by
/// par2cmdline. Includes a unicode filename - exercises the non-ASCII
/// packet path against a real decoder. (0-byte files and `/` in recorded
/// names are covered by the golden/differential tests instead: par2cmdline
/// misreports both even on upstream-JS output.)
#[test]
fn edge_inputs_par2() {
    let Some(par2) = par2_bin() else {
        eprintln!("par2cmdline not in PATH - skipping edge-input verify");
        return;
    };
    let dir = std::env::temp_dir().join("e2e_edge_par2");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("exact.bin"), vec![0xabu8; 4096]).unwrap();
    std::fs::write(dir.join("tiny.bin"), vec![7u8; 1000]).unwrap();
    std::fs::write(dir.join("uni-\u{20ac}.bin"), vec![3u8; 9000]).unwrap();

    let names = ["exact.bin", "tiny.bin", "uni-\u{20ac}.bin"];
    let mut files = file_info_for(&dir, &names);
    generate(
        &mut files,
        &dir.join("edge"),
        MethodRef::Name("".into()),
        4.0,
    )
    .unwrap();

    let st = std::process::Command::new(&par2)
        .arg("v")
        .arg("edge.par2")
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "par2 verify failed on edge-input set:\n{}\n{}",
        String::from_utf8_lossy(&st.stdout),
        String::from_utf8_lossy(&st.stderr)
    );
}

/// Seeded randomized differential: random file trees x random option
/// combos through the FFI pipeline vs the native pipeline. Byte-identical
/// output on success; identical error text on failure.
#[test]
fn randomized_differential() {
    let mut rng = 0xC0FFEEu64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let pick = |n: u64, r: &mut dyn FnMut() -> u64| (r() % n) as usize;
    let sizes = [0usize, 500, 4096, 5000, 40000, 130000];
    let slices = [512.0, 1024.0, 4096.0, 8192.0, 16384.0];
    let mems = [None, Some(16384.0), Some(65536.0), Some(262144.0)];
    let schemes = ["pow2", "uniform", "equal"];
    let disps = ["common", "basename", "keep", "path"];
    let unicodes = [None, Some(true), Some(false)];

    for case in 0..40u32 {
        let dir = std::env::temp_dir().join(format!("e2e_rand_{case}"));
        let _ = std::fs::remove_dir_all(&dir);
        let in_dir = dir.join("in");
        std::fs::create_dir_all(&in_dir).unwrap();

        let n_files = 1 + pick(4, &mut next);
        let mut names = Vec::new();
        for fi in 0..n_files {
            let name = if case % 5 == 4 && fi == 0 {
                format!("u\u{20ac}{fi}.bin")
            } else {
                format!("f{fi}.bin")
            };
            let size = sizes[pick(6, &mut next)];
            let data: Vec<u8> = (0..size).map(|_| (next() >> 33) as u8).collect();
            std::fs::write(in_dir.join(&name), data).unwrap();
            names.push(name);
        }
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        let slice = slices[pick(5, &mut next)];
        let rec = (next() % 13) as f64;
        let mem = mems[pick(4, &mut next)];
        let scheme = schemes[pick(3, &mut next)];
        let disp = disps[pick(4, &mut next)];
        let uni = unicodes[pick(3, &mut next)];
        let offset = (next() % 5) as i32;
        let comment = if case % 7 == 3 {
            Some(format!("cmt{case}"))
        } else {
            None
        };

        let mut results = Vec::new();
        for (i, (nh, ng)) in [(false, false), (true, true)].iter().enumerate() {
            let out_dir = dir.join(format!("o{i}"));
            std::fs::create_dir_all(&out_dir).unwrap();
            let mut files = file_info_for(&in_dir, &name_refs);
            let comment = comment.clone();
            results.push(generate_opts(
                &mut files,
                &out_dir.join("r"),
                slice,
                mem,
                rec,
                MethodRef::Name("".into()),
                *nh,
                *ng,
                move |o| {
                    o.output_size_scheme = scheme.into();
                    o.display_name_format = disp.into();
                    o.unicode = uni;
                    o.recovery_offset = offset;
                    if let Some(c) = comment {
                        o.comments = vec![c];
                    }
                },
            ));
        }
        match (&results[0], &results[1]) {
            (Ok(ffi), Ok(native)) => assert_eq!(
                ffi, native,
                "case {case}: FFI vs native output differs (slice={slice} rec={rec} mem={mem:?} scheme={scheme} disp={disp} uni={uni:?} off={offset})"
            ),
            (Err(a), Err(b)) => assert_eq!(
                a, b,
                "case {case}: error text differs: FFI `{a}` vs native `{b}`"
            ),
            (a, b) => panic!(
                "case {case}: outcome mismatch (slice={slice} rec={rec} mem={mem:?}): FFI {a:?} vs native {b:?}",
                a = a.as_ref().map(|_| "ok"),
                b = b.as_ref().map(|_| "ok")
            ),
        }
    }
}

/// CLI-level error parity vs the JS CLI: identical exit codes and stderr
/// for missing files, bad specs, unknown flags. Skips without node.
#[test]
fn cli_error_parity() {
    if js_cli().is_none() {
        eprintln!("no JS CLI - skipping CLI error parity");
        return;
    }
    let rust_bin = std::path::PathBuf::from(env!("CARGO_BIN_EXE_parpar"));
    let js = js_cli().unwrap();
    let dir = std::env::temp_dir().join("e2e_clierr");
    std::fs::create_dir_all(&dir).unwrap();

    let cases: Vec<Vec<String>> = vec![
        // missing input file
        vec!["-q", "-s4K", "-r3", "-o", "x", "nope.bin"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        // missing slice spec
        vec!["-q", "-o", "x", "nope.bin"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        // bad slice spec value
        vec!["-q", "-sbanana", "-r3", "-o", "x", "nope.bin"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        // unknown flag
        vec![
            "-q",
            "--definitely-not-an-option",
            "-s4K",
            "-r3",
            "-o",
            "x",
            "nope.bin",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        // unknown gf method
        vec!["-q", "-s4K", "-r3", "-mnope", "-o", "x", "nope.bin"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    ];

    for (i, args) in cases.iter().enumerate() {
        let js_out = std::process::Command::new("node")
            .arg(&js)
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap();
        let rs_out = std::process::Command::new(&rust_bin)
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert_eq!(
            js_out.status.code(),
            rs_out.status.code(),
            "case {i} {args:?}: exit code differs (js={:?} rust={:?})\njs stderr: {}\nrs stderr: {}",
            js_out.status.code(),
            rs_out.status.code(),
            String::from_utf8_lossy(&js_out.stderr),
            String::from_utf8_lossy(&rs_out.stderr)
        );
        assert_eq!(
            js_out.stderr,
            rs_out.stderr,
            "case {i} {args:?}: stderr differs\njs: {}\nrs: {}",
            String::from_utf8_lossy(&js_out.stderr),
            String::from_utf8_lossy(&rs_out.stderr)
        );
    }
}

/// Regression: the CLI used to drop the GenOpts default
/// max_recovery_slices {slices:65536}, so any non-integral -r spec
/// (percent, bytes, power, file-based) failed with "Could not satisfy
/// min/max recovery slice count constraints". Drives the real binaries
/// and requires byte-identical output.
#[test]
fn cli_recovery_specs() {
    let Some(js) = js_cli() else {
        eprintln!("node/JS CLI unavailable - skipping CLI spec parity");
        return;
    };
    let rust_bin = std::path::PathBuf::from(env!("CARGO_BIN_EXE_parpar"));
    let dir = std::env::temp_dir().join("e2e_clispec");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("in.bin"), vec![0xabu8; 100_000]).unwrap();

    for rspec in ["10%", "2K", "0.5w"].iter() {
        let js_out = std::process::Command::new("node")
            .arg(&js)
            .args(["-q", "-O", "-s4K", "-r"])
            .arg(rspec)
            .args(["-o", "js", "in.bin"])
            .current_dir(&dir)
            .output()
            .unwrap();
        let rs_out = std::process::Command::new(&rust_bin)
            .args(["-q", "-O", "-s4K", "-r"])
            .arg(rspec)
            .args(["-o", "rs", "in.bin"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert_eq!(
            js_out.status.code(),
            rs_out.status.code(),
            "-r{rspec}: exit differs (js={:?} rust={:?})\nrs stderr: {}",
            js_out.status.code(),
            rs_out.status.code(),
            String::from_utf8_lossy(&rs_out.stderr)
        );
        assert!(
            rs_out.status.success(),
            "-r{rspec}: rust CLI failed\n{}",
            String::from_utf8_lossy(&rs_out.stderr)
        );
        // every generated file must be byte-identical to the JS oracle's
        for ent in std::fs::read_dir(&dir).unwrap() {
            let name = ent.unwrap().file_name().into_string().unwrap();
            if let Some(rest) = name.strip_prefix("js") {
                let ours = dir.join(format!("rs{rest}"));
                assert!(
                    ours.exists(),
                    "-r{rspec}: missing output {ours:?} for js file {name}"
                );
                assert_eq!(
                    std::fs::read(dir.join(&name)).unwrap(),
                    std::fs::read(&ours).unwrap(),
                    "-r{rspec}: {name} differs"
                );
            }
        }
    }
}

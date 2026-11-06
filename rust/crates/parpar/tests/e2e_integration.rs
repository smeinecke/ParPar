//! Integration tests beyond oracle-equality - pure Rust, runs on every
//! platform the crate builds for:
//!
//! - method-invariance: every host-supported GF16 kernel must produce
//!   byte-identical .par2 output (kernels change speed, never bytes)
//! - determinism / thread-invariance: identical inputs+opts must give
//!   identical bytes regardless of run or worker count
//! - independent repair: `par2` (par2cmdline) verifies our output and
//!   recovers missing/damaged inputs - proof against a shared spec
//!   misreading in both our code and the JS oracle
//! - CLI parity vs the JS CLI wherever node + the native addon exist
//!
//! In-process tests drive SharedBackend<NativeBackend> - the same wiring
//! the --no-default-features binary uses. FFI-vs-native differentials
//! live in e2e_differential.rs (x86_64-gnu only).

mod common;
use common::*;

use parpar::args::MethodRef;
use parpar::par2::GF_METHODS;
use std::collections::BTreeMap;
use std::path::PathBuf;

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

/// num_threads changes scheduling, never bytes: 1 vs 8 workers must
/// produce identical output. Covers the multi-threaded staging paths
/// (packed staging buffers, write queue ordering) for byte-identity.
#[test]
fn thread_invariance() {
    let dir = std::env::temp_dir().join("e2e_thread_inv");
    let _ = std::fs::remove_dir_all(&dir);
    write_inputs(&dir.join("in"));

    let mut runs = Vec::new();
    for (i, t) in [1i32, 8].iter().enumerate() {
        let out_dir = dir.join(format!("t{i}"));
        std::fs::create_dir_all(&out_dir).unwrap();
        let mut files = file_info_for(&dir.join("in"), &["f1.bin", "f2.bin"]);
        runs.push(
            generate_opts(
                native_shared(),
                &mut files,
                &out_dir.join("o"),
                4096.0,
                None,
                4.0,
                MethodRef::Name("".into()),
                |o| o.num_threads = Some(*t),
            )
            .unwrap(),
        );
    }
    assert_eq!(runs[0], runs[1], "thread counts produced different bytes");
}

/// 0-byte and 1-byte inputs through the native pipeline. par2cmdline
/// misreports empty files even for upstream-JS output, so assert the
/// pipeline succeeds and stays deterministic instead of verifying.
#[test]
fn empty_and_tiny_inputs() {
    let dir = std::env::temp_dir().join("e2e_empty_tiny");
    let _ = std::fs::remove_dir_all(&dir);
    let in_dir = dir.join("in");
    std::fs::create_dir_all(&in_dir).unwrap();
    std::fs::write(in_dir.join("empty.bin"), Vec::<u8>::new()).unwrap();
    std::fs::write(in_dir.join("one.bin"), vec![0xabu8]).unwrap();
    std::fs::write(in_dir.join("tiny.bin"), vec![7u8; 3]).unwrap();

    let names = ["empty.bin", "one.bin", "tiny.bin"];
    let mut runs = Vec::new();
    for r in 0..2 {
        let out_dir = dir.join(format!("r{r}"));
        std::fs::create_dir_all(&out_dir).unwrap();
        let mut files = file_info_for(&in_dir, &names);
        runs.push(
            generate(
                &mut files,
                &out_dir.join("et"),
                MethodRef::Name("".into()),
                2.0,
            )
            .unwrap(),
        );
    }
    assert!(
        runs[0].keys().any(|k| k.ends_with(".par2")),
        "no .par2 produced for empty/tiny input set"
    );
    assert_eq!(
        runs[0], runs[1],
        "empty/tiny generation is not deterministic"
    );
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

/// CLI-level error parity vs the JS CLI: identical exit codes and stderr
/// for missing files, bad specs, unknown flags. Skips without node.
#[test]
fn cli_error_parity() {
    if js_cli().is_none() {
        eprintln!("no JS CLI - skipping CLI error parity");
        return;
    }
    let rust_bin = PathBuf::from(env!("CARGO_BIN_EXE_parpar"));
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
        let (js_out, rs_out) = cli_pair(&js, &rust_bin, args, &dir);
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
    let rust_bin = PathBuf::from(env!("CARGO_BIN_EXE_parpar"));
    let dir = std::env::temp_dir().join("e2e_clispec");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("in.bin"), vec![0xabu8; 100_000]).unwrap();

    for rspec in ["10%", "2K", "0.5w"].iter() {
        let js_args: Vec<String> = ["-q", "-O", "-s4K", "-r", rspec, "-o", "js", "in.bin"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rs_args: Vec<String> = ["-q", "-O", "-s4K", "-r", rspec, "-o", "rs", "in.bin"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let js_out = std::process::Command::new("node")
            .arg(&js)
            .args(&js_args)
            .current_dir(&dir)
            .output()
            .unwrap();
        let rs_out = std::process::Command::new(&rust_bin)
            .args(&rs_args)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert_cli_outputs_identical(
            &dir,
            "js",
            "rs",
            &js_out,
            &rs_out,
            &format!("-r{rspec}"),
        );
        // clean up for the next spec iteration
        for ent in std::fs::read_dir(&dir).unwrap() {
            let p = ent.unwrap().path();
            if p.file_name().unwrap().to_string_lossy() != "in.bin" {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
}

/// CLI option matrix: a broad spread of generation flags driven through
/// both binaries; every emitted file must be byte-identical to the JS
/// oracle's. Covers volume splitting, naming schemes, recorded-name
/// formats, comments, unicode packets and thread-count independence in
/// one sweep. Skips without node+addon.
#[test]
fn cli_option_matrix() {
    let Some(js) = js_cli() else {
        eprintln!("node/JS CLI unavailable - skipping CLI option matrix");
        return;
    };
    let rust_bin = PathBuf::from(env!("CARGO_BIN_EXE_parpar"));
    let dir = std::env::temp_dir().join("e2e_climatrix");
    let _ = std::fs::remove_dir_all(&dir);
    let in_dir = dir.join("in");
    std::fs::create_dir_all(&in_dir).unwrap();

    // deterministic inputs incl. a nested path for -f/-B cases
    let mut rng = 0xB16B00B5u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    std::fs::create_dir_all(in_dir.join("sub")).unwrap();
    for (name, size) in [
        ("f1.bin", 70000usize),
        ("f2.bin", 20000usize),
        ("sub/deep.bin", 9000usize),
    ] {
        std::fs::write(
            in_dir.join(name),
            (0..size).map(|_| (next() >> 33) as u8).collect::<Vec<u8>>(),
        )
        .unwrap();
    }
    let inputs: Vec<String> = ["f1.bin", "f2.bin", "sub/deep.bin"]
        .iter()
        .map(|n| in_dir.join(n).to_string_lossy().into_owned())
        .collect();
    let base = in_dir.to_string_lossy().into_owned();

    let cases: &[&[&str]] = &[
        // small slices forced into multiple volumes
        &["-s512B", "-r3", "-F2"],
        // percent spec + uniform/equal distribution schemes
        &["-s4K", "-r10%", "-duniform"],
        &["-s4K", "-r3", "-dequal"],
        // recorded-name formats and base-path handling
        &["-s4K", "-r3", "-fkeep"],
        &["-s4K", "-r3", "-fpath"],
        &["-s4K", "-r3", "-fcommon"],
        // comment packet
        &["-s4K", "-r3", "-c", "matrix comment"],
        // thread count must not alter bytes
        &["-s4K", "-r3", "-t1"],
        // recovery offset + first-file slice distribution
        &["-s4K", "-r3", "-e2"],
        &["-s4K", "-r3", "--slices-first-file", "2"],
        // explicit volume count over several recovery slices
        &["-s4K", "-r6", "-F3"],
        // byte-quantity and power specs
        &["-s4K", "-r2K"],
        &["-s4K", "-r0.5w"],
        // no index file, alternate volume naming
        &["-s4K", "-r3", "--noindex"],
        &["-s4K", "-r3", "--no-std-naming"],
        // forced unicode packets
        &["-s4K", "-r3", "--unicode"],
    ];

    let mut cases: Vec<Vec<String>> = cases
        .iter()
        .map(|c| c.iter().map(|s| s.to_string()).collect())
        .collect();
    // filepath-base with an absolute base dir
    cases.push(vec![
        "-s4K".into(),
        "-r3".into(),
        "-fpath".into(),
        "-B".into(),
        base.clone(),
    ]);

    for (i, opts) in cases.iter().enumerate() {
        // fresh output namespace per case
        let case_dir = dir.join(format!("c{i}"));
        std::fs::create_dir_all(&case_dir).unwrap();
        let mut args: Vec<String> = vec!["-q".into(), "-O".into()];
        args.extend(opts.iter().cloned());
        args.extend(["-o".into(), "j".into()]);
        args.extend(inputs.iter().cloned());

        let js_out = std::process::Command::new("node")
            .arg(&js)
            .args(&args)
            .current_dir(&case_dir)
            .output()
            .unwrap();
        let mut rs_args = args.clone();
        *rs_args.iter_mut().find(|a| *a == "j").unwrap() = "r".into();
        let rs_out = std::process::Command::new(&rust_bin)
            .args(&rs_args)
            .current_dir(&case_dir)
            .output()
            .unwrap();
        assert_cli_outputs_identical(
            &case_dir,
            "j",
            "r",
            &js_out,
            &rs_out,
            &format!("case {i} {opts:?}"),
        );
    }
}

/// Edge-size inputs vs the JS CLI: empty, 1-byte, slice-boundary sizes and
/// an unaligned tail must all produce byte-identical .par2 sets.
/// par2cmdline misreports several of these even for upstream JS output,
/// so the JS oracle is the arbiter here.
#[test]
fn cli_edge_inputs() {
    let Some(js) = js_cli() else {
        eprintln!("node/JS CLI unavailable - skipping CLI edge inputs");
        return;
    };
    let rust_bin = PathBuf::from(env!("CARGO_BIN_EXE_parpar"));
    let dir = std::env::temp_dir().join("e2e_cliedge");
    let _ = std::fs::remove_dir_all(&dir);
    let in_dir = dir.join("in");
    std::fs::create_dir_all(&in_dir).unwrap();

    for (name, size) in [
        ("empty.bin", 0usize),
        ("one.bin", 1),
        ("below.bin", 4095),
        ("exact.bin", 4096),
        ("over.bin", 4097),
        ("big.bin", 131_073),
    ] {
        let mut v = Vec::with_capacity(size);
        for i in 0..size {
            v.push((i * 37 % 251) as u8);
        }
        std::fs::write(in_dir.join(name), v).unwrap();
    }
    let inputs = [
        "in/empty.bin",
        "in/one.bin",
        "in/below.bin",
        "in/exact.bin",
        "in/over.bin",
        "in/big.bin",
    ];

    let js_args: Vec<String> = ["-q", "-O", "-s4K", "-r3", "-o", "j"]
        .iter()
        .map(|s| s.to_string())
        .chain(inputs.iter().map(|s| s.to_string()))
        .collect();
    let rs_args: Vec<String> = ["-q", "-O", "-s4K", "-r3", "-o", "r"]
        .iter()
        .map(|s| s.to_string())
        .chain(inputs.iter().map(|s| s.to_string()))
        .collect();
    let js_out = std::process::Command::new("node")
        .arg(&js)
        .args(&js_args)
        .current_dir(&dir)
        .output()
        .unwrap();
    let rs_out = std::process::Command::new(&rust_bin)
        .args(&rs_args)
        .current_dir(&dir)
        .output()
        .unwrap();
    assert_cli_outputs_identical(&dir, "j", "r", &js_out, &rs_out, "edge inputs");
}

/// Coarse pipeline performance requirement: an 8 MiB / 64 KiB-slice /
/// 8-recovery-slice generation through the native pipeline must finish
/// well under this bound on any supported host. Not a benchmark - the
/// floor catches pathological regressions (accidentally-serial stages,
/// quadratic paths) while staying far above real times (~1 s or less).
/// Kernel-level throughput floors live in parpar-gf16/parpar-hasher
/// perf_gate tests.
#[test]
fn pipeline_perf_smoke() {
    if cfg!(debug_assertions) {
        eprintln!("debug build - skipping pipeline perf smoke");
        return;
    }
    if std::env::var("PARPAR_PERF").map(|v| v == "0").unwrap_or(false) {
        eprintln!("PARPAR_PERF=0 - skipping pipeline perf smoke");
        return;
    }
    let dir = std::env::temp_dir().join("e2e_perf_smoke");
    let _ = std::fs::remove_dir_all(&dir);
    let in_dir = dir.join("in");
    std::fs::create_dir_all(&in_dir).unwrap();
    let mut rng = 0x5EEDu64;
    std::fs::write(
        in_dir.join("data.bin"),
        (0..8 << 20)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 33) as u8
            })
            .collect::<Vec<u8>>(),
    )
    .unwrap();

    let t = std::time::Instant::now();
    let mut files = file_info_for(&in_dir, &["data.bin"]);
    generate_opts(
        native_shared(),
        &mut files,
        &dir.join("perf"),
        65536.0,
        None,
        8.0,
        MethodRef::Name("".into()),
        |_| {},
    )
    .unwrap();
    let el = t.elapsed().as_secs_f64();
    eprintln!("pipeline_perf_smoke: 8MiB gen in {el:.2}s");
    assert!(
        el < 15.0,
        "8 MiB generation took {el:.1}s - exceeds the 15 s bound"
    );
}

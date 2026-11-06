//! Shared helpers for the parpar integration suite.
//!
//! Two backend wirings exist:
//! - `native_shared()` wraps `NativeBackend` with native hashers/proc -
//!   the exact wiring the `--no-default-features` production binary
//!   uses, available on every platform
//! - `ffi_shared()` wraps the C++ oracle - only where `parpar-ffi` is a
//!   dev-dep (x86_64-gnu, see Cargo.toml); FFI-vs-native differentials
//!   live in e2e_differential.rs
#![allow(dead_code)]

use parpar::args::{MethodRef, SpecTerm, SpecUnit};
use parpar::fileinfo::FileInfo;
use parpar::generate::{GenOpts, Par2Gen, SysEnv};
use parpar::native_backend::NativeBackend;
use parpar::par2::SharedBackend;
use parpar::pipeline::GenDriver;
use parpar_backend::ComputeBackend;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn sys_env() -> SysEnv {
    SysEnv {
        total_mem: u64::MAX,
        free_mem: u64::MAX,
        constrained_mem: u64::MAX,
        is_64b: true,
        k_max_length: u64::MAX,
    }
}

pub fn file_info_for(dir: &Path, names: &[&str]) -> Vec<FileInfo> {
    let paths: Vec<String> = names
        .iter()
        .map(|n| dir.join(n).to_string_lossy().into_owned())
        .collect();
    parpar::fileinfo::file_info(&paths, parpar::args::Recurse::No, false, 0, None).unwrap()
}

/// Pure-Rust backend wiring - mirrors main.rs for --no-default-features.
pub fn native_shared() -> SharedBackend<NativeBackend> {
    let mut shared = SharedBackend::new(NativeBackend::new());
    shared.set_native_hashers(true);
    shared.set_native_proc(true);
    shared
}

/// C++ oracle backend; `native_hash`/`native_gf` select which stages run
/// as Rust. x86_64-gnu only - parpar-ffi is a target-gated dev-dep.
#[cfg(all(target_env = "gnu", target_arch = "x86_64"))]
pub fn ffi_shared(native_hash: bool, native_gf: bool) -> SharedBackend<parpar_ffi::FfiBackend> {
    let mut shared = SharedBackend::new(parpar_ffi::FfiBackend::new().unwrap());
    shared.set_native_hashers(native_hash);
    shared.set_native_proc(native_gf);
    shared
}

/// Run one generation through `shared`; returns produced files keyed by
/// basename.
#[allow(clippy::too_many_arguments)]
pub fn generate_opts<B>(
    shared: SharedBackend<B>,
    files: &mut Vec<FileInfo>,
    out_base: &Path,
    slice: f64,
    memory_limit: Option<f64>,
    rec_slices: f64,
    gf_method: MethodRef,
    extra: impl FnOnce(&mut GenOpts),
) -> Result<BTreeMap<String, Vec<u8>>, String>
where
    B: ComputeBackend,
    B::InputHasher: 'static,
    B::OutputHasher: 'static,
    B::Proc: 'static,
{
    let mut o = GenOpts::default();
    o.gf_method = Some(gf_method);
    o.output_base = out_base.to_string_lossy().into_owned();
    o.output_overwrite = true;
    o.creator = "ParPar v0.5.3 x64 [https://github.com/animetosho/parpar]".into();
    o.memory_limit = memory_limit;
    o.recovery_slices = vec![SpecTerm {
        unit: SpecUnit::Slices,
        value: rec_slices,
        scale: 1.0,
    }];
    extra(&mut o);

    let env = sys_env();
    let mut shared = shared;
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

/// Convenience: native pipeline, 4 KiB slices, N recovery slices.
pub fn generate(
    files: &mut Vec<FileInfo>,
    out_base: &Path,
    gf_method: MethodRef,
    rec_slices: f64,
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    generate_opts(
        native_shared(),
        files,
        out_base,
        4096.0,
        None,
        rec_slices,
        gf_method,
        |_| {},
    )
}

/// Deterministic two-file input tree; sizes cross 4 KiB slice boundaries.
pub fn write_inputs(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
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

/// Path to the JS CLI entry point: `$PARPAR_JS` override, else repo-
/// relative. Requires the compiled addon - without it parpar.js dies on
/// MODULE_NOT_FOUND instead of behaving like the oracle, so parity tests
/// must skip rather than compare a crash trace.
pub fn js_cli() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PARPAR_JS") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // CARGO_MANIFEST_DIR = rust/crates/parpar -> repo root is ../../..
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let p = root.join("bin/parpar.js");
    let addon = root.join("build/Release/parpar_gf.node");
    (p.exists() && addon.exists()).then_some(p)
}

/// `par2` binary in PATH (par2cmdline) - the independent verifier.
pub fn par2_bin() -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_string_lossy()
        .split(if cfg!(windows) { ';' } else { ':' })
        .map(|d| Path::new(d).join(if cfg!(windows) { "par2.exe" } else { "par2" }))
        .find(|p| p.is_file())
}

/// Spawn `node <js>` and `rust_bin` with the same args in `dir`; returns
/// (js_output, rust_output).
pub fn cli_pair(
    js: &Path,
    rust_bin: &Path,
    args: &[String],
    dir: &Path,
) -> (std::process::Output, std::process::Output) {
    let js_out = std::process::Command::new("node")
        .arg(js)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    let rs_out = std::process::Command::new(rust_bin)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    (js_out, rs_out)
}

/// Assert exit-code parity and byte-identity of every file the JS run
/// produced under `dir` with prefix `jpre` against our `rpre` files.
pub fn assert_cli_outputs_identical(
    dir: &Path,
    jpre: &str,
    rpre: &str,
    js_out: &std::process::Output,
    rs_out: &std::process::Output,
    ctx: &str,
) {
    assert_eq!(
        js_out.status.code(),
        rs_out.status.code(),
        "{ctx}: exit differs (js={:?} rust={:?})\njs stderr: {}\nrs stderr: {}",
        js_out.status.code(),
        rs_out.status.code(),
        String::from_utf8_lossy(&js_out.stderr),
        String::from_utf8_lossy(&rs_out.stderr)
    );
    assert!(
        rs_out.status.success(),
        "{ctx}: rust CLI failed\n{}",
        String::from_utf8_lossy(&rs_out.stderr)
    );
    let mut compared = 0;
    for ent in std::fs::read_dir(dir).unwrap() {
        let name = ent.unwrap().file_name().into_string().unwrap();
        if let Some(rest) = name.strip_prefix(jpre) {
            let ours = dir.join(format!("{rpre}{rest}"));
            assert!(ours.exists(), "{ctx}: missing output {ours:?} for {name}");
            assert_eq!(
                std::fs::read(dir.join(&name)).unwrap(),
                std::fs::read(&ours).unwrap(),
                "{ctx}: {name} differs"
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "{ctx}: no js* output files found");
}

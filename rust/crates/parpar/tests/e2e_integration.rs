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
    let mut o = GenOpts::default();
    o.gf_method = Some(gf_method);
    o.output_base = out_base.to_string_lossy().into_owned();
    o.output_overwrite = true;
    o.recovery_slices = vec![SpecTerm {
        unit: SpecUnit::Slices,
        value: rec_slices,
        scale: 1.0,
    }];

    let env = sys_env();
    let backend = FfiBackend::new().unwrap();
    let mut shared = SharedBackend::new(backend);
    shared.set_native_hashers(true);
    shared.set_native_proc(true);
    let mut gen = Par2Gen::new(
        files,
        &SpecTerm {
            unit: SpecUnit::Bytes,
            value: 4096.0,
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

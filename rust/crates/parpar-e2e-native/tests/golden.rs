//! Golden-output e2e for the native pipeline - runs on EVERY platform
//! (no FFI, no C++ toolchain), unlike the gnu-only differential suite.
//!
//! .par2 output is fully deterministic: fixed input bytes + fixed opts
//! must give identical bytes on Linux, Windows, macOS, ARM, RISC-V.
//! `golden.sha256` pins each produced file's digest, so a platform that
//! diverges (endianness, path separators, unicode packet encoding,
//! kernel-specific ordering) fails here.

use parpar::args::{MethodRef, SpecTerm, SpecUnit};
use parpar::fileinfo::FileInfo;
use parpar::generate::{GenOpts, Par2Gen, SysEnv};
use parpar::native_backend::NativeBackend;
use parpar::par2::SharedBackend;
use parpar::pipeline::GenDriver;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const CREATOR: &str = "ParPar v0.4.6 x64 [https://github.com/animetosho/parpar]";
const SLICE: f64 = 4096.0;

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

fn generate(
    files: &mut Vec<FileInfo>,
    out_base: &Path,
    slice: f64,
    memory_limit: Option<f64>,
    rec_slices: f64,
) -> (Par2Gen<SharedBackend<NativeBackend>>, Vec<PathBuf>) {
    let mut o = GenOpts::default();
    o.gf_method = Some(MethodRef::Name("".into()));
    o.output_base = out_base.to_string_lossy().into_owned();
    o.output_overwrite = true;
    o.creator = CREATOR.into();
    o.memory_limit = memory_limit;
    o.recovery_slices = vec![SpecTerm {
        unit: SpecUnit::Slices,
        value: rec_slices,
        scale: 1.0,
    }];

    let env = sys_env();
    let backend = NativeBackend::new();
    let mut shared = SharedBackend::new(backend);
    shared.set_native_hashers(true);
    shared.set_native_proc(true);
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
    )
    .unwrap();
    let mut drv = GenDriver::new(&mut gen).unwrap();
    drv.run(&mut |_| {}).unwrap();
    let outs = drv
        .out_files
        .iter()
        .map(|f| PathBuf::from(&f.name))
        .collect();
    (gen, outs)
}

fn sha256_hex(p: &Path) -> String {
    let mut h = Sha256::new();
    h.update(std::fs::read(p).unwrap());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// `name  sha256` lines for a produced file set, sorted by name.
fn manifest_for(outs: &[PathBuf]) -> String {
    outs.iter()
        .map(|p| {
            format!(
                "{}  {}",
                p.file_name().unwrap().to_string_lossy(),
                sha256_hex(p)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Deterministic byte generator (xorshift32) - identical on all targets.
fn fill(n: usize, mut s: u32) -> Vec<u8> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect()
}

fn write_edge_inputs(dir: &Path) {
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("empty.bin"), []).unwrap(); // 0 bytes
    std::fs::write(dir.join("exact.bin"), fill(SLICE as usize, 7)).unwrap(); // == slice
    std::fs::write(dir.join("tiny.bin"), fill(1000, 11)).unwrap(); // < 16k hash block
    std::fs::write(dir.join("uni-\u{20ac}.bin"), fill(9000, 13)).unwrap(); // unicode name
    std::fs::write(dir.join("sub/nested.bin"), fill(30000, 17)).unwrap(); // subdir
}

fn edge_names() -> [&'static str; 5] {
    [
        "empty.bin",
        "exact.bin",
        "tiny.bin",
        "uni-\u{20ac}.bin",
        "sub/nested.bin",
    ]
}

/// Load `golden.sha256` (committed) as name->digest map.
fn golden_manifest(name: &str) -> BTreeMap<String, String> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (n, h) = l.split_once("  ").expect("manifest line '<name>  <hex>'");
            (n.to_string(), h.to_string())
        })
        .collect()
}

/// Edge-input set incl. a forced multi-pass run - full byte-equality vs
/// the committed digest manifest. `UPDATE_GOLDEN=1` regenerates the
/// manifests (e.g. after an intentional output-format change).
#[test]
fn golden_edge_and_multipass() {
    let dir = std::env::temp_dir().join("e2e_native_golden");
    let _ = std::fs::remove_dir_all(&dir);
    let in_dir = dir.join("in");
    write_edge_inputs(&in_dir);
    let update = std::env::var_os("UPDATE_GOLDEN").is_some();
    let mdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");

    // single pass
    let mut files = file_info_for(&in_dir, &edge_names());
    let (_g, outs) = generate(&mut files, &dir.join("gold"), SLICE, None, 4.0);
    let got = manifest_for(&outs);
    if update {
        std::fs::write(mdir.join("golden.sha256"), &got).unwrap();
    } else {
        let want = golden_manifest("golden.sha256");
        let got_map: BTreeMap<String, String> = got
            .lines()
            .map(|l| l.split_once("  ").unwrap())
            .map(|(n, h)| (n.to_string(), h.to_string()))
            .collect();
        assert_eq!(
            want, got_map,
            "golden mismatch - run UPDATE_GOLDEN=1 to regenerate"
        );
    }

    // forced multi-pass: 256 recovery slices over a 512 KiB limit
    // (same shape as the mem_limit_min oracle case -> 7 passes)
    let mut files = file_info_for(&in_dir, &edge_names());
    let (g, outs) = generate(&mut files, &dir.join("mp"), 8192.0, Some(524288.0), 256.0);
    assert!(g.passes > 1.0, "expected multi-pass, got {}", g.passes);
    let got = manifest_for(&outs);
    if update {
        std::fs::write(mdir.join("golden_mp.sha256"), &got).unwrap();
    } else {
        let want = golden_manifest("golden_mp.sha256");
        let got_map: BTreeMap<String, String> = got
            .lines()
            .map(|l| l.split_once("  ").unwrap())
            .map(|(n, h)| (n.to_string(), h.to_string()))
            .collect();
        assert_eq!(want, got_map, "multi-pass golden mismatch");
    }
}

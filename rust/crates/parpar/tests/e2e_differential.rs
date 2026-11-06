//! FFI-vs-native differentials - x86_64-gnu only (the parpar-ffi dev-dep
//! is target-gated to where the C++ oracle builds).
//!
//! Every case runs the full C++ pipeline (FFI backend, no native
//! stages) AND the pure-Rust pipeline (SharedBackend<NativeBackend>,
//! the wiring the shipped pure-Rust binary uses) and demands
//! byte-identical output or identical error text - the end-to-end
//! property that the pure binary matches the oracle, solver and host
//! answers included, not just kernel-for-kernel inside the FFI shell.

#![cfg(all(target_env = "gnu", target_arch = "x86_64"))]

mod common;
use common::*;

use parpar::args::MethodRef;
use parpar::fileinfo::FileInfo;
use std::collections::BTreeMap;
use std::path::Path;

/// Run one case through the C++ pipeline and the pure-Rust pipeline.
fn both_pipelines(
    files_fn: impl Fn() -> Vec<FileInfo>,
    out_dir: &Path,
    slice: f64,
    memory_limit: Option<f64>,
    rec_slices: f64,
    extra: impl Fn(&mut parpar::generate::GenOpts) + Clone,
) -> (
    Result<BTreeMap<String, Vec<u8>>, String>,
    Result<BTreeMap<String, Vec<u8>>, String>,
) {
    // separate dirs, same output basename - produced names must match
    for d in ["ffi", "nat"] {
        std::fs::create_dir_all(out_dir.join(d)).unwrap();
    }
    let ffi = generate_opts(
        ffi_shared(false, false),
        &mut files_fn(),
        &out_dir.join("ffi/o"),
        slice,
        memory_limit,
        rec_slices,
        MethodRef::Name("".into()),
        extra.clone(),
    );
    let nat = generate_opts(
        native_shared(),
        &mut files_fn(),
        &out_dir.join("nat/o"),
        slice,
        memory_limit,
        rec_slices,
        MethodRef::Name("".into()),
        extra.clone(),
    );
    (ffi, nat)
}

fn assert_same_outcome(
    case: &str,
    ffi: &Result<BTreeMap<String, Vec<u8>>, String>,
    nat: &Result<BTreeMap<String, Vec<u8>>, String>,
) {
    match (ffi, nat) {
        (Ok(a), Ok(b)) => {
            if a != b {
                for (name, fb) in a.iter() {
                    match b.get(name) {
                        None => eprintln!("{case}: {name} missing in native output"),
                        Some(nb) if nb != fb => {
                            let off = fb
                                .iter()
                                .zip(nb.iter())
                                .position(|(x, y)| x != y)
                                .unwrap_or(usize::MAX);
                            eprintln!(
                                "{case}: {name} differs at byte {off} (ffi len {} nat len {})",
                                fb.len(),
                                nb.len()
                            );
                        }
                        _ => {}
                    }
                }
                for name in b.keys() {
                    if !a.contains_key(name) {
                        eprintln!("{case}: {name} only in native output");
                    }
                }
                panic!("{case}: FFI vs native output differs");
            }
        }
        (Err(a), Err(b)) => {
            assert_eq!(a, b, "{case}: error text differs: FFI `{a}` vs native `{b}`")
        }
        (a, b) => panic!(
            "{case}: outcome mismatch: FFI {:?} vs native {:?}",
            a.as_ref().map(|_| "ok"),
            b.as_ref().map(|_| "ok")
        ),
    }
}

/// Forced multi-pass recovery (256 slices over a 512 KiB limit, the
/// mem_limit_min oracle shape): C++ pipeline and the pure-Rust pipeline
/// must produce byte-identical output.
#[test]
fn multi_pass_differential() {
    let dir = std::env::temp_dir().join("e2e_multipass");
    let _ = std::fs::remove_dir_all(&dir);
    write_inputs(&dir.join("in"));

    let (ffi, nat) = both_pipelines(
        || file_info_for(&dir.join("in"), &["f1.bin", "f2.bin"]),
        &dir,
        8192.0,
        Some(524288.0),
        256.0,
        |_| {},
    );
    assert_same_outcome("multi-pass", &ffi, &nat);
}

/// Seeded randomized differential: random file trees x random option
/// combos through the C++ pipeline vs the pure-Rust pipeline.
/// Byte-identical output on success; identical error text on failure.
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

        let c = comment.clone();
        let (ffi, nat) = both_pipelines(
            || file_info_for(&in_dir, &name_refs),
            &dir,
            slice,
            mem,
            rec,
            move |o| {
                o.output_size_scheme = scheme.into();
                o.display_name_format = disp.into();
                o.unicode = uni;
                o.recovery_offset = offset;
                if let Some(c) = &c {
                    o.comments = vec![c.clone()];
                }
            },
        );
        assert_same_outcome(
            &format!(
                "case {case} (slice={slice} rec={rec} mem={mem:?} scheme={scheme} disp={disp} uni={uni:?} off={offset})"
            ),
            &ffi,
            &nat,
        );
    }
}

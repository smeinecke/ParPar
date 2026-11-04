//! Differential test: fileinfo scan results vs JS oracle fixtures
//! (`test/dump_fileinfo.js` -> `rust/fixtures/fi_*.json`).
//! Fixture paths are absolute into `rust/fixtures/tree/`; the recorded
//! prefix is patched to the runtime manifest dir.

use parpar::args::Recurse;
use parpar::fileinfo::file_info;
use serde_json::Value;
use std::os::unix::fs::FileTypeExt;

const ORACLE_PREFIX: &str = "/home/stefan/github/ParPar/rust/fixtures/tree";

/// Git cannot store the fifo (special file) or the empty `nodesc` dir -
/// recreate them on first use so a fresh checkout scans identically.
fn ensure_fixtures() {
    let dir = format!("{}/../../fixtures/tree", env!("CARGO_MANIFEST_DIR"));
    let _ = std::fs::create_dir(format!("{}/nodesc", dir));
    let fifo = format!("{}/fifo", dir);
    match std::fs::symlink_metadata(&fifo) {
        Ok(m) if m.file_type().is_fifo() => {}
        _ => unsafe {
            let c = std::ffi::CString::new(fifo.as_str()).unwrap();
            libc::mkfifo(c.as_ptr(), 0o644);
        },
    }
}

fn tree() -> String {
    ensure_fixtures();
    std::fs::canonicalize(format!(
        "{}/../../fixtures/tree",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
    .to_string_lossy()
    .to_string()
}

fn load(name: &str) -> Value {
    let p = format!(
        "{}/../../fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let s = std::fs::read_to_string(&p).unwrap();
    let s = s.replace(ORACLE_PREFIX, &tree());
    serde_json::from_str(&s).unwrap()
}

fn t(name: &str) -> String {
    format!("{}/{}", tree(), name)
}

fn compare(results: &[parpar::fileinfo::FileInfo], fx: &Value) {
    let expected = fx["results"].as_array().unwrap();
    let mut got: Vec<(String, u64, String, Option<String>)> = results
        .iter()
        .map(|f| {
            (
                f.name.clone(),
                f.size,
                f.md5_16k.iter().map(|b| format!("{:02x}", b)).collect(),
                f.md5
                    .map(|m| m.iter().map(|b| format!("{:02x}", b)).collect()),
            )
        })
        .collect();
    got.sort();
    let mut exp: Vec<(String, u64, String, Option<String>)> = expected
        .iter()
        .map(|e| {
            (
                e["name"].as_str().unwrap().to_string(),
                e["size"].as_u64().unwrap(),
                e["md5_16k"].as_str().unwrap().to_string(),
                e["md5"].as_str().map(|s| s.to_string()),
            )
        })
        .collect();
    exp.sort();
    assert_eq!(got.len(), exp.len(), "result count");
    for (g, e) in got.iter().zip(exp.iter()) {
        assert_eq!(g, e);
    }
}

fn scan(
    recurse: Recurse,
    skip: bool,
    files: &[&str],
) -> Result<Vec<parpar::fileinfo::FileInfo>, String> {
    let files: Vec<String> = files.iter().map(|f| t(f)).collect();
    file_info(&files, recurse, skip, 2, None)
}

/// recurse / skip-symlinks matrix vs oracle (no erroring files).
#[test]
fn scan_matrix() {
    let base: &[&str] = &[
        "a.bin",
        "b.bin",
        "deep",
        "empty.txt",
        "link_a",
        "dang",
        "nodesc",
    ];
    let nolinks: &[&str] = &["a.bin", "b.bin", "deep", "empty.txt", "nodesc"];
    for (rname, r) in [
        ("rfalse", Recurse::No),
        ("rtrue", Recurse::All),
        ("r1", Recurse::Depth(1)),
        ("r0", Recurse::Depth(0)),
        ("r-1", Recurse::Depth(-1)),
    ] {
        // skip=true -> symlinks skipped silently
        let fx = load(&format!("fi_{}_skip", rname));
        let got = scan(r, true, base).expect(&format!("scan {} skip", rname));
        compare(&got, &fx);

        // skip=false -> stat follows; dang removed (it errors)
        let fx = load(&format!("fi_{}_noskip", rname));
        let got = scan(r, false, nolinks).expect(&format!("scan {} noskip", rname));
        compare(&got, &fx);
    }
}

/// fs error texts: stat vs lstat syscall labels and the inspect format.
#[test]
fn scan_errors() {
    let fx = load("fi_missstat");
    let err = scan(Recurse::All, false, &["missing.bin"]).unwrap_err();
    assert_eq!(err, fx["error"].as_str().unwrap());

    let fx = load("fi_misslstat");
    let err = scan(Recurse::All, true, &["missing.bin"]).unwrap_err();
    assert_eq!(err, fx["error"].as_str().unwrap());

    // fifo is not a valid file
    let fx = load("fi_fifo");
    let err = scan(Recurse::All, false, &["fifo"]).unwrap_err();
    assert_eq!(err, fx["error"].as_str().unwrap());
}

/// The `path.join` normalization used for readdir children.
#[test]
fn join_dir_norm() {
    use parpar::fileinfo::join_dir;
    assert_eq!(join_dir("foo", "bar"), "foo/bar");
    assert_eq!(join_dir("foo/", "bar"), "foo/bar");
    assert_eq!(join_dir("foo//x", "bar"), "foo/x/bar");
    assert_eq!(join_dir("a/b/..", "f"), "a/f");
    assert_eq!(join_dir("/r", "x"), "/r/x");
}

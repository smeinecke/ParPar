//! Port of the `fileInfo` scanner at the end of lib/par2gen.js:
//! stat (+ optional lstat for skip-symlinks), directory recursion with a
//! numeric depth, 16KiB head-read for `md5_16k`, the `d41d8c...` empty
//! constant and the `size < 16KiB` whole-file digest reuse.
//!
//! JS runs the stat/read waterfalls through a BufferPool concurrency gate
//! (default 2). Here a worker pool over a shared queue provides the same
//! parallelism; results are pushed in completion order like JS (the PAR2
//! model sorts by file ID anyway).

use crate::args::Recurse;
use crate::compat::{errno_info, node_fs_error};
use md5::{Digest, Md5};
use std::collections::VecDeque;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// MD5 of the empty string (JS: `Buffer.from('d41d8c...','hex')`).
pub const EMPTY_MD5: [u8; 16] = [
    0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8, 0x42, 0x7e,
];

const HEAD_READ: usize = 16384;

/// One scanned input file (`{name, size, md5_16k, md5?}` in JS).
#[derive(Debug, Clone, Default)]
pub struct FileInfo {
    /// The path as passed/readdir-joined (display names derive later).
    /// May be lossy for non-UTF-8 names - use fs_path() for syscalls.
    pub name: String,
    /// Raw path for filesystem access when `name` was lossy-derived from
    /// a non-UTF-8 readdir entry (JS handles raw bytes; String can't).
    pub fs_path: Option<std::path::PathBuf>,
    /// Set by the displayNameFormat pass in the solver stage.
    pub display_name: Option<String>,
    /// `file._fullPath` - resolved absolute path (displayNameFormat 'common').
    pub full_path: Option<String>,
    pub size: u64,
    pub md5_16k: [u8; 16],
    /// Present for empty files and files smaller than 16KiB (the whole
    /// file was hashed by the head read).
    pub md5: Option<[u8; 16]>,
}

impl FileInfo {
    /// Path suitable for filesystem syscalls - the raw PathBuf when the
    /// display name was lossy, else `name`.
    pub fn fs_path(&self) -> &std::path::Path {
        self.fs_path
            .as_deref()
            .unwrap_or_else(|| std::path::Path::new(&self.name))
    }
}

/// `path.join` on POSIX - joins then normalizes `.`/`..` and collapses
/// duplicate separators. Node's join resolves dot segments.
fn node_path_join(base: &str, name: &str) -> String {
    let joined = if base.is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", base.trim_end_matches('/'), name)
    };
    // normalize: collapse '//' and resolve '.'/'..' segments
    let abs = joined.starts_with('/');
    let mut segs: Vec<&str> = Vec::new();
    for part in joined.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if !segs.is_empty() && *segs.last().unwrap() != ".." {
                    segs.pop();
                } else if !abs {
                    segs.push("..");
                }
            }
            s => segs.push(s),
        }
    }
    let mut out = String::new();
    if abs {
        out.push('/');
    }
    out.push_str(&segs.join("/"));
    if out.is_empty() {
        out.push('.');
    }
    out
}

struct WorkQueue {
    /// (fs path for syscalls, display name for FileInfo.name, depth)
    q: VecDeque<(std::path::PathBuf, String, Recurse)>,
    in_flight: usize,
    failed: Option<String>,
}

/// `ParPar.fileInfo(files, recurse, skipSymlinks, concurrency)` -
/// synchronous Rust version; `progress` (if given) is bumped per
/// completed file for the 200ms-interval scan display.
pub fn file_info(
    files: &[String],
    recurse: Recurse,
    skip_symlinks: bool,
    concurrency: usize,
    progress: Option<Arc<AtomicUsize>>,
) -> Result<Vec<FileInfo>, String> {
    let concurrency = if concurrency == 0 {
        2 // JS: `if(!concurrency) concurrency = 2`
    } else {
        // JS only gated a callback count with this; real threads must be
        // bounded or an absurd --chunk-read-threads crashes the process.
        // 64 is still far beyond useful I/O parallelism
        concurrency.min(64)
    };
    let shared = Arc::new((
        Mutex::new(WorkQueue {
            q: files
                .iter()
                .map(|f| (std::path::PathBuf::from(f), f.clone(), recurse))
                .collect(),
            in_flight: 0,
            failed: None,
        }),
        Condvar::new(),
    ));
    let results = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let shared = shared.clone();
        let results = results.clone();
        let progress = progress.clone();
        let skip_symlinks = skip_symlinks;
        handles.push(std::thread::spawn(move || {
            loop {
                let (fs_path, path, depth) = {
                    let (lock, cv) = &*shared;
                    let mut wq = lock.lock().unwrap();
                    loop {
                        if wq.failed.is_some() {
                            cv.notify_all();
                            return;
                        }
                        if let Some(item) = wq.q.pop_front() {
                            wq.in_flight += 1;
                            break item;
                        }
                        // queue empty: exit only when nothing else is in
                        // flight (a scanning dir may still push children)
                        if wq.in_flight == 0 {
                            cv.notify_all();
                            return;
                        }
                        wq = cv.wait(wq).unwrap();
                    }
                };
                match scan_one(&fs_path, &path, depth, skip_symlinks) {
                    Ok(ScanResult::File(info)) => {
                        results.lock().unwrap().push(info);
                        if let Some(p) = &progress {
                            p.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Ok(ScanResult::Children(kids)) => {
                        let (lock, cv) = &*shared;
                        let mut wq = lock.lock().unwrap();
                        // children inherit numeric depth via Recurse::next
                        for k in kids {
                            wq.q.push_back(k);
                        }
                        cv.notify_all();
                    }
                    Ok(ScanResult::Skip) => {}
                    Err(e) => {
                        let (lock, cv) = &*shared;
                        let mut wq = lock.lock().unwrap();
                        if wq.failed.is_none() {
                            wq.failed = Some(e);
                        }
                        wq.q.clear();
                        cv.notify_all();
                    }
                }
                {
                    let (lock, cv) = &*shared;
                    let mut wq = lock.lock().unwrap();
                    wq.in_flight -= 1;
                    cv.notify_all();
                }
            }
        }));
    }

    // wait for workers; they exit when queue is empty or failed
    for h in handles {
        let _ = h.join();
    }
    let (lock, _) = &*shared;
    let wq = lock.lock().unwrap();
    if let Some(e) = wq.failed.clone() {
        return Err(e);
    }
    drop(wq);
    let results = Arc::try_unwrap(results)
        .map_err(|_| "results still shared")
        .unwrap()
        .into_inner()
        .unwrap();
    Ok(results)
}

enum ScanResult {
    File(FileInfo),
    Children(Vec<(std::path::PathBuf, String, Recurse)>),
    Skip,
}

/// The per-file stat+read waterfall. `fs_path` is the raw filesystem
/// path; `path` is the display name stored in FileInfo (may be lossy
/// for non-UTF-8 readdir entries - JS uses raw bytes for both).
fn scan_one(
    fs_path: &std::path::Path,
    path: &str,
    recurse: Recurse,
    skip_symlinks: bool,
) -> Result<ScanResult, String> {
    let meta = if skip_symlinks {
        std::fs::symlink_metadata(fs_path)
    } else {
        std::fs::metadata(fs_path)
    };
    let meta = match meta {
        Ok(m) => m,
        Err(e) => {
            return Err(stat_err(&e, path, skip_symlinks));
        }
    };
    let ft = meta.file_type();
    if ft.is_dir() {
        if !recurse.enabled() {
            return Ok(ScanResult::Skip);
        }
        let rd = match std::fs::read_dir(fs_path) {
            Ok(r) => r,
            Err(e) => {
                return Err(io_err(&e, "scandir", Some(path)));
            }
        };
        let mut kids = Vec::new();
        for ent in rd {
            // JS propagates readdir entry errors - don't silently drop files
            let ent = match ent {
                Ok(e) => e,
                Err(e) => return Err(io_err(&e, "scandir", Some(path))),
            };
            // raw path for syscalls; lossy string for the display name
            let raw = fs_path.join(ent.file_name());
            let child = node_path_join(path, &ent.file_name().to_string_lossy());
            kids.push((raw, child, recurse.next()));
        }
        return Ok(ScanResult::Children(kids));
    }
    if skip_symlinks && ft.is_symlink() {
        return Ok(ScanResult::Skip);
    }
    if !ft.is_file() {
        // JS: plain `new Error(file + ' is not a valid file')` - printed
        // via console.error(err) as `Error: msg` + stack; the first line
        // is what we can match deterministically.
        return Err(format!("Error: {} is not a valid file", path));
    }

    let size = meta.len();
    let mut info = FileInfo {
        name: path.to_string(),
        fs_path: Some(fs_path.to_path_buf()),
        display_name: None,
        full_path: None,
        size,
        md5_16k: EMPTY_MD5,
        md5: None,
    };
    if size == 0 {
        info.md5 = Some(EMPTY_MD5);
        return Ok(ScanResult::File(info));
    }
    let mut f = match std::fs::File::open(fs_path) {
        Ok(f) => f,
        Err(e) => return Err(io_err(&e, "open", Some(path))),
    };
    let want = (size as usize).min(HEAD_READ);
    let mut buf = vec![0u8; want];
    // read() may legally return short (FUSE/NFS, truncation) - loop so the
    // 16k hash always covers a full head, and only tag md5 for whole-file reads
    let mut n = 0usize;
    while n < want {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(r) => n += r,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io_err(&e, "read", None)),
        }
    }
    if n < want {
        return Err(format!(
            "Error: Read failure - expected {} bytes, got {} bytes instead.",
            want, n
        ));
    }
    let mut h = Md5::new();
    h.update(&buf);
    let digest: [u8; 16] = h.finalize().into();
    info.md5_16k = digest;
    if size < HEAD_READ as u64 {
        info.md5 = Some(digest);
    }
    Ok(ScanResult::File(info))
}

/// fs.stat / fs.lstat error -> Node syscall label + inspect format.
fn stat_err(e: &std::io::Error, path: &str, lstat: bool) -> String {
    let errno = e.raw_os_error().unwrap_or(5);
    let (code, desc) = errno_info(errno);
    node_fs_error(
        -errno,
        code,
        desc,
        if lstat { "lstat" } else { "stat" },
        Some(path),
    )
}

fn io_err(e: &std::io::Error, syscall: &str, path: Option<&str>) -> String {
    let errno = e.raw_os_error().unwrap_or(5);
    let (code, desc) = errno_info(errno);
    node_fs_error(-errno, code, desc, syscall, path)
}

impl Recurse {
    /// `typeof recurse == 'number' ? recurse-1 : recurse` - Depth values
    /// decrement per level; Depth<=0 is falsy (dir skipped), negative
    /// numbers are truthy in JS (unlimited).
    fn next(self) -> Recurse {
        match self {
            Recurse::Depth(n) => Recurse::Depth(n.saturating_sub(1)),
            other => other,
        }
    }
    /// JS `if(recurse)` truthiness on the depth.
    fn enabled(self) -> bool {
        match self {
            Recurse::No => false,
            Recurse::All => true,
            Recurse::Depth(n) => n != 0,
        }
    }
}

/// Directory listing join helper kept public for tests.
pub fn join_dir(base: &str, name: &str) -> String {
    node_path_join(base, name)
}

/// A `PathBuf` for filesystem ops derived from the display path.
pub fn to_pathbuf(name: &str) -> PathBuf {
    PathBuf::from(name)
}

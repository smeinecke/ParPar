//! System information used by the solver and CLI - ports of the `os`,
//! `process` and `path` facts par2gen/parpar.js read:
//! `os.totalmem`, `os.freemem`, `process.constrainedMemory`,
//! `os.cpus().length`, `os.cpus()[0].model`, `process.arch`,
//! `buffer.kMaxLength` and the path separator helpers.

use crate::compat;

/// `os.totalmem()` - libuv uses sysconf(_SC_PHYS_PAGES*_SC_PAGESIZE).
#[cfg(unix)]
pub fn total_mem() -> u64 {
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let psz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages <= 0 || psz <= 0 {
        return 0;
    }
    pages as u64 * psz as u64
}

#[cfg(not(unix))]
pub fn total_mem() -> u64 {
    0
}

/// `os.freemem()` - libuv >=1.45 reads MemAvailable from /proc/meminfo on
/// Linux, falling back to sysinfo.freeram*mem_unit.
#[cfg(target_os = "linux")]
pub fn free_mem() -> u64 {
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest
                    .trim()
                    .trim_end_matches(" kB")
                    .trim()
                    .parse()
                    .unwrap_or(0);
                if kb > 0 {
                    return kb * 1024;
                }
            }
        }
    }
    let mut info: libc::sysinfo = unsafe { std::mem::zeroed() };
    if unsafe { libc::sysinfo(&mut info) } == 0 {
        return info.freeram as u64 * info.mem_unit as u64;
    }
    0
}

#[cfg(not(target_os = "linux"))]
pub fn free_mem() -> u64 {
    total_mem() / 4
}

/// `process.constrainedMemory()` - libuv's uv_get_constrained_memory:
/// cgroup v2 `memory.max`, then v1 `memory/memory.limit_in_bytes`;
/// returns `(uint64_t)-1` when no constraint is readable. Node passes the
/// sentinel through as a number (verified on Node 24), so we return it
/// verbatim - the solver treats any truthy value as constrained.
pub fn constrained_memory() -> u64 {
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        if let Ok(buf) = std::fs::read_to_string(path) {
            let t = buf.trim();
            // strtoull semantics: unparseable text ("max") -> 0 which JS
            // treats as unconstrained; keep the first parseable value
            if let Ok(v) = t.parse::<u64>() {
                if v > 0 {
                    return v;
                }
            }
        }
    }
    u64::MAX
}

/// `os.cpus().length`.
pub fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// `os.cpus()[0].model` - first `model name` in /proc/cpuinfo, trimmed.
/// Returns "unknown" when no model string is found (JS callers map
/// 'unknown' to 'CPU' for display).
#[cfg(target_os = "linux")]
pub fn cpu_model() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some((_, v)) = rest.split_once(':') {
                    return v.trim().to_string();
                }
            }
        }
        // ARM-style fields
        for key in ["Hardware", "Processor"] {
            for line in cpuinfo.lines() {
                if let Some(rest) = line.strip_prefix(key) {
                    if let Some((_, v)) = rest.split_once(':') {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    "unknown".to_string()
}

#[cfg(not(target_os = "linux"))]
pub fn cpu_model() -> String {
    "unknown".to_string()
}

/// par2gen's `is64bPlatform` - `['arm64','ppc64','x64']` on process.arch.
/// Note riscv64 deliberately takes the non-64-bit branch.
pub fn is_64b_platform() -> bool {
    matches!(compat::node_arch(), "arm64" | "ppc64" | "x64")
}

/// `buffer.kMaxLength` (MAX_BUFFER_SIZE subtracts 1024+68 at the use
/// site). 64-bit Node reports 2^53-1; 32-bit Node reports 2^30-1.
pub const K_MAX_LENGTH: u64 = if cfg!(target_pointer_width = "64") {
    9007199254740991
} else {
    1073741823
};

/// `MAX_BUFFER_SIZE` from par2gen.js - kMaxLength with headroom for
/// alignment padding + the recovery header.
pub fn max_buffer_size() -> u64 {
    K_MAX_LENGTH - 1024 - 68
}

/// `pathToPar2` - converts native separators to '/' for display names.
/// Identity on unix.
pub fn path_to_par2(p: &str) -> String {
    if cfg!(windows) {
        p.replace('\\', "/")
    } else {
        p.to_string()
    }
}

/// `pathNormalize` - lowercase + backslash on Windows; identity on unix.
pub fn path_normalize(p: &str) -> String {
    if cfg!(windows) {
        p.replace('/', "\\").to_lowercase()
    } else {
        p.to_string()
    }
}

/// `path.sep` value used by the JS platform check.
pub fn path_sep() -> char {
    if cfg!(windows) {
        '\\'
    } else {
        '/'
    }
}

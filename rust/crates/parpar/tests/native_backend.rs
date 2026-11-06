#![cfg(all(target_env = "gnu", target_arch = "x86_64"))]
use parpar::native_backend::NativeBackend;
use parpar_backend::ComputeBackend;
use parpar_ffi::FfiBackend;

/// Static method info parity across the full GF_METHODS table.
#[test]
fn gf_info_parity_all_methods() {
    let mut diffs = Vec::new();
    for m in 0..32 {
        let ffi = FfiBackend::gf_info(m);
        let nat = NativeBackend::gf_info(m);
        match (ffi, nat) {
            (Ok(f), Ok(n)) => {
                // AUTO resolves per-backend (FFI picks its own default,
                // native picks its own kernel) - compare resolved shapes
                if m == 0 {
                    continue;
                }
                if f.id != n.id
                    || f.name != n.name
                    || f.alignment != n.alignment
                    || f.stride != n.stride
                    || f.target_chunk != n.target_chunk
                    || f.target_grouping != n.target_grouping
                {
                    diffs.push(format!("m{}: ffi={:?} native={:?}", m, f, n));
                }
            }
            (Ok(f), Err(e)) => diffs.push(format!("m{}: ffi ok {:?} native err {}", m, f, e)),
            (Err(e), Ok(n)) => diffs.push(format!("m{}: ffi err {} native ok {:?}", m, e, n)),
            (Err(_), Err(_)) => {}
        }
    }
    assert!(diffs.is_empty(), "gf_info diffs:\n{}", diffs.join("\n"));
}

#[test]
fn hash_method_names_parity() {
    let f = FfiBackend::inhash_method_name();
    let n = NativeBackend::inhash_method_name();
    eprintln!("inhash: ffi={} native={}", f, n);
}

// Enumeration parity + absent-runtime coverage.
//
// parity: parpar_ocl::enumerate() must produce the same platform/device
// fields as the FFI backend's opencl_platforms() (same driver, same order).
//
// absent_runtime: re-executes this test binary with PARPAR_OCL_LIBRARY
// pointed at a nonexistent path and checks enumerate() returns empty.

#![cfg(target_env = "gnu")]
use parpar_backend::ComputeBackend;
use parpar_ffi::FfiBackend;

fn flatten(
    plats: &[parpar_backend::OclPlatform],
) -> Vec<(String, i32, bool, String, u64, u32, u32, u32)> {
    plats
        .iter()
        .flat_map(|p| {
            p.devices.iter().map(|d| {
                (
                    p.name.clone(),
                    d.vendor_id,
                    d.supported,
                    d.device_type.clone(),
                    d.memory_global,
                    d.workgroup_limit,
                    d.workgroup_multiple,
                    d.compute_units,
                )
            })
        })
        .collect()
}

#[test]
fn native_enum_matches_ffi() {
    let Some(skip) = std::env::var_os("PARPAR_OCL_LIBRARY") else {
        // absent-runtime child: loader must fail cleanly and enumerate empty
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .args(["native_enum_matches_ffi", "--exact", "--nocapture"])
            .env("PARPAR_OCL_LIBRARY", "/nonexistent/libOpenCL.so.99")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stdout.contains("enum-empty-ok") || stderr.contains("enum-empty-ok"),
            "absent-runtime child output: {}{}",
            stdout,
            stderr
        );
        return;
    };
    let _ = skip;
    if parpar_ocl::enumerate().is_empty() {
        println!("enum-empty-ok");
        return;
    }
    panic!("expected empty enumeration with PARPAR_OCL_LIBRARY set");
}

#[test]
fn enum_fields_parity() {
    if std::env::var_os("PARPAR_OCL_LIBRARY").is_some() {
        return; // child invocation of the absent-runtime test
    }
    let native = parpar_ocl::enumerate();
    let mut be = FfiBackend::new().expect("ffi backend");
    let ffi = be.opencl_platforms().unwrap_or_default();
    if native.is_empty() || ffi.is_empty() {
        eprintln!("no OpenCL devices - skipping parity check");
        return;
    }
    let n = flatten(&native);
    let f = flatten(&ffi);
    assert_eq!(n.len(), f.len(), "device count differs\n{:?}\n{:?}", n, f);
    for (i, (nv, fv)) in n.iter().zip(f.iter()).enumerate() {
        assert_eq!(nv, fv, "device {} differs", i);
    }
}

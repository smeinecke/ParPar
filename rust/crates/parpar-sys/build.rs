use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_dir = manifest_dir.join("native");
    let shim_dir = manifest_dir.join("shim");

    // rebuild if the build description or shim change
    println!("cargo:rerun-if-changed={}", native_dir.display());
    println!("cargo:rerun-if-changed={}", shim_dir.display());
    println!("cargo:rerun-if-env-changed=PARPAR_NATIVE_SANITIZE");
    println!("cargo:rerun-if-env-changed=PARPAR_NATIVE_OPENCL");
    println!("cargo:rerun-if-env-changed=PARPAR_NATIVE_TUNING");
    println!("cargo:rerun-if-env-changed=CMAKE_BUILD_PARALLEL_LEVEL");

    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".into());
    let build_type = if profile == "debug" {
        "Debug"
    } else {
        "Release"
    };

    let build_dir = out_dir.join("native-build");
    std::fs::create_dir_all(&build_dir).expect("create cmake build dir");

    let opencl = env::var("CARGO_FEATURE_OPENCL").is_ok()
        && env::var("PARPAR_NATIVE_OPENCL").unwrap_or_else(|_| "1".into()) != "0";
    let sanitize = env::var("PARPAR_NATIVE_SANITIZE").unwrap_or_default() == "1";
    let tuning = env::var("PARPAR_NATIVE_TUNING").unwrap_or_default() == "1";

    let mut cfg = Command::new("cmake");
    cfg.arg(format!("-DCMAKE_BUILD_TYPE={}", build_type))
        .arg(format!(
            "-DPARPAR_NATIVE_OPENCL={}",
            if opencl { "ON" } else { "OFF" }
        ))
        .arg(format!(
            "-DPARPAR_NATIVE_SANITIZE={}",
            if sanitize { "ON" } else { "OFF" }
        ))
        .arg(format!(
            "-DPARPAR_NATIVE_TUNING={}",
            if tuning { "ON" } else { "OFF" }
        ))
        .arg(&native_dir)
        .current_dir(&build_dir);
    run(&mut cfg, "cmake configure");

    let mut bld = Command::new("cmake");
    bld.arg("--build")
        .arg(".")
        .arg("--config")
        .arg(build_type)
        .arg("--parallel")
        .current_dir(&build_dir);
    run(&mut bld, "cmake build");

    // static archive location differs per generator/platform
    let lib_dir = find_archive_dir(&build_dir, build_type);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=parpar_native");

    // system libuv
    if pkg_config_has("libuv") {
        println!("cargo:rustc-link-lib=dylib=uv");
    } else {
        // best-effort fallback: rely on default search paths
        println!("cargo:rustc-link-lib=dylib=uv");
    }

    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("linux") || target.contains("bsd") || target.contains("darwin") {
        if target.contains("linux") || target.contains("bsd") {
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
        }
        if target.contains("darwin") {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }

    // allow overriding the linker via env for cross builds
    if sanitize {
        println!("cargo:rustc-link-arg=-fsanitize=address");
        println!("cargo:rustc-link-arg=-fsanitize=undefined");
    }
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {}", what, e));
    if !status.success() {
        panic!("{} failed with status {}", what, status);
    }
}

fn pkg_config_has(name: &str) -> bool {
    Command::new("pkg-config")
        .args(["--exists", name])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn find_archive_dir(build_dir: &Path, build_type: &str) -> PathBuf {
    let candidates = [
        build_dir.join(build_type),
        build_dir.to_path_buf(),
        build_dir.join("lib"),
        build_dir.join("Release"),
    ];
    for c in &candidates {
        if c.join("libparpar_native.a").exists() || c.join("parpar_native.lib").exists() {
            return c.to_path_buf();
        }
    }
    build_dir.to_path_buf()
}

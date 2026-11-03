# parpar (Rust)

Native Rust port of ParPar. The `parpar` binary produced here is a
drop-in replacement for the Node.js CLI: same arguments, same progress
and JSON output, byte-identical `.par2` recovery files.

## Build

```sh
cargo build --release -p parpar
```

The default `ffi` cargo feature links the C++ reference implementation
(via `parpar-sys` + cmake) so the differential oracle stays available
at runtime. For a pure-Rust binary with no C++ toolchain or runtime:

```sh
cargo build --release -p parpar --no-default-features
```

`--no-default-features` builds contain no `parpar-sys`/`parpar-ffi`
dependency, no libstdc++/libuv linkage, and report a clean error if
FFI-only functionality is requested.

## Backend selection (environment variables)

| Variable | Effect |
| --- | --- |
| `PARPAR_FORCE_FFI=1` | Route the whole backend through the C++ FFI oracle instead of native Rust (needs the `ffi` feature) |
| `PARPAR_NATIVE_GF=1` | On the FFI path, use native GF16 processors (coexistence/debug) |
| `PARPAR_NATIVE_HASH=1` | On the FFI path, use native hashers (coexistence/debug) |
| `PARPAR_OCL_LIBRARY=/path/libOpenCL.so` | Override the OpenCL runtime soname (custom ICD; a bogus path simulates an absent runtime) |
| `PARPAR_DUMP_ARGS=1` | Dump parsed CLI arguments as JSON (debug) |

## OpenCL

OpenCL is loaded at runtime, never linked: `libOpenCL.so`,
`libOpenCL.so.1`, `libOpenCL.so.1.0.0` on POSIX and `OpenCL.dll` on
Windows, mirroring the C++ loader. No runtime or no suitable device
means CPU-only processing with the same diagnostics as the reference
implementation.

## Method selection

`--method` accepts the same GF16 kernel names as the reference CLI.

Kernel coverage:

| Methods | Kernels | Verification |
| --- | --- | --- |
| 1-3 | Lookup (+ labelled variants) | native |
| 4, 28, 29 | NEON Shuffle, CLMul NEON, CLMul SHA3 | QEMU aarch64, e2e |
| 10-12 | SSSE3/AVX/AVX2 Shuffle | native, e2e |
| 13, 14, 16, 20 | AVX512 Shuffle, VBMI, Shuffle2x, static Xor-Jit | Intel SDE + FFI differential, e2e |
| 15 | Shuffle2x AVX2 | native, e2e |
| 17-19 | Xor SSE2 + static Xor-Jit SSE2/AVX2 | native, e2e |
| 21-26 | GFNI/AVX2/AVX512 Affine + Affine2x | Intel SDE + FFI differential, e2e |
| 5-9, 27, 30, 31 | SVE/SVE2/RVV, AVX512-BMM | unimplemented: no stable-Rust intrinsics (BMM: ISA cancelled upstream) |

Explicitly requested unsupported methods error cleanly
(`GF16 method not supported on this CPU`); the reference build crashes
with SIGILL in that situation. Auto selection mirrors the C++ priority
order (GFNI affine > VBMI > AVX512 > AVX2 > SSE). The C++ XOR-Jit
methods are served by static bit-plane kernels computing the identical
transform; auto still prefers Shuffle AVX2 over the static XOR path
since the JIT speed advantage does not exist in a static port.

## Platforms

| Target | Status |
| --- | --- |
| `x86_64-unknown-linux-gnu` | Full suite incl. C++ oracle; SSE2/AVX2/PCLMUL kernels + x86 asm MD5 |
| `aarch64-unknown-linux-gnu` | `cargo check` + CI-verified; NEON/CLMul/SHA3 kernels verified under QEMU e2e |
| `x86_64-pc-windows-msvc` | `cargo check` + CI-verified (`seek_read`/`seek_write` I/O paths) |
| `x86_64-apple-darwin`, `aarch64-apple-darwin` | `cargo check` + CI-verified |
| `riscv64gc-unknown-linux-musl` | scalar Lookup verified under QEMU e2e (RVV unimplementable on stable Rust) |

CI (`.github/workflows/test-rust.yml`) runs the differential suite on
Linux, native-crate tests plus `--no-default-features` builds on
macOS/Windows/ARM64, a no-C++-linkage check, and a job that builds the
pinned JS addon for the `e2e_gen` node oracle.

## Performance

End-to-end generation is within ~5-10% of the C++ backend on x86_64
(700 MiB input, `-s4M -r30`: ~0.73-0.90s native vs ~0.70s FFI). The
GF16 stage uses tile-major scheduling so input tiles stay cache-hot
across recovery outputs; input hashing uses a port of the reference
x86 MD5x2 assembly. On non-x86 targets the NEON GF16 kernels and scalar
hasher fallbacks are used.

## Layout

- `crates/parpar` - CLI, solver, pipeline, file info, backend routing
- `crates/parpar-backend` - shared traits (`GfProc`, `ComputeBackend`,
  composite CPU/GPU proc)
- `crates/parpar-gf16` - GF(2^16) math, CPU kernels, `NativeGfProc`
- `crates/parpar-hasher` - MD5 (x86 asm kernel, multi-buffer), CRC32
- `crates/parpar-ocl` - runtime OpenCL loader + host backend
- `crates/parpar-ffi`, `crates/parpar-sys` - reference oracle bindings
  (`ffi` feature, excluded from production no-default-features builds)

## Tests

```sh
cargo test --workspace --all-targets
```

Differential tests compare native and FFI output byte-for-byte; OpenCL
tests run when a device is present and skip cleanly otherwise;
`e2e_gen` compares against `node ../bin/parpar.js` when the JS addon
is built (or the FFI binary via `PARPAR_FFI_BIN`), skipping otherwise.

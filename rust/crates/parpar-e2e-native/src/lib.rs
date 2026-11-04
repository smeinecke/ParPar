//! Host crate for cross-platform native-pipeline e2e tests.
//!
//! Depends on `parpar` with `default-features = false` so the whole tree
//! builds without the C++ oracle - the tests under `tests/` therefore run
//! on every target (Windows MSVC, macOS, ARM) unlike the gnu-only
//! differential suite in `crates/parpar/tests/`.

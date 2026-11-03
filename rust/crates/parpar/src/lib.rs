//! parpar - high performance PAR2 creation tool (Rust port).
//!
//! Library root for the CLI + orchestration crate. Modules are ported in
//! the plan's Phase 1 dependency order: compat + cliparse + args first,
//! then the PAR2 packet model, scanning, solver/layout, I/O scheduling
//! and the full pipeline.

pub mod args;
pub mod cliparse;
pub mod compat;
pub mod dump;
pub mod fileinfo;
pub mod generate;
pub mod native_backend;
pub mod outfile;
pub mod par2;
pub mod pipeline;
pub mod present;
pub mod readers;
pub mod sysinfo;
pub mod throttle;

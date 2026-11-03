//! Native GF(2^16) framework for ParPar - field arithmetic, coefficient
//! generation, scalar lookup kernels, and a synchronous GfProc.

#[cfg(target_arch = "x86_64")]
pub mod affine;
#[cfg(target_arch = "aarch64")]
pub mod clmul_neon;
pub mod gf;
pub mod kernel;
pub mod proc;
#[cfg(target_arch = "x86_64")]
pub mod shuffle2x;
#[cfg(target_arch = "x86_64")]
pub mod shuffle_avx2;
#[cfg(target_arch = "x86_64")]
pub mod shuffle_avx512;
#[cfg(target_arch = "aarch64")]
pub mod shuffle_neon;
#[cfg(target_arch = "x86_64")]
pub mod shuffle_sse;
#[cfg(target_arch = "x86_64")]
pub mod shuffle_vbmi;
#[cfg(target_arch = "x86_64")]
pub mod xor_avx2;
#[cfg(target_arch = "x86_64")]
pub mod xor_avx512;
#[cfg(target_arch = "x86_64")]
pub mod xor_sse;

pub use gf::{
    calc_table, gf16_exp, gf16_mul, gf16_mul2, gfmat_coeff, gfmat_coeff_from_log, gfmat_coeff_log,
    gfmat_input_log, mul_region, muladd_region, GF16_POLYNOMIAL,
};
pub use kernel::{method_info, packed_len, pick, pick_explicit, GfKernel};
pub mod range;
pub use proc::{EventQueues, NativeGfProc};
pub use range::CpuRangeBackend;

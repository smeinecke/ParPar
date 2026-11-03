//! Raw FFI bindings to the ParPar C++ compute backend via parpar_shim.h.
//!
//! These declarations are hand-maintained to mirror the shim header exactly.
//! All functions are unsafe; callers must uphold the buffer-lifetime and
//! serialisation contracts documented in parpar_shim.h.

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct pps_ctx {
    _private: [u8; 0],
}
#[repr(C)]
pub struct pps_gfp {
    _private: [u8; 0],
}
#[repr(C)]
pub struct pps_hin {
    _private: [u8; 0],
}
#[repr(C)]
pub struct pps_hout {
    _private: [u8; 0],
}

pub const PPS_EV_INPUT_DONE: c_int = 1;
pub const PPS_EV_ENDED: c_int = 2;
pub const PPS_EV_OUTPUT_DONE: c_int = 3;
pub const PPS_EV_PROGRESS: c_int = 4;
pub const PPS_EV_ERROR: c_int = 5;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct pps_event {
    pub proc_: *mut pps_gfp,
    pub kind: c_int,
    pub token: u64,
    pub value: i64,
    pub err: c_int,
    pub errbuf: [c_char; 192],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct pps_cpu_spec {
    pub method: c_int,
    pub input_batchsize: u32,
    pub input_minbatchsize: u32,
    pub chunk_size: u32,
    pub slice_offset: u64,
    pub slice_size: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct pps_ocl_spec {
    pub platform: c_int,
    pub device: c_int,
    pub slice_size: u64,
    pub slice_offset: u64,
    pub method: c_int,
    pub cksum_method: c_int,
    pub input_batchsize: c_int,
    pub input_minbatchsize: c_int,
    pub target_iters: u32,
    pub target_grouping: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct pps_gf_cpu_info {
    pub threads: c_int,
    pub method_desc: [c_char; 64],
    pub chunk_size: u64,
    pub staging_count: u32,
    pub staging_size: u64,
    pub alignment: u32,
    pub stride: u64,
    pub slice_mem: u64,
    pub num_output_slices: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct pps_gf_ocl_info {
    pub device_name: [c_char; 128],
    pub method_desc: [c_char; 64],
    pub staging_count: u32,
    pub staging_size: u64,
    pub chunk_size: u64,
    pub output_chunks: u32,
    pub slice_mem: u64,
    pub num_output_slices: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct pps_method_info {
    pub id: c_int,
    pub name: [c_char; 64],
    pub alignment: u64,
    pub stride: u64,
    pub target_chunk: u64,
    pub target_grouping: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct pps_ocl_dev {
    pub id: c_int,
    pub name: [c_char; 256],
    pub vendor_id: i32,
    pub available: c_int,
    pub supported: c_int,
    pub type_: [c_char; 16],
    pub memory_global: u64,
    pub cache_global: u64,
    pub memory_unified: c_int,
    pub memory_constant: u64,
    pub memory_local: u64,
    pub has_memory_local: c_int,
    pub memory_max_alloc: u64,
    pub workgroup_limit: u32,
    pub workgroup_multiple: u32,
    pub compute_units: u32,
}

unsafe impl Send for pps_ctx {}
unsafe impl Sync for pps_ctx {}
unsafe impl Send for pps_gfp {}
unsafe impl Send for pps_hin {}
unsafe impl Send for pps_hout {}

extern "C" {
    pub fn pps_ctx_new() -> *mut pps_ctx;
    pub fn pps_ctx_free(ctx: *mut pps_ctx);
    pub fn pps_wait_event(ctx: *mut pps_ctx, out: *mut pps_event, timeout_ms: c_int) -> c_int;

    pub fn pps_gfp_new(
        ctx: *mut pps_ctx,
        slice_size: u64,
        cpu: *const pps_cpu_spec,
        ocl: *const pps_ocl_spec,
        n_ocl: usize,
        staging_areas: c_int,
        out: *mut *mut pps_gfp,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_set_num_threads(
        g: *mut pps_gfp,
        threads: c_int,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_set_current_slice_size(
        g: *mut pps_gfp,
        slice_size: u64,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_set_recovery_slices(
        g: *mut pps_gfp,
        exponents: *const u16,
        n: usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_free_mem(g: *mut pps_gfp, errbuf: *mut c_char, errlen: usize) -> c_int;
    pub fn pps_gfp_add_input(
        g: *mut pps_gfp,
        buf: *const c_void,
        len: usize,
        input_index: c_int,
        token: u64,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_end_input(
        g: *mut pps_gfp,
        token: u64,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_get_output(
        g: *mut pps_gfp,
        output_index: c_int,
        out: *mut c_void,
        out_len: usize,
        token: u64,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_set_progress(
        g: *mut pps_gfp,
        on: c_int,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_info(
        g: *mut pps_gfp,
        has_cpu: *mut c_int,
        cpu: *mut pps_gf_cpu_info,
        n_ocl: *mut c_int,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_ocl_info(
        g: *mut pps_gfp,
        index: c_int,
        out: *mut pps_gf_ocl_info,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_gfp_close(g: *mut pps_gfp, errbuf: *mut c_char, errlen: usize) -> c_int;
    pub fn pps_gfp_free(g: *mut pps_gfp);

    pub fn pps_gf_info(method: c_int, out: *mut pps_method_info) -> c_int;
    pub fn pps_set_inhash(method: c_int) -> c_int;
    pub fn pps_set_outhash(level: c_int);
    pub fn pps_inhash_method() -> *const c_char;
    pub fn pps_outhash_method() -> *const c_char;

    pub fn pps_ocl_load(errbuf: *mut c_char, errlen: usize) -> c_int;
    pub fn pps_ocl_default_platform() -> c_int;
    pub fn pps_ocl_default_device(platform: c_int) -> c_int;
    pub fn pps_ocl_platform_count() -> c_int;
    pub fn pps_ocl_platform_name(platform: c_int, buf: *mut c_char, len: usize) -> c_int;
    pub fn pps_ocl_device_count(platform: c_int) -> c_int;
    pub fn pps_ocl_device_info(platform: c_int, device: c_int, out: *mut pps_ocl_dev) -> c_int;

    pub fn pps_hin_new(slice_size: u64, block_out: *mut u8, block_out_len: usize) -> *mut pps_hin;
    pub fn pps_hin_update(
        h: *mut pps_hin,
        buf: *const c_void,
        len: usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_hin_end(
        h: *mut pps_hin,
        out: *mut u8,
        out_len: usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_hin_reset(h: *mut pps_hin, errbuf: *mut c_char, errlen: usize) -> c_int;
    pub fn pps_hin_free(h: *mut pps_hin);

    pub fn pps_hout_new(regions: u32) -> *mut pps_hout;
    pub fn pps_hout_update(
        h: *mut pps_hout,
        bufs: *const *const c_void,
        num_bufs: usize,
        len: usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_hout_get(
        h: *mut pps_hout,
        out: *mut u8,
        out_len: usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    pub fn pps_hout_reset(h: *mut pps_hout, errbuf: *mut c_char, errlen: usize) -> c_int;
    pub fn pps_hout_free(h: *mut pps_hout);
}

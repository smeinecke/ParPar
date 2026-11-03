//! Safe adapter implementing parpar-backend contracts over the C++ FFI shim.
//!
//! Buffer ownership model: a submitted SliceBuf is parked in `pending`
//! keyed by (proc_id, token) while C++ borrows its memory, and is moved
//! into the completion event - mirroring Node's attachValue/callback-arg
//! behaviour without unsafe borrows.

use parpar_backend::*;
use parpar_sys as sys;
use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

const ERRBUF_LEN: usize = 192;

struct ErrBuf([u8; ERRBUF_LEN]);
impl ErrBuf {
    fn new() -> Self {
        ErrBuf([0; ERRBUF_LEN])
    }
    fn msg(&self) -> Option<String> {
        let s = unsafe { CStr::from_ptr(self.0.as_ptr() as *const std::os::raw::c_char) };
        let s = s.to_string_lossy();
        if s.is_empty() {
            None
        } else {
            Some(s.into_owned())
        }
    }
    fn as_mut_ptr(&mut self) -> *mut std::os::raw::c_char {
        self.0.as_mut_ptr() as *mut std::os::raw::c_char
    }
}

fn last_err(eb: &ErrBuf, fallback: &str) -> BackendError {
    BackendError::new(eb.msg().unwrap_or_else(|| fallback.to_string()))
}

/// Shared FFI state: the shim context plus parked submission buffers.
struct FfiInner {
    ctx: *mut sys::pps_ctx,
    pending: Mutex<HashMap<(usize, u64), SliceBuf>>,
}

impl fmt::Debug for FfiInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FfiInner")
            .field("ctx", &self.ctx)
            .field("pending", &self.pending.lock().map(|m| m.len()))
            .finish()
    }
}

// The shim serialises all controller calls through its own libuv thread, so
// the context pointer is safe to share across threads.
unsafe impl Send for FfiInner {}
unsafe impl Sync for FfiInner {}

impl Drop for FfiInner {
    fn drop(&mut self) {
        unsafe { sys::pps_ctx_free(self.ctx) }
    }
}

#[derive(Clone)]
pub struct FfiBackend {
    inner: Arc<FfiInner>,
}

impl Default for FfiBackend {
    fn default() -> Self {
        Self::new().expect("failed to create native context")
    }
}

impl FfiBackend {
    pub fn new() -> Result<FfiBackend> {
        let ctx = unsafe { sys::pps_ctx_new() };
        if ctx.is_null() {
            return Err(BackendError::new("Failed to create compute context"));
        }
        Ok(FfiBackend {
            inner: Arc::new(FfiInner {
                ctx,
                pending: Mutex::new(HashMap::new()),
            }),
        })
    }

    fn cvt_dev(d: &sys::pps_ocl_dev) -> OclDevice {
        unsafe {
            OclDevice {
                id: d.id,
                name: CStr::from_ptr(d.name.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
                vendor_id: d.vendor_id,
                available: d.available != 0,
                supported: d.supported != 0,
                device_type: CStr::from_ptr(d.type_.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
                memory_global: d.memory_global,
                cache_global: d.cache_global,
                memory_unified: d.memory_unified != 0,
                memory_constant: d.memory_constant,
                memory_local: if d.has_memory_local != 0 {
                    Some(d.memory_local)
                } else {
                    None
                },
                memory_max_alloc: d.memory_max_alloc,
                workgroup_limit: d.workgroup_limit,
                workgroup_multiple: d.workgroup_multiple,
                compute_units: d.compute_units,
            }
        }
    }
}

impl ComputeBackend for FfiBackend {
    type Proc = FfiGfProc;
    type InputHasher = FfiHasherInput;
    type OutputHasher = FfiHasherOutput;

    fn new_proc(
        &mut self,
        slice_size: u64,
        cpu: Option<&CpuSpec>,
        ocl: &[OclSpec],
        staging_areas: i32,
    ) -> Result<FfiGfProc> {
        let cpu_spec = cpu.map(|c| sys::pps_cpu_spec {
            method: c.method,
            input_batchsize: c.input_batchsize,
            input_minbatchsize: c.input_minbatchsize,
            chunk_size: c.chunk_size,
            slice_offset: c.slice_offset,
            slice_size: c.slice_size,
        });
        let ocl_specs: Vec<sys::pps_ocl_spec> = ocl
            .iter()
            .map(|o| sys::pps_ocl_spec {
                platform: o.platform,
                device: o.device,
                slice_size: o.slice_size,
                slice_offset: o.slice_offset,
                method: o.method,
                cksum_method: o.cksum_method,
                input_batchsize: o.input_batchsize,
                input_minbatchsize: o.input_minbatchsize,
                target_iters: o.target_iters,
                target_grouping: o.target_grouping,
            })
            .collect();
        let mut raw: *mut sys::pps_gfp = std::ptr::null_mut();
        let mut eb = ErrBuf::new();
        let rc = unsafe {
            sys::pps_gfp_new(
                self.inner.ctx,
                slice_size,
                cpu_spec
                    .as_ref()
                    .map_or(std::ptr::null(), |c| c as *const _),
                if ocl_specs.is_empty() {
                    std::ptr::null()
                } else {
                    ocl_specs.as_ptr()
                },
                ocl_specs.len(),
                staging_areas,
                &mut raw,
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 || raw.is_null() {
            return Err(last_err(&eb, "Failed to create processor"));
        }
        Ok(FfiGfProc {
            inner: Arc::clone(&self.inner),
            raw,
            closed: false,
        })
    }

    fn wait_event(&mut self, timeout_ms: i32) -> Option<(usize, ProcEvent)> {
        let mut ev = sys::pps_event {
            proc_: std::ptr::null_mut(),
            kind: 0,
            token: 0,
            value: 0,
            err: 0,
            errbuf: [0; 192],
        };
        let rc = unsafe { sys::pps_wait_event(self.inner.ctx, &mut ev, timeout_ms) };
        if rc <= 0 {
            return None;
        }
        let proc_id = ev.proc_ as usize;
        let token = ev.token;
        let ev_kind = ev.kind;
        // a pending-map miss means the event references a token we never
        // parked (stale/foreign event) - surface it rather than handing a
        // fabricated empty buffer downstream
        let take_buf = |key: (usize, u64)| -> Option<SliceBuf> {
            self.inner.pending.lock().unwrap().remove(&key)
        };
        let e = match ev_kind {
            sys::PPS_EV_INPUT_DONE => match take_buf((proc_id, token)) {
                Some(buf) => ProcEvent::InputDone {
                    token,
                    input_index: ev.value,
                    buf,
                },
                None => ProcEvent::Error {
                    token,
                    message: "input completion for unknown token".into(),
                },
            },
            sys::PPS_EV_ENDED => ProcEvent::Ended { token },
            sys::PPS_EV_OUTPUT_DONE => match take_buf((proc_id, token)) {
                Some(buf) => ProcEvent::OutputDone {
                    token,
                    cksum_valid: ev.value != 0,
                    buf,
                },
                None => ProcEvent::Error {
                    token,
                    message: "output completion for unknown token".into(),
                },
            },
            sys::PPS_EV_PROGRESS => ProcEvent::Progress {
                num_inputs: ev.value,
            },
            _ => {
                let msg = unsafe {
                    CStr::from_ptr(ev.errbuf.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                ProcEvent::Error {
                    token,
                    message: msg,
                }
            }
        };
        Some((proc_id, e))
    }

    fn gf_info(method: i32) -> Result<MethodInfo> {
        let mut info = sys::pps_method_info {
            id: 0,
            name: [0; 64],
            alignment: 0,
            stride: 0,
            target_chunk: 0,
            target_grouping: 0,
        };
        if unsafe { sys::pps_gf_info(method, &mut info) } != 0 {
            return Err(BackendError::new("Unknown method"));
        }
        Ok(MethodInfo {
            id: info.id,
            name: unsafe {
                CStr::from_ptr(info.name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            },
            alignment: info.alignment,
            stride: info.stride,
            target_chunk: info.target_chunk,
            target_grouping: info.target_grouping,
        })
    }

    fn set_inhash_method(method: i32) -> bool {
        unsafe { sys::pps_set_inhash(method) == 1 }
    }
    fn set_outhash_method(level: i32) {
        unsafe { sys::pps_set_outhash(level) }
    }
    fn inhash_method_name() -> &'static str {
        unsafe { CStr::from_ptr(sys::pps_inhash_method()) }
            .to_str()
            .unwrap_or("")
    }
    fn outhash_method_name() -> &'static str {
        unsafe { CStr::from_ptr(sys::pps_outhash_method()) }
            .to_str()
            .unwrap_or("")
    }

    fn opencl_platforms(&mut self) -> Result<Vec<OclPlatform>> {
        let mut eb = ErrBuf::new();
        if unsafe { sys::pps_ocl_load(eb.as_mut_ptr(), ERRBUF_LEN) } != 0 {
            return Err(last_err(&eb, "Could not load OpenCL runtime"));
        }
        let n_plat = unsafe { sys::pps_ocl_platform_count() };
        let mut platforms = Vec::with_capacity(n_plat.max(0) as usize);
        for p in 0..n_plat.max(0) {
            let mut name_buf = [0u8; 256];
            let name = if unsafe {
                sys::pps_ocl_platform_name(p, name_buf.as_mut_ptr() as *mut _, name_buf.len())
            } == 0
            {
                unsafe { CStr::from_ptr(name_buf.as_ptr() as *const _) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };
            let n_dev = unsafe { sys::pps_ocl_device_count(p) };
            let mut devices = Vec::with_capacity(n_dev.max(0) as usize);
            for d in 0..n_dev.max(0) {
                let mut dev: sys::pps_ocl_dev = unsafe { std::mem::zeroed() };
                if unsafe { sys::pps_ocl_device_info(p, d, &mut dev) } == 0 {
                    devices.push(Self::cvt_dev(&dev));
                }
            }
            platforms.push(OclPlatform {
                id: p,
                name,
                devices,
            });
        }
        Ok(platforms)
    }

    fn opencl_default_platform(&mut self) -> i32 {
        unsafe { sys::pps_ocl_default_platform() }
    }

    fn opencl_default_device(&mut self, platform: i32) -> i32 {
        unsafe { sys::pps_ocl_default_device(platform) }
    }

    fn opencl_device_info(
        &mut self,
        platform: i32,
        device: i32,
    ) -> Result<Option<(i32, OclDevice)>> {
        // JS: platform -1 resolves to the default platform first
        let platform = if platform == -1 {
            unsafe { sys::pps_ocl_default_platform() }
        } else {
            platform
        };
        if platform < 0 {
            return Ok(None);
        }
        let mut dev: sys::pps_ocl_dev = unsafe { std::mem::zeroed() };
        if unsafe { sys::pps_ocl_device_info(platform, device, &mut dev) } != 0 {
            return Ok(None);
        }
        Ok(Some((platform, Self::cvt_dev(&dev))))
    }

    fn new_input_hasher(
        &mut self,
        slice_size: u64,
        block_out: Box<[u8]>,
    ) -> Result<FfiHasherInput> {
        FfiHasherInput::new(slice_size, block_out)
    }

    fn new_output_hasher(&mut self, regions: u32) -> Result<FfiHasherOutput> {
        FfiHasherOutput::new(regions)
    }
}

#[derive(Debug)]
pub struct FfiGfProc {
    inner: Arc<FfiInner>,
    raw: *mut sys::pps_gfp,
    closed: bool,
}

// Calls are serialised through the shim's libuv thread; the raw handle is
// only used as an identifier + marshal target.
unsafe impl Send for FfiGfProc {}

impl GfProc for FfiGfProc {
    fn proc_id(&self) -> usize {
        self.raw as usize
    }

    fn set_num_threads(&mut self, threads: i32) -> Result<i32> {
        let mut eb = ErrBuf::new();
        let rc =
            unsafe { sys::pps_gfp_set_num_threads(self.raw, threads, eb.as_mut_ptr(), ERRBUF_LEN) };
        if rc < 0 {
            Err(last_err(&eb, "setNumThreads failed"))
        } else {
            Ok(rc)
        }
    }

    fn set_current_slice_size(&mut self, slice_size: u64) -> Result<()> {
        let mut eb = ErrBuf::new();
        let rc = unsafe {
            sys::pps_gfp_set_current_slice_size(self.raw, slice_size, eb.as_mut_ptr(), ERRBUF_LEN)
        };
        if rc != 0 {
            Err(last_err(&eb, "setCurrentSliceSize failed"))
        } else {
            Ok(())
        }
    }

    fn set_recovery_slices(&mut self, exponents: Option<&[u16]>, count: usize) -> Result<()> {
        let mut eb = ErrBuf::new();
        let (ptr, n) = match exponents {
            Some(e) => (e.as_ptr(), e.len()),
            None => (std::ptr::null(), count),
        };
        let rc = unsafe {
            sys::pps_gfp_set_recovery_slices(self.raw, ptr, n, eb.as_mut_ptr(), ERRBUF_LEN)
        };
        if rc != 0 {
            Err(last_err(&eb, "setRecoverySlices failed"))
        } else {
            Ok(())
        }
    }

    fn free_mem(&mut self) -> Result<()> {
        let mut eb = ErrBuf::new();
        let rc = unsafe { sys::pps_gfp_free_mem(self.raw, eb.as_mut_ptr(), ERRBUF_LEN) };
        if rc != 0 {
            Err(last_err(&eb, "freeMem failed"))
        } else {
            Ok(())
        }
    }

    fn add_input(&mut self, buf: SliceBuf, input_index: i32, token: u64) -> Result<AddResult> {
        let mut eb = ErrBuf::new();
        let key = (self.proc_id(), token);
        // park the buffer before the call so the C++ borrow is always owned
        let ptr = buf.data.as_ptr() as *const c_void;
        let len = buf.len;
        self.inner.pending.lock().unwrap().insert(key, buf);
        let rc = unsafe {
            sys::pps_gfp_add_input(
                self.raw,
                ptr,
                len,
                input_index,
                token,
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        match rc {
            0 => Ok(AddResult::Accepted),
            1 => Ok(AddResult::Full(
                self.inner.pending.lock().unwrap().remove(&key).unwrap(),
            )),
            _ => {
                self.inner.pending.lock().unwrap().remove(&key);
                Err(last_err(&eb, "addInput failed"))
            }
        }
    }

    fn end_input(&mut self, token: u64) -> Result<()> {
        let mut eb = ErrBuf::new();
        let rc = unsafe { sys::pps_gfp_end_input(self.raw, token, eb.as_mut_ptr(), ERRBUF_LEN) };
        if rc != 0 {
            Err(last_err(&eb, "endInput failed"))
        } else {
            Ok(())
        }
    }

    fn get_output(&mut self, output_index: i32, out: SliceBuf, token: u64) -> Result<()> {
        let mut eb = ErrBuf::new();
        let key = (self.proc_id(), token);
        let ptr = out.data.as_ptr() as *mut c_void;
        let cap = out.data.len();
        self.inner.pending.lock().unwrap().insert(key, out);
        let rc = unsafe {
            sys::pps_gfp_get_output(
                self.raw,
                output_index,
                ptr,
                cap,
                token,
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 {
            self.inner.pending.lock().unwrap().remove(&key);
            Err(last_err(&eb, "getOutput failed"))
        } else {
            Ok(())
        }
    }

    fn set_progress(&mut self, on: bool) -> Result<()> {
        let mut eb = ErrBuf::new();
        let rc = unsafe {
            sys::pps_gfp_set_progress(
                self.raw,
                if on { 1 } else { 0 },
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 {
            Err(last_err(&eb, "setProgressCb failed"))
        } else {
            Ok(())
        }
    }

    fn info(&mut self) -> Result<GfInfo> {
        let mut eb = ErrBuf::new();
        let mut has_cpu: i32 = 0;
        let mut cpu_info: sys::pps_gf_cpu_info = unsafe { std::mem::zeroed() };
        let mut n_ocl: i32 = 0;
        let rc = unsafe {
            sys::pps_gfp_info(
                self.raw,
                &mut has_cpu,
                &mut cpu_info,
                &mut n_ocl,
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 {
            return Err(last_err(&eb, "info failed"));
        }
        let cpu = if has_cpu != 0 {
            Some(GfCpuInfo {
                threads: cpu_info.threads,
                method_desc: unsafe {
                    CStr::from_ptr(cpu_info.method_desc.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                },
                chunk_size: cpu_info.chunk_size,
                staging_count: cpu_info.staging_count,
                staging_size: cpu_info.staging_size,
                alignment: cpu_info.alignment,
                stride: cpu_info.stride,
                slice_mem: cpu_info.slice_mem,
                num_output_slices: cpu_info.num_output_slices,
            })
        } else {
            None
        };
        let mut ocl = Vec::with_capacity(n_ocl.max(0) as usize);
        for i in 0..n_ocl.max(0) {
            let mut oi: sys::pps_gf_ocl_info = unsafe { std::mem::zeroed() };
            if unsafe { sys::pps_gfp_ocl_info(self.raw, i, &mut oi, eb.as_mut_ptr(), ERRBUF_LEN) }
                != 0
            {
                return Err(last_err(&eb, "info failed"));
            }
            ocl.push(GfOclInfo {
                device_name: unsafe {
                    CStr::from_ptr(oi.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                },
                method_desc: unsafe {
                    CStr::from_ptr(oi.method_desc.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                },
                staging_count: oi.staging_count,
                staging_size: oi.staging_size,
                chunk_size: oi.chunk_size,
                output_chunks: oi.output_chunks,
                slice_mem: oi.slice_mem,
                num_output_slices: oi.num_output_slices,
            });
        }
        Ok(GfInfo { cpu, ocl })
    }

    fn close(&mut self) -> Result<()> {
        let mut eb = ErrBuf::new();
        let rc = unsafe { sys::pps_gfp_close(self.raw, eb.as_mut_ptr(), ERRBUF_LEN) };
        if rc != 0 {
            Err(last_err(&eb, "close failed"))
        } else {
            self.closed = true;
            Ok(())
        }
    }
}

impl Drop for FfiGfProc {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
        }
        // purge parked buffers keyed to this proc: after free the raw
        // pointer may be reused by a new proc, and a stale event could
        // otherwise resolve against the wrong pending entry
        self.inner
            .pending
            .lock()
            .unwrap()
            .retain(|k, _| k.0 != self.raw as usize);
        unsafe { sys::pps_gfp_free(self.raw) }
    }
}

#[derive(Debug)]
pub struct FfiHasherInput {
    raw: *mut sys::pps_hin,
    block_out: Option<Box<[u8]>>,
    ended: bool,
}

// Caller must serialise calls per handle (trait contract), so Send is fine.
unsafe impl Send for FfiHasherInput {}

impl FfiHasherInput {
    fn new(slice_size: u64, mut block_out: Box<[u8]>) -> Result<FfiHasherInput> {
        // slice_size 0 makes the C++ blockLeft==0 loop spin forever
        if slice_size == 0 {
            return Err(BackendError::new("Invalid slice size for input hasher"));
        }
        let raw = unsafe { sys::pps_hin_new(slice_size, block_out.as_mut_ptr(), block_out.len()) };
        if raw.is_null() {
            return Err(BackendError::new("Failed to create input hasher"));
        }
        Ok(FfiHasherInput {
            raw,
            block_out: Some(block_out),
            ended: false,
        })
    }
}

impl HasherInput for FfiHasherInput {
    fn update(&mut self, buf: &[u8]) -> Result<()> {
        // C++ retains the block_out pointer - after end() the buffer belongs
        // to the caller and any write through it would be a UAF
        if self.ended {
            return Err(BackendError::new("Process already ended"));
        }
        let mut eb = ErrBuf::new();
        let rc = unsafe {
            sys::pps_hin_update(
                self.raw,
                buf.as_ptr() as *const c_void,
                buf.len(),
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 {
            Err(last_err(&eb, "update failed"))
        } else {
            Ok(())
        }
    }

    fn end(&mut self, out: &mut [u8]) -> Result<Box<[u8]>> {
        if self.ended {
            return Err(BackendError::new("Process already ended"));
        }
        let mut eb = ErrBuf::new();
        let rc = unsafe {
            sys::pps_hin_end(
                self.raw,
                out.as_mut_ptr(),
                out.len(),
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 {
            return Err(last_err(&eb, "end failed"));
        }
        self.ended = true;
        Ok(self
            .block_out
            .take()
            .unwrap_or_else(|| Vec::new().into_boxed_slice()))
    }

    fn reset(&mut self) -> Result<()> {
        if self.ended {
            return Err(BackendError::new("Process already ended"));
        }
        let mut eb = ErrBuf::new();
        let rc = unsafe { sys::pps_hin_reset(self.raw, eb.as_mut_ptr(), ERRBUF_LEN) };
        if rc != 0 {
            Err(last_err(&eb, "reset failed"))
        } else {
            Ok(())
        }
    }
}

impl Drop for FfiHasherInput {
    fn drop(&mut self) {
        unsafe { sys::pps_hin_free(self.raw) }
    }
}

#[derive(Debug)]
pub struct FfiHasherOutput {
    raw: *mut sys::pps_hout,
    regions: u32,
}

unsafe impl Send for FfiHasherOutput {}

impl FfiHasherOutput {
    fn new(regions: u32) -> Result<FfiHasherOutput> {
        let raw = unsafe { sys::pps_hout_new(regions) };
        if raw.is_null() {
            return Err(BackendError::new("Invalid number of regions specified"));
        }
        Ok(FfiHasherOutput { raw, regions })
    }
}

impl HasherOutput for FfiHasherOutput {
    fn update(&mut self, bufs: &[&[u8]]) -> Result<()> {
        if bufs.len() != self.regions as usize {
            return Err(BackendError::new("Invalid number of array items given"));
        }
        if bufs.is_empty() {
            return Err(BackendError::new("All inputs must be Buffers"));
        }
        let len = bufs[0].len();
        if bufs.iter().any(|b| b.len() != len) {
            return Err(BackendError::new("All inputs' length must be equal"));
        }
        let ptrs: Vec<*const c_void> = bufs.iter().map(|b| b.as_ptr() as *const c_void).collect();
        let mut eb = ErrBuf::new();
        let rc = unsafe {
            sys::pps_hout_update(
                self.raw,
                ptrs.as_ptr(),
                ptrs.len(),
                len,
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 {
            Err(last_err(&eb, "update failed"))
        } else {
            Ok(())
        }
    }

    fn get(&mut self, out: &mut [u8]) -> Result<()> {
        let mut eb = ErrBuf::new();
        let rc = unsafe {
            sys::pps_hout_get(
                self.raw,
                out.as_mut_ptr(),
                out.len(),
                eb.as_mut_ptr(),
                ERRBUF_LEN,
            )
        };
        if rc != 0 {
            Err(last_err(&eb, "get failed"))
        } else {
            Ok(())
        }
    }

    fn reset(&mut self) -> Result<()> {
        let mut eb = ErrBuf::new();
        let rc = unsafe { sys::pps_hout_reset(self.raw, eb.as_mut_ptr(), ERRBUF_LEN) };
        if rc != 0 {
            Err(last_err(&eb, "reset failed"))
        } else {
            Ok(())
        }
    }
}

impl Drop for FfiHasherOutput {
    fn drop(&mut self) {
        unsafe { sys::pps_hout_free(self.raw) }
    }
}

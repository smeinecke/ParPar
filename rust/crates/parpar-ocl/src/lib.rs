//! Minimal safe OpenCL wrapper used by the native OCL GF backend.
//! Mirrors the small subset of the C++ cl.hpp API that
//! controller_ocl*.cpp uses.

pub mod host;
pub mod oclsource;
mod sys;
pub use sys::*;

use parpar_backend::{OclDevice, OclPlatform};
use std::ffi::{c_void, CString};

pub struct Error(pub i32, pub &'static str);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "OpenCL error {} ({})", self.0, self.1)
    }
}
type R<T> = Result<T, Error>;

macro_rules! ck {
    ($e:expr, $w:literal) => {{
        let r = $e;
        if r != CL_SUCCESS {
            return Err(Error(r, $w));
        }
        r
    }};
}

pub fn available() -> bool {
    cl().is_some()
}

// ---------- info helpers ----------
unsafe fn dev_info_string(d: ClDeviceId, what: ClDeviceInfo) -> String {
    let cl = cl().unwrap();
    let mut n = 0usize;
    (cl.clGetDeviceInfo)(d, what, 0, std::ptr::null_mut(), &mut n);
    if n == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; n];
    (cl.clGetDeviceInfo)(d, what, n, buf.as_mut_ptr() as _, &mut n);
    // a racing driver may grow the size between calls - bound by buf len,
    // then strip the NUL terminator if it's within bounds
    let n = n.min(buf.len());
    let n = if n > 0 && buf[n - 1] == 0 { n - 1 } else { n };
    String::from_utf8_lossy(&buf[..n]).to_string()
}

unsafe fn dev_info_scalar<T: Copy + Default>(d: ClDeviceId, what: ClDeviceInfo) -> T {
    let cl = cl().unwrap();
    let mut v = T::default();
    (cl.clGetDeviceInfo)(
        d,
        what,
        std::mem::size_of::<T>(),
        &mut v as *mut T as _,
        std::ptr::null_mut(),
    );
    v
}

unsafe fn plat_info_string(p: ClPlatformId, what: ClPlatformInfo) -> String {
    let cl = cl().unwrap();
    let mut n = 0usize;
    (cl.clGetPlatformInfo)(p, what, 0, std::ptr::null_mut(), &mut n);
    if n == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; n];
    (cl.clGetPlatformInfo)(p, what, n, buf.as_mut_ptr() as _, &mut n);
    let n = n.min(buf.len());
    let n = if n > 0 && buf[n - 1] == 0 { n - 1 } else { n };
    String::from_utf8_lossy(&buf[..n]).to_string()
}

// ---------- RAII wrappers ----------
#[derive(Clone, Copy)]
pub struct Platform(pub ClPlatformId);
#[derive(Clone, Copy)]
pub struct Device(pub ClDeviceId);
pub struct Context(pub ClContext);
pub struct Queue(pub ClQueue);
pub struct Buffer(pub ClMem);
pub struct Program(pub ClProgram);
pub struct Kernel(pub ClKernel);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { (cl().unwrap().clReleaseContext)(self.0) };
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { (cl().unwrap().clReleaseCommandQueue)(self.0) };
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { (cl().unwrap().clReleaseMemObject)(self.0) };
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        unsafe { (cl().unwrap().clReleaseProgram)(self.0) };
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { (cl().unwrap().clReleaseKernel)(self.0) };
    }
}

pub fn platforms() -> Vec<Platform> {
    let Some(cl) = cl() else { return Vec::new() };
    let mut n = 0u32;
    unsafe {
        if (cl.clGetPlatformIDs)(0, std::ptr::null_mut(), &mut n) != CL_SUCCESS || n == 0 {
            return Vec::new();
        }
        let mut ids = vec![std::ptr::null_mut(); n as usize];
        if (cl.clGetPlatformIDs)(n, ids.as_mut_ptr(), std::ptr::null_mut()) != CL_SUCCESS {
            return Vec::new();
        }
        ids.into_iter().map(Platform).collect()
    }
}

impl Platform {
    pub fn name(&self) -> String {
        unsafe { plat_info_string(self.0, CL_PLATFORM_NAME) }
    }
    pub fn devices(&self, ty: ClDeviceType) -> Vec<Device> {
        let cl = cl().unwrap();
        unsafe {
            let mut n = 0u32;
            let r = (cl.clGetDeviceIDs)(self.0, ty, 0, std::ptr::null_mut(), &mut n);
            if (r != CL_SUCCESS && r != CL_DEVICE_NOT_FOUND) || n == 0 {
                return Vec::new();
            }
            let mut ids = vec![std::ptr::null_mut(); n as usize];
            if (cl.clGetDeviceIDs)(self.0, ty, n, ids.as_mut_ptr(), std::ptr::null_mut())
                != CL_SUCCESS
            {
                return Vec::new();
            }
            ids.into_iter().map(Device).collect()
        }
    }
}

impl Device {
    pub fn name(&self) -> String {
        unsafe { dev_info_string(self.0, CL_DEVICE_NAME) }
    }
}

impl Context {
    pub fn create(dev: Device) -> R<Context> {
        let cl = cl().unwrap();
        let mut err = 0;
        let ctx = unsafe {
            (cl.clCreateContext)(
                std::ptr::null(),
                1,
                &dev.0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut err,
            )
        };
        if ctx.is_null() {
            return Err(Error(err, "clCreateContext"));
        }
        Ok(Context(ctx))
    }
}

impl Queue {
    pub fn create(ctx: &Context, dev: Device) -> R<Queue> {
        let cl = cl().unwrap();
        let mut err = 0;
        let q = unsafe { (cl.clCreateCommandQueue)(ctx.0, dev.0, 0, &mut err) };
        if q.is_null() {
            return Err(Error(err, "clCreateCommandQueue"));
        }
        Ok(Queue(q))
    }
    pub fn finish(&self) -> R<()> {
        unsafe { ck!((cl().unwrap().clFinish)(self.0), "clFinish") };
        Ok(())
    }
    pub fn flush(&self) -> R<()> {
        unsafe { ck!((cl().unwrap().clFlush)(self.0), "clFlush") };
        Ok(())
    }
}

impl Buffer {
    pub fn create(ctx: &Context, flags: ClMemFlags, size: usize) -> R<Buffer> {
        let cl = cl().unwrap();
        let mut err = 0;
        let m = unsafe { (cl.clCreateBuffer)(ctx.0, flags, size, std::ptr::null_mut(), &mut err) };
        if m.is_null() {
            return Err(Error(err, "clCreateBuffer"));
        }
        Ok(Buffer(m))
    }
}

impl Program {
    pub fn build(ctx: &Context, dev: Device, source: &str) -> R<Program> {
        Self::build_opts(ctx, dev, source, "")
    }
    pub fn build_opts(ctx: &Context, dev: Device, source: &str, opts: &str) -> R<Program> {
        let cl = cl().unwrap();
        let mut err = 0;
        let src = CString::new(source).unwrap();
        let opts_c = CString::new(opts).unwrap();
        let len = source.len();
        let prog =
            unsafe { (cl.clCreateProgramWithSource)(ctx.0, 1, &src.as_ptr(), &len, &mut err) };
        if prog.is_null() {
            return Err(Error(err, "clCreateProgramWithSource"));
        }
        let p = Program(prog);
        let r = unsafe {
            (cl.clBuildProgram)(
                p.0,
                1,
                &dev.0,
                opts_c.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if r != CL_SUCCESS {
            // fetch build log for diagnostics
            let mut n = 0usize;
            unsafe {
                (cl.clGetProgramBuildInfo)(
                    p.0,
                    dev.0,
                    CL_PROGRAM_BUILD_LOG,
                    0,
                    std::ptr::null_mut(),
                    &mut n,
                );
                let mut log = vec![0u8; n.max(1)];
                (cl.clGetProgramBuildInfo)(
                    p.0,
                    dev.0,
                    CL_PROGRAM_BUILD_LOG,
                    log.len(),
                    log.as_mut_ptr() as _,
                    &mut n,
                );
                // drivers may report the required size rather than bytes
                // written - clamp before slicing off the NUL
                let used = n.min(log.len()).saturating_sub(1);
                eprintln!(
                    "OpenCL build log: {}",
                    String::from_utf8_lossy(&log[..used])
                );
            }
            return Err(Error(r, "clBuildProgram"));
        }
        Ok(p)
    }
}

impl Kernel {
    pub fn create(prog: &Program, name: &str) -> R<Kernel> {
        let cl = cl().unwrap();
        let cname = CString::new(name).unwrap();
        let mut err = 0;
        let k = unsafe { (cl.clCreateKernel)(prog.0, cname.as_ptr(), &mut err) };
        if k.is_null() {
            return Err(Error(err, "clCreateKernel"));
        }
        Ok(Kernel(k))
    }
    pub unsafe fn set_arg(&self, idx: u32, val: *const c_void, size: usize) -> R<()> {
        ck!(
            (cl().unwrap().clSetKernelArg)(self.0, idx, size, val),
            "clSetKernelArg"
        );
        Ok(())
    }
    pub fn set_mem(&self, idx: u32, buf: &Buffer) -> R<()> {
        unsafe {
            self.set_arg(
                idx,
                &buf.0 as *const ClMem as _,
                std::mem::size_of::<ClMem>(),
            )
        }
    }
    pub fn set_u32(&self, idx: u32, v: u32) -> R<()> {
        unsafe { self.set_arg(idx, &v as *const u32 as _, 4) }
    }
    pub fn set_u64(&self, idx: u32, v: u64) -> R<()> {
        unsafe { self.set_arg(idx, &v as *const u64 as _, 8) }
    }
    pub fn set_u16(&self, idx: u32, v: u16) -> R<()> {
        unsafe { self.set_arg(idx, &v as *const u16 as _, 2) }
    }
    pub fn set_local(&self, idx: u32, size: usize) -> R<()> {
        unsafe { self.set_arg(idx, std::ptr::null(), size) }
    }
    /// preferred work-group size multiple (the getWGSize probe)
    pub fn preferred_multiple(&self, dev: Device) -> R<usize> {
        let cl = cl().unwrap();
        let mut v = 0usize;
        unsafe {
            ck!(
                (cl.clGetKernelWorkGroupInfo)(
                    self.0,
                    dev.0,
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                    std::mem::size_of::<usize>(),
                    &mut v as *mut usize as _,
                    std::ptr::null_mut()
                ),
                "clGetKernelWorkGroupInfo"
            );
        }
        Ok(v)
    }
}

impl Queue {
    /// Blocking write - must be CL_TRUE: callers hand in temporary buffers,
    /// and a non-blocking write would let the driver DMA from freed host
    /// memory after return.
    pub fn write(&self, buf: &Buffer, off: usize, data: &[u8]) -> R<()> {
        unsafe {
            ck!(
                (cl().unwrap().clEnqueueWriteBuffer)(
                    self.0,
                    buf.0,
                    CL_TRUE,
                    off,
                    data.len(),
                    data.as_ptr() as _,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut()
                ),
                "clEnqueueWriteBuffer"
            );
        }
        Ok(())
    }
    pub fn map(&self, buf: &Buffer, flags: ClMapFlags, off: usize, len: usize) -> R<*mut c_void> {
        let cl = cl().unwrap();
        let mut err = 0;
        let p = unsafe {
            (cl.clEnqueueMapBuffer)(
                self.0,
                buf.0,
                CL_TRUE,
                flags,
                off,
                len,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut err,
            )
        };
        if p.is_null() {
            return Err(Error(err, "clEnqueueMapBuffer"));
        }
        Ok(p)
    }
    // p is the opaque driver mapping returned by map(); only the driver
    // dereferences it, not Rust
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn unmap(&self, buf: &Buffer, p: *mut c_void) -> R<()> {
        unsafe {
            ck!(
                (cl().unwrap().clEnqueueUnmapMemObject)(
                    self.0,
                    buf.0,
                    p,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut()
                ),
                "clEnqueueUnmapMemObject"
            );
        }
        Ok(())
    }
    pub fn ndrange(&self, k: &Kernel, global: &[usize], local: &[usize]) -> R<()> {
        // the completion event isn't needed - release it rather than leak
        let ev = self.ndrange_ev(k, global, local, &[])?;
        release_event(ev);
        Ok(())
    }
    /// Enqueue an NDRange kernel with optional wait events; returns the
    /// completion event.
    pub fn ndrange_ev(
        &self,
        k: &Kernel,
        global: &[usize],
        local: &[usize],
        wait: &[ClEvent],
    ) -> R<ClEvent> {
        let mut ev: ClEvent = std::ptr::null_mut();
        unsafe {
            ck!(
                (cl().unwrap().clEnqueueNDRangeKernel)(
                    self.0,
                    k.0,
                    global.len() as u32,
                    std::ptr::null(),
                    global.as_ptr(),
                    if local.is_empty() {
                        std::ptr::null()
                    } else {
                        local.as_ptr()
                    },
                    wait.len() as u32,
                    if wait.is_empty() {
                        std::ptr::null()
                    } else {
                        wait.as_ptr()
                    },
                    &mut ev
                ),
                "clEnqueueNDRangeKernel"
            );
        }
        Ok(ev)
    }
    // (no write_ev: non-blocking writes of temporary host buffers are a
    //  use-after-free hazard - use `write`, which is CL_TRUE/blocking)
    /// Nonblocking map (unlike `map` which is blocking) - returns ptr + event.
    pub fn map_ev(
        &self,
        buf: &Buffer,
        flags: ClMapFlags,
        off: usize,
        len: usize,
        wait: &[ClEvent],
    ) -> R<(*mut c_void, ClEvent)> {
        let cl = cl().unwrap();
        let mut err = 0;
        let mut ev: ClEvent = std::ptr::null_mut();
        let p = unsafe {
            (cl.clEnqueueMapBuffer)(
                self.0,
                buf.0,
                CL_FALSE,
                flags,
                off,
                len,
                wait.len() as u32,
                if wait.is_empty() {
                    std::ptr::null()
                } else {
                    wait.as_ptr()
                },
                &mut ev,
                &mut err,
            )
        };
        if p.is_null() {
            return Err(Error(err, "clEnqueueMapBuffer"));
        }
        Ok((p, ev))
    }
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // opaque driver handle
    pub fn unmap_ev(&self, buf: &Buffer, p: *mut c_void, wait: &[ClEvent]) -> R<ClEvent> {
        let mut ev: ClEvent = std::ptr::null_mut();
        unsafe {
            ck!(
                (cl().unwrap().clEnqueueUnmapMemObject)(
                    self.0,
                    buf.0,
                    p,
                    wait.len() as u32,
                    if wait.is_empty() {
                        std::ptr::null()
                    } else {
                        wait.as_ptr()
                    },
                    &mut ev
                ),
                "clEnqueueUnmapMemObject"
            );
        }
        Ok(ev)
    }
    /// Blocking wait for an event list.
    pub fn wait_events(evs: &[ClEvent]) -> R<()> {
        unsafe {
            ck!(
                (cl().unwrap().clWaitForEvents)(evs.len() as u32, evs.as_ptr()),
                "clWaitForEvents"
            );
        }
        Ok(())
    }
}

/// Event status query (CL_EVENT_COMMAND_EXECUTION_STATUS).
/// Returns None when the query itself fails - callers must NOT treat a
/// failed query as "done" (a stuck -1 status is <= CL_COMPLETE, which
/// would free an in-flight staging area).
#[allow(clippy::not_unsafe_ptr_arg_deref)] // opaque driver handle
pub fn event_status(ev: ClEvent) -> Option<ClInt> {
    let mut v: ClInt = -1;
    let rc = unsafe {
        (cl().unwrap().clGetEventInfo)(
            ev,
            CL_EVENT_COMMAND_EXECUTION_STATUS,
            4,
            &mut v as *mut ClInt as _,
            std::ptr::null_mut(),
        )
    };
    (rc == CL_SUCCESS).then_some(v)
}

/// Release an event handle.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // opaque driver handle
pub fn release_event(ev: ClEvent) {
    unsafe { (cl().unwrap().clReleaseEvent)(ev) };
}

// ---------- enumeration matching GF16OCL_DeviceInfo ----------

/// Probe CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE by compiling a
/// trivial kernel (port of getWGSize from controller_ocl.cpp).
fn probe_wg_multiple(ctx: &Context, dev: Device) -> u32 {
    const SRC: &str = "#define GWO(type) global type* restrict\n\
                       #define GRO(type) global const type* restrict\n\
                       #define BODY int i = get_global_id(0); out[i] = in1[i] + in2[i]\n\
                       #define _KRN(T, N) kernel void sum##N(GWO(T##N) out, GRO(T##N) in1, GRO(T##N) in2) { BODY; }\n\
                       #define KRN(N) _KRN(int, N)\n\
                       KRN()\n";
    let Ok(prog) = Program::build(ctx, dev, SRC) else {
        return 0;
    };
    let Ok(k) = Kernel::create(&prog, "sum") else {
        return 0;
    };
    k.preferred_multiple(dev).unwrap_or(0) as u32
}

fn device_type_name(ty: ClDeviceType) -> String {
    if ty & CL_DEVICE_TYPE_GPU != 0 {
        "GPU".to_string()
    } else if ty & (1 << 1) != 0 {
        "CPU".to_string()
    } else if ty & (1 << 3) != 0 {
        "Accelerator".to_string()
    } else {
        "Unknown".to_string()
    }
}

/// Enumerate all platforms and their devices as backend types -
/// mirrors PAR2ProcOCL::getDevices + GF16OCL_DeviceInfo.
pub fn enumerate() -> Vec<OclPlatform> {
    if !available() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (pid, plat) in platforms().into_iter().enumerate() {
        let devices = plat.devices(CL_DEVICE_TYPE_ALL);
        let mut devs = Vec::with_capacity(devices.len());
        for (did, dev) in devices.iter().enumerate() {
            let ty: ClDeviceType = unsafe { dev_info_scalar(dev.0, CL_DEVICE_TYPE) };
            let available_: ClBool = unsafe { dev_info_scalar(dev.0, CL_DEVICE_AVAILABLE) };
            let compiler: ClBool = unsafe { dev_info_scalar(dev.0, CL_DEVICE_COMPILER_AVAILABLE) };
            let endian: ClBool = unsafe { dev_info_scalar(dev.0, CL_DEVICE_ENDIAN_LITTLE) };
            let available = available_ != 0 && compiler != 0;
            let local_mem_type: ClDeviceLocalMemType =
                unsafe { dev_info_scalar(dev.0, CL_DEVICE_LOCAL_MEM_TYPE) };
            let global: u64 = unsafe { dev_info_scalar(dev.0, CL_DEVICE_GLOBAL_MEM_SIZE) };
            let local: u64 = unsafe { dev_info_scalar(dev.0, CL_DEVICE_LOCAL_MEM_SIZE) };
            let local_is_global = local_mem_type == CL_GLOBAL;

            // workgroup multiple needs a context - build the probe lazily
            let wg_multiple = if available {
                match Context::create(*dev).and_then(|c| Ok(probe_wg_multiple(&c, *dev))) {
                    Ok(v) => v,
                    Err(_) => 0,
                }
            } else {
                0
            };

            devs.push(OclDevice {
                id: did as i32,
                name: dev.name(),
                vendor_id: unsafe { dev_info_scalar::<u32>(dev.0, CL_DEVICE_VENDOR_ID) } as i32,
                available,
                supported: available && endian != 0,
                device_type: device_type_name(ty),
                memory_global: global,
                cache_global: unsafe { dev_info_scalar(dev.0, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) },
                memory_unified: unsafe {
                    dev_info_scalar::<ClBool>(dev.0, CL_DEVICE_HOST_UNIFIED_MEMORY)
                } != 0,
                memory_constant: unsafe {
                    dev_info_scalar(dev.0, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)
                },
                memory_local: if local_is_global { None } else { Some(local) },
                memory_max_alloc: unsafe { dev_info_scalar(dev.0, CL_DEVICE_MAX_MEM_ALLOC_SIZE) },
                workgroup_limit: unsafe {
                    dev_info_scalar::<usize>(dev.0, CL_DEVICE_MAX_WORK_GROUP_SIZE)
                } as u32,
                workgroup_multiple: wg_multiple,
                compute_units: unsafe { dev_info_scalar(dev.0, CL_DEVICE_MAX_COMPUTE_UNITS) },
            });
        }
        out.push(OclPlatform {
            id: pid as i32,
            name: plat.name(),
            devices: devs,
        });
    }
    out
}

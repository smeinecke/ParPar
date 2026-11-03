//! Raw OpenCL API loaded at runtime via dlopen - keeps libOpenCL an
//! optional dependency: `load()` returns None when no ICD is present.

#[cfg(unix)]
use std::ffi::CStr;
use std::ffi::{c_char, c_int, c_uint, c_ulong, c_void, CString};
use std::sync::OnceLock;

pub type ClPlatformId = *mut c_void;
pub type ClDeviceId = *mut c_void;
pub type ClContext = *mut c_void;
pub type ClQueue = *mut c_void;
pub type ClMem = *mut c_void;
pub type ClProgram = *mut c_void;
pub type ClKernel = *mut c_void;
pub type ClEvent = *mut c_void;
pub type ClInt = c_int;
pub type ClUint = c_uint;
pub type ClUlong = c_ulong;
pub type ClBool = ClUint;
pub type ClDeviceType = u64;
pub type ClDeviceLocalMemType = c_uint;
pub type ClMapFlags = u64;
pub type ClMemFlags = u64;
pub type ClDeviceInfo = c_uint;
pub type ClPlatformInfo = c_uint;
pub type ClKernelWorkGroupInfo = c_uint;
pub type ClContextProperties = isize;
pub type ClQueueProperties = isize;
pub type ClProgramBuildInfo = c_uint;
pub type ClKernelInfo = c_uint;

pub const CL_SUCCESS: ClInt = 0;
pub const CL_DEVICE_NOT_FOUND: ClInt = -1;
pub const CL_DEVICE_TYPE_ALL: ClDeviceType = 0xFFFFFFFF;
pub const CL_DEVICE_TYPE_GPU: ClDeviceType = 1 << 2;
pub const CL_DEVICE_AVAILABLE: ClDeviceInfo = 0x1027;
pub const CL_DEVICE_COMPILER_AVAILABLE: ClDeviceInfo = 0x1028;
pub const CL_DEVICE_ENDIAN_LITTLE: ClDeviceInfo = 0x1026;
pub const CL_DEVICE_NAME: ClDeviceInfo = 0x102B;
pub const CL_DEVICE_VENDOR_ID: ClDeviceInfo = 0x1001;
pub const CL_DEVICE_TYPE: ClDeviceInfo = 0x1000;
pub const CL_DEVICE_GLOBAL_MEM_SIZE: ClDeviceInfo = 0x101F;
pub const CL_DEVICE_MAX_WORK_GROUP_SIZE: ClDeviceInfo = 0x1004;
pub const CL_DEVICE_MAX_COMPUTE_UNITS: ClDeviceInfo = 0x1002;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: ClDeviceInfo = 0x101D;
pub const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: ClDeviceInfo = 0x1020;
pub const CL_DEVICE_LOCAL_MEM_SIZE: ClDeviceInfo = 0x1023;
pub const CL_DEVICE_LOCAL_MEM_TYPE: ClDeviceInfo = 0x1022;
pub const CL_DEVICE_MAX_MEM_ALLOC_SIZE: ClDeviceInfo = 0x1010;
pub const CL_DEVICE_HOST_UNIFIED_MEMORY: ClDeviceInfo = 0x1035;
pub const CL_PLATFORM_NAME: ClPlatformInfo = 0x0902;
pub const CL_TRUE: ClBool = 1;
pub const CL_FALSE: ClBool = 0;
pub const CL_MAP_READ: ClMapFlags = 1;
pub const CL_MAP_WRITE: ClMapFlags = 1 << 1;
pub const CL_MAP_WRITE_INVALIDATE_REGION: ClMapFlags = 1 << 2;
pub const CL_MEM_READ_WRITE: ClMemFlags = 1 << 0;
pub const CL_GLOBAL: ClDeviceLocalMemType = 1;
pub const CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: ClKernelWorkGroupInfo = 0x11B3;
pub const CL_PROGRAM_BUILD_LOG: ClProgramBuildInfo = 0x1183;
pub const CL_KERNEL_FUNCTION_NAME: ClKernelInfo = 0x1190;
pub const CL_KERNEL_NUM_ARGS: ClKernelInfo = 0x1172;
pub const CL_PLATFORM_VERSION: ClPlatformInfo = 0x0901;
pub const CL_DEVICE_VERSION: ClDeviceInfo = 0x102F;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: ClDeviceInfo = 0x1007;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: ClDeviceInfo = 0x1006;
pub const CL_MEM_READ_ONLY: ClMemFlags = 1 << 2;
pub const CL_MEM_ALLOC_HOST_PTR: ClMemFlags = 1 << 4;
pub const CL_MEM_HOST_WRITE_ONLY: ClMemFlags = 1 << 7;
pub const CL_MEM_HOST_READ_ONLY: ClMemFlags = 1 << 8;
pub const CL_EVENT_COMMAND_EXECUTION_STATUS: c_uint = 0x11D3;
pub const CL_COMPLETE: ClInt = 0x0;

pub type ClEventInfo = c_uint;

#[allow(non_snake_case)]
pub struct Cl {
    pub clGetPlatformIDs: unsafe extern "C" fn(c_uint, *mut ClPlatformId, *mut c_uint) -> ClInt,
    pub clGetPlatformInfo:
        unsafe extern "C" fn(ClPlatformId, ClPlatformInfo, usize, *mut c_void, *mut usize) -> ClInt,
    pub clGetDeviceIDs: unsafe extern "C" fn(
        ClPlatformId,
        ClDeviceType,
        c_uint,
        *mut ClDeviceId,
        *mut c_uint,
    ) -> ClInt,
    pub clGetDeviceInfo:
        unsafe extern "C" fn(ClDeviceId, ClDeviceInfo, usize, *mut c_void, *mut usize) -> ClInt,
    pub clCreateContext: unsafe extern "C" fn(
        *const ClContextProperties,
        c_uint,
        *const ClDeviceId,
        *mut c_void,
        *mut c_void,
        *mut ClInt,
    ) -> ClContext,
    pub clCreateCommandQueue:
        unsafe extern "C" fn(ClContext, ClDeviceId, ClQueueProperties, *mut ClInt) -> ClQueue,
    pub clCreateBuffer:
        unsafe extern "C" fn(ClContext, ClMemFlags, usize, *mut c_void, *mut ClInt) -> ClMem,
    pub clEnqueueWriteBuffer: unsafe extern "C" fn(
        ClQueue,
        ClMem,
        ClBool,
        usize,
        usize,
        *const c_void,
        c_uint,
        *const ClEvent,
        *mut ClEvent,
    ) -> ClInt,
    pub clEnqueueMapBuffer: unsafe extern "C" fn(
        ClQueue,
        ClMem,
        ClBool,
        ClMapFlags,
        usize,
        usize,
        c_uint,
        *const ClEvent,
        *mut ClEvent,
        *mut ClInt,
    ) -> *mut c_void,
    pub clEnqueueUnmapMemObject: unsafe extern "C" fn(
        ClQueue,
        ClMem,
        *mut c_void,
        c_uint,
        *const ClEvent,
        *mut ClEvent,
    ) -> ClInt,
    pub clCreateProgramWithSource: unsafe extern "C" fn(
        ClContext,
        c_uint,
        *const *const c_char,
        *const usize,
        *mut ClInt,
    ) -> ClProgram,
    pub clBuildProgram: unsafe extern "C" fn(
        ClProgram,
        c_uint,
        *const ClDeviceId,
        *const c_char,
        *mut c_void,
        *mut c_void,
    ) -> ClInt,
    pub clGetProgramBuildInfo: unsafe extern "C" fn(
        ClProgram,
        ClDeviceId,
        ClProgramBuildInfo,
        usize,
        *mut c_void,
        *mut usize,
    ) -> ClInt,
    pub clCreateKernel: unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel,
    pub clGetKernelInfo:
        unsafe extern "C" fn(ClKernel, ClKernelInfo, usize, *mut c_void, *mut usize) -> ClInt,
    pub clGetKernelWorkGroupInfo: unsafe extern "C" fn(
        ClKernel,
        ClDeviceId,
        ClKernelWorkGroupInfo,
        usize,
        *mut c_void,
        *mut usize,
    ) -> ClInt,
    pub clSetKernelArg: unsafe extern "C" fn(ClKernel, c_uint, usize, *const c_void) -> ClInt,
    pub clEnqueueNDRangeKernel: unsafe extern "C" fn(
        ClQueue,
        ClKernel,
        c_uint,
        *const usize,
        *const usize,
        *const usize,
        c_uint,
        *const ClEvent,
        *mut ClEvent,
    ) -> ClInt,
    pub clFlush: unsafe extern "C" fn(ClQueue) -> ClInt,
    pub clFinish: unsafe extern "C" fn(ClQueue) -> ClInt,
    pub clWaitForEvents: unsafe extern "C" fn(c_uint, *const ClEvent) -> ClInt,
    pub clGetEventInfo:
        unsafe extern "C" fn(ClEvent, ClEventInfo, usize, *mut c_void, *mut usize) -> ClInt,
    pub clReleaseEvent: unsafe extern "C" fn(ClEvent) -> ClInt,
    pub clReleaseMemObject: unsafe extern "C" fn(ClMem) -> ClInt,
    pub clReleaseKernel: unsafe extern "C" fn(ClKernel) -> ClInt,
    pub clReleaseProgram: unsafe extern "C" fn(ClProgram) -> ClInt,
    pub clReleaseCommandQueue: unsafe extern "C" fn(ClQueue) -> ClInt,
    pub clReleaseContext: unsafe extern "C" fn(ClContext) -> ClInt,
}

// Runtime loader - same library names/semantics as gf16/opencl-include/cl.c:
// POSIX: libOpenCL.so -> .so.1 -> .so.1.0.0 via dlopen; Windows: OpenCL.dll
// via LoadLibrary. PARPAR_OCL_LIBRARY overrides the search.
#[cfg(unix)]
mod dl {
    use super::*;
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *mut c_char;
    }
    // RTLD_LOCAL is 0 - do not OR it in. Value 4 is RTLD_NOLOAD, which
    // silently refuses to load anything not already mapped.
    const RTLD_NOW: c_int = 2;
    pub const CANDIDATES: [&str; 3] = ["libOpenCL.so", "libOpenCL.so.1", "libOpenCL.so.1.0.0"];
    pub fn open(name: &str) -> *mut c_void {
        let cname = CString::new(name).unwrap();
        unsafe { dlopen(cname.as_ptr(), RTLD_NOW) }
    }
    pub fn sym(h: *mut c_void, name: &str) -> *mut c_void {
        let s = CString::new(name).unwrap();
        unsafe { dlsym(h, s.as_ptr()) }
    }
    pub fn close(h: *mut c_void) {
        unsafe { dlclose(h) };
    }
    pub fn err() -> Option<String> {
        let e = unsafe { dlerror() };
        if e.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned())
        }
    }
}

#[cfg(windows)]
mod dl {
    use super::*;
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }
    pub const CANDIDATES: [&str; 1] = ["OpenCL.dll"];
    pub fn open(name: &str) -> *mut c_void {
        let w: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe { LoadLibraryW(w.as_ptr()) }
    }
    pub fn sym(h: *mut c_void, name: &str) -> *mut c_void {
        let s = CString::new(name).unwrap();
        unsafe { GetProcAddress(h, s.as_ptr()) }
    }
    pub fn close(h: *mut c_void) {
        unsafe { FreeLibrary(h) };
    }
    pub fn err() -> Option<String> {
        None
    }
}

static CL: OnceLock<Option<Cl>> = OnceLock::new();

/// Get the loaded OpenCL table, or None when no ICD is installed.
pub fn cl() -> Option<&'static Cl> {
    CL.get_or_init(load_cl).as_ref()
}

fn load_cl() -> Option<Cl> {
    // PARPAR_OCL_LIBRARY overrides the soname search (tests + custom ICDs)
    let candidates: Vec<String> = match std::env::var("PARPAR_OCL_LIBRARY") {
        Ok(v) if !v.is_empty() => vec![v],
        _ => dl::CANDIDATES.iter().map(|s| s.to_string()).collect(),
    };
    for name in &candidates {
        let h = dl::open(name);
        if h.is_null() {
            if let Some(e) = dl::err() {
                eprintln!("dlopen {}: {}", name, e);
            }
            continue;
        }
        macro_rules! sym {
            ($n:literal) => {{
                let p = dl::sym(h, $n);
                if p.is_null() {
                    dl::close(h);
                    eprintln!("libOpenCL: missing symbol {}", $n);
                    return None;
                }
                unsafe { std::mem::transmute(p) }
            }};
        }
        return Some(Cl {
            clGetPlatformIDs: sym!("clGetPlatformIDs"),
            clGetPlatformInfo: sym!("clGetPlatformInfo"),
            clGetDeviceIDs: sym!("clGetDeviceIDs"),
            clGetDeviceInfo: sym!("clGetDeviceInfo"),
            clCreateContext: sym!("clCreateContext"),
            clCreateCommandQueue: sym!("clCreateCommandQueue"),
            clCreateBuffer: sym!("clCreateBuffer"),
            clEnqueueWriteBuffer: sym!("clEnqueueWriteBuffer"),
            clEnqueueMapBuffer: sym!("clEnqueueMapBuffer"),
            clEnqueueUnmapMemObject: sym!("clEnqueueUnmapMemObject"),
            clCreateProgramWithSource: sym!("clCreateProgramWithSource"),
            clBuildProgram: sym!("clBuildProgram"),
            clGetProgramBuildInfo: sym!("clGetProgramBuildInfo"),
            clCreateKernel: sym!("clCreateKernel"),
            clGetKernelInfo: sym!("clGetKernelInfo"),
            clGetKernelWorkGroupInfo: sym!("clGetKernelWorkGroupInfo"),
            clSetKernelArg: sym!("clSetKernelArg"),
            clEnqueueNDRangeKernel: sym!("clEnqueueNDRangeKernel"),
            clFlush: sym!("clFlush"),
            clFinish: sym!("clFinish"),
            clWaitForEvents: sym!("clWaitForEvents"),
            clGetEventInfo: sym!("clGetEventInfo"),
            clReleaseEvent: sym!("clReleaseEvent"),
            clReleaseMemObject: sym!("clReleaseMemObject"),
            clReleaseKernel: sym!("clReleaseKernel"),
            clReleaseProgram: sym!("clReleaseProgram"),
            clReleaseCommandQueue: sym!("clReleaseCommandQueue"),
            clReleaseContext: sym!("clReleaseContext"),
        });
    }
    None
}

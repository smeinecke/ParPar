//! `NativeBackend` - a `ComputeBackend` implemented entirely over the native
//! Rust crates (parpar-gf16, parpar-hasher, parpar-ocl). This is the
//! production backend once cutover completes; `FfiBackend` remains as the
//! differential oracle.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use parpar_backend::{
    BackendError, ComputeBackend, CpuSpec, GfProc, MethodInfo, OclDevice, OclPlatform, OclSpec,
    ProcEvent, RangeBackend, Result,
};
use parpar_gf16::{CpuRangeBackend, EventQueues, NativeGfProc};

pub struct NativeBackend {
    queues: EventQueues,
    next_pid: usize,
}

impl Default for NativeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeBackend {
    pub fn new() -> Self {
        NativeBackend {
            queues: Default::default(),
            next_pid: 1,
        }
    }

    fn alloc_pid(&mut self) -> usize {
        let pid = self.next_pid;
        self.next_pid += 1;
        self.queues.lock().unwrap().insert(pid, VecDeque::new());
        pid
    }
}

/// Build a native proc - shared by `NativeBackend::new_proc` and the
/// SharedBackend native-proc coexistence path. Mirrors `pps_gfp_new`.
pub fn native_new_proc(
    pid: usize,
    queues: EventQueues,
    slice_size: u64,
    cpu: Option<&CpuSpec>,
    ocl: &[OclSpec],
    staging: i32,
) -> Result<Box<dyn GfProc>> {
    if ocl.is_empty() {
        return Ok(Box::new(NativeGfProc::new(
            pid, queues, slice_size, cpu, staging,
        )?));
    }
    // composite CPU+OCL path (mirrors pps_gfp_new in the shim):
    // OCL backends first, then the CPU range backend last
    let sz = slice_size as usize;
    let staging_areas = if staging <= 0 { 2 } else { staging };
    let mut backends: Vec<(Box<dyn parpar_backend::RangeBackend>, usize, usize)> = Vec::new();
    let mut used_slice_size = 0usize;
    for o in ocl {
        if o.input_batchsize < 0 || o.input_batchsize > 32768 {
            return Err(BackendError::new("OpenCL input batchsize is invalid"));
        }
        if o.target_grouping > 65535 {
            return Err(BackendError::new("OpenCL target grouping is invalid"));
        }
        let off = o.slice_offset as usize;
        let size = o.slice_size as usize;
        if size == 0 || size & 1 != 0 {
            return Err(BackendError::new(
                "Invalid slice size allocated to OpenCL device",
            ));
        }
        if off & 1 != 0 || off > sz {
            return Err(BackendError::new(
                "Invalid slice offset allocated to OpenCL device",
            ));
        }
        if off + size > sz {
            return Err(BackendError::new(
                "Invalid slice offset+size allocated to OpenCL device",
            ));
        }
        // cksumMethod: 0 inherits the CPU spec's GF method; the checksum
        // width is the method's alignment floored at 8
        let ck_method = if o.cksum_method != 0 {
            o.cksum_method
        } else {
            cpu.map(|c| c.method).unwrap_or(0)
        };
        let cksum_size = parpar_gf16::pick(ck_method).alignment().max(8);
        let mut be =
            parpar_ocl::host::OclBackend::new(o.platform, o.device, staging_areas as usize)?;
        be.set_slice_size(size);
        be.init(
            o.method,
            o.input_batchsize.max(0) as u32,
            o.input_minbatchsize,
            o.target_iters,
            o.target_grouping,
            cksum_size,
        )?;
        backends.push((Box::new(be), off, size));
        used_slice_size += size;
    }
    if let Some(c) = cpu {
        let off = c.slice_offset as usize;
        if off & 1 != 0 || off > sz {
            return Err(BackendError::new("Invalid CPU slice offset"));
        }
        let size = if c.slice_size != 0 {
            c.slice_size as usize
        } else {
            sz
        };
        if size < 2 || size & 1 != 0 {
            return Err(BackendError::new("CPU slice size must be a multiple of 2"));
        }
        if off + size > sz {
            return Err(BackendError::new(
                "CPU slice offset+size cannot exceed the slice size",
            ));
        }
        let be = CpuRangeBackend::new(pid, size, Some(c), staging_areas)?;
        backends.push((Box::new(be), off, size));
    }
    if (cpu.is_some() && used_slice_size >= sz) || (cpu.is_none() && used_slice_size != sz) {
        return Err(BackendError::new(
            "Slice portions allocated to OpenCL devices is invalid",
        ));
    }
    Ok(Box::new(parpar_backend::CompositeProc::new(
        pid, queues, sz, backends,
    )?))
}

impl ComputeBackend for NativeBackend {
    type Proc = Box<dyn GfProc>;
    type InputHasher = parpar_hasher::NativeHasherInput;
    type OutputHasher = parpar_hasher::NativeHasherOutput;

    fn new_proc(
        &mut self,
        slice_size: u64,
        cpu: Option<&CpuSpec>,
        ocl: &[OclSpec],
        staging: i32,
    ) -> Result<Self::Proc> {
        let pid = self.alloc_pid();
        native_new_proc(pid, self.queues.clone(), slice_size, cpu, ocl, staging)
    }

    fn wait_event(&mut self, timeout_ms: i32) -> Option<(usize, ProcEvent)> {
        // Native procs post events synchronously from add_input/get_output on
        // the calling thread, so events are almost always already queued; the
        // poll loop only spins for threaded CPU procs.
        let deadline =
            (timeout_ms >= 0).then(|| Instant::now() + Duration::from_millis(timeout_ms as u64));
        loop {
            {
                let mut qs = self.queues.lock().unwrap();
                for (pid, q) in qs.iter_mut() {
                    if let Some(ev) = q.pop_front() {
                        return Some((*pid, ev));
                    }
                }
            }
            match deadline {
                Some(d) if Instant::now() >= d => return None,
                _ => std::thread::sleep(Duration::from_micros(200)),
            }
        }
    }

    fn gf_info(method: i32) -> Result<MethodInfo> {
        Ok(parpar_gf16::method_info(method))
    }

    fn set_inhash_method(_method: i32) -> bool {
        // native hasher auto-selects its best implementation; output is
        // identical regardless of method choice (MD5+CRC32 are fixed)
        true
    }

    fn set_outhash_method(_level: i32) {}

    fn inhash_method_name() -> &'static str {
        parpar_hasher::inhash_method_name()
    }

    fn outhash_method_name() -> &'static str {
        parpar_hasher::outhash_method_name()
    }

    fn opencl_platforms(&mut self) -> Result<Vec<OclPlatform>> {
        if !parpar_ocl::available() {
            return Err(BackendError::new("Could not load OpenCL runtime"));
        }
        Ok(parpar_ocl::enumerate())
    }

    fn opencl_default_platform(&mut self) -> i32 {
        // cl::Platform::getDefault() returns the first platform - matching it
        // against the enumerated list yields index 0 when any exist
        if parpar_ocl::available() && !parpar_ocl::enumerate().is_empty() {
            0
        } else {
            -1
        }
    }

    fn opencl_default_device(&mut self, platform: i32) -> i32 {
        let plats = parpar_ocl::enumerate();
        let Some(plat) = plats.get(platform.max(0) as usize) else {
            return -1;
        };
        // first available GPU, else first available device (defaultDeviceId)
        let mut first_avail = -1;
        for d in &plat.devices {
            if d.available && d.supported {
                if first_avail == -1 {
                    first_avail = d.id;
                }
                if d.device_type == "GPU" {
                    return d.id;
                }
            }
        }
        first_avail
    }

    fn opencl_device_info(
        &mut self,
        platform: i32,
        device: i32,
    ) -> Result<Option<(i32, OclDevice)>> {
        // JS: platform -1 resolves to the default platform first
        let platform = if platform == -1 {
            self.opencl_default_platform()
        } else {
            platform
        };
        if platform < 0 {
            return Ok(None);
        }
        if !parpar_ocl::available() {
            return Ok(None);
        }
        let plats = parpar_ocl::enumerate();
        let Some(plat) = plats.get(platform as usize) else {
            return Ok(None);
        };
        // device -1 resolves to the platform's default device
        let device = if device == -1 {
            self.opencl_default_device(platform)
        } else {
            device
        };
        let Some(d) = plat.devices.iter().find(|d| d.id == device) else {
            return Ok(None);
        };
        Ok(Some((platform, d.clone())))
    }

    fn new_input_hasher(
        &mut self,
        slice_size: u64,
        block_out: Box<[u8]>,
    ) -> Result<Self::InputHasher> {
        Ok(parpar_hasher::NativeHasherInput::new(slice_size, block_out))
    }

    fn new_output_hasher(&mut self, regions: u32) -> Result<Self::OutputHasher> {
        Ok(parpar_hasher::NativeHasherOutput::new(regions))
    }
}

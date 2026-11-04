//! Shared backend contracts for ParPar compute backends.
//!
//! Both the transitional C++ FFI backend (parpar-ffi) and the eventual
//! pure-Rust backend (parpar-gf16/parpar-hasher) implement these traits.
//! The orchestration layer (parpar) depends only on these contracts -
//! backend crates must never depend on parpar.
//!
//! Buffer ownership mirrors the JS addon: submitted buffers are captured by
//! the backend and returned through the completion event, exactly as Node
//! attached the Buffer to the callback and handed it back in the callback
//! arguments. This preserves zero-copy behaviour without unsafe borrows
//! crossing an async boundary.

use std::fmt;

pub mod composite;
pub use composite::{CanAdd, CompositeProc, EventQueues, RangeBackend};

/// An error equivalent to what the JS addon would raise.
/// `message` carries the JS-equivalent error text.
#[derive(Debug, Clone)]
pub struct BackendError {
    pub message: String,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BackendError {}

pub type Result<T> = std::result::Result<T, BackendError>;

impl BackendError {
    pub fn new(msg: impl Into<String>) -> Self {
        BackendError {
            message: msg.into(),
        }
    }
}

/// Buffer type moved into and out of the backend.
/// `len` tracks the meaningful bytes; the allocation may be larger.
pub struct SliceBuf {
    pub data: Box<[u8]>,
    pub len: usize,
    /// invoked with the storage Vec when the SliceBuf is dropped - lets an
    /// origin buffer pool (e.g. the reader's) reclaim it instead of freeing
    #[doc(hidden)]
    pub reclaim: Option<Box<dyn FnOnce(Vec<u8>) + Send>>,
}

impl SliceBuf {
    pub fn new(data: Box<[u8]>, len: usize) -> Self {
        // hard invariant - as_bytes() would panic anyway; fail at
        // construction so the blame lands on the caller
        assert!(len <= data.len(), "SliceBuf len exceeds allocation");
        SliceBuf {
            data,
            len,
            reclaim: None,
        }
    }
    /// Attach a reclaim callback; when this SliceBuf drops the storage Vec
    /// is handed to `f` instead of the allocator.
    pub fn with_reclaim(mut self, f: impl FnOnce(Vec<u8>) + Send + 'static) -> Self {
        self.reclaim = Some(Box::new(f));
        self
    }
    /// Hand the storage Vec to the reclaim callback immediately (if set),
    /// leaving an empty buffer behind for bookkeeping. Called once the
    /// backend has consumed the bytes (e.g. after kernel prepare) so the
    /// origin pool can reuse the storage without waiting for InputDone.
    pub fn release_storage(&mut self) {
        if let Some(r) = self.reclaim.take() {
            r(std::mem::take(&mut self.data).into_vec());
        }
    }
    /// Consume into the storage Vec, discarding any reclaim callback.
    pub fn into_vec(mut self) -> Vec<u8> {
        let data = std::mem::take(&mut self.data).into_vec();
        drop(self.reclaim.take());
        std::mem::forget(self);
        data
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data[..self.len]
    }
}

impl std::fmt::Debug for SliceBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SliceBuf")
            .field("data", &self.data)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl Drop for SliceBuf {
    fn drop(&mut self) {
        if let Some(r) = self.reclaim.take() {
            r(std::mem::take(&mut self.data).into_vec());
        }
    }
}

/// GF16 multiplication method identifiers (Galois16Methods order).
/// `Auto = 0` resolves to the platform default.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum GfMethod {
    Auto = 0,
    Lookup = 1,
    LookupSse2 = 2,
    Lookup3 = 3,
    ShuffleNeon = 4,
    Shuffle128Sve = 5,
    Shuffle128Sve2 = 6,
    Shuffle2x128Sve2 = 7,
    Shuffle512Sve2 = 8,
    Shuffle128Rvv = 9,
    ShuffleSsse3 = 10,
    ShuffleAvx = 11,
    ShuffleAvx2 = 12,
    ShuffleAvx512 = 13,
    ShuffleVbmi = 14,
    Shuffle2xAvx2 = 15,
    Shuffle2xAvx512 = 16,
    XorSse2 = 17,
    XorJitSse2 = 18,
    XorJitAvx2 = 19,
    XorJitAvx512 = 20,
    AffineGfni = 21,
    AffineAvx2 = 22,
    AffineAvx512 = 23,
    Affine2xGfni = 24,
    Affine2xAvx2 = 25,
    Affine2xAvx512 = 26,
    AffineBmm = 27,
    ClmulNeon = 28,
    ClmulSha3 = 29,
    ClmulSve2 = 30,
    ClmulRvv = 31,
}

impl GfMethod {
    pub fn from_i32(v: i32) -> Option<GfMethod> {
        (0..=31)
            .contains(&v)
            .then(|| unsafe { std::mem::transmute::<i32, GfMethod>(v) })
    }
}

/// OpenCL kernel variants (Galois16OCLMethods order).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum GfOclMethod {
    Auto = 0,
    Normal = 1,
    Log = 2,
    LogSeq = 3,
    LogBeta = 4,
    LogGamma = 5,
}

impl GfOclMethod {
    pub fn from_i32(v: i32) -> Option<GfOclMethod> {
        (0..=5)
            .contains(&v)
            .then(|| unsafe { std::mem::transmute::<i32, GfOclMethod>(v) })
    }
}

/// Input-hasher method identifiers (HasherInputMethods order).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum InHashMethod {
    Scalar = 0,
    Simd = 1,
    Crc = 2,
    SimdCrc = 3,
    Bmi1 = 4,
    Avx512 = 5,
}

/// Output-hasher method identifiers (MD5MultiLevels order).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum OutHashMethod {
    Scalar = 0,
    Sse = 1,
    Avx2 = 2,
    Xop = 3,
    Avx512f = 4,
    Avx512vl = 5,
    Neon = 6,
    Sve2 = 7,
}

/// Static info for a GF16 method (mirrors addon gf_info()).
#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub id: i32,
    pub name: String,
    pub alignment: u64,
    pub stride: u64,
    pub target_chunk: u64,
    pub target_grouping: u32,
}

/// Runtime info for a configured processor (mirrors addon .info()).
#[derive(Debug, Clone)]
pub struct GfCpuInfo {
    pub threads: i32,
    pub method_desc: String,
    pub chunk_size: u64,
    pub staging_count: u32,
    pub staging_size: u64,
    pub alignment: u32,
    pub stride: u64,
    pub slice_mem: u64,
    pub num_output_slices: i32,
}

#[derive(Debug, Clone)]
pub struct GfOclInfo {
    pub device_name: String,
    pub method_desc: String,
    pub staging_count: u32,
    pub staging_size: u64,
    pub chunk_size: u64,
    pub output_chunks: u32,
    pub slice_mem: u64,
    pub num_output_slices: i32,
}

#[derive(Debug, Clone)]
pub struct GfInfo {
    pub cpu: Option<GfCpuInfo>,
    pub ocl: Vec<GfOclInfo>,
}

/// OpenCL device descriptor (mirrors OclDeviceToJS fields).
#[derive(Debug, Clone)]
pub struct OclDevice {
    pub id: i32,
    pub name: String,
    pub vendor_id: i32,
    pub available: bool,
    pub supported: bool,
    pub device_type: String,
    pub memory_global: u64,
    pub cache_global: u64,
    pub memory_unified: bool,
    pub memory_constant: u64,
    /// None when JS would omit the field (unified && local == global).
    pub memory_local: Option<u64>,
    pub memory_max_alloc: u64,
    pub workgroup_limit: u32,
    pub workgroup_multiple: u32,
    pub compute_units: u32,
}

#[derive(Debug, Clone)]
pub struct OclPlatform {
    pub id: i32,
    pub name: String,
    pub devices: Vec<OclDevice>,
}

/// CPU backend allocation for a GfProc.
#[derive(Debug, Clone)]
pub struct CpuSpec {
    pub method: i32,
    pub input_batchsize: u32,
    pub input_minbatchsize: u32,
    pub chunk_size: u32,
    pub slice_offset: u64,
    /// 0 means "the full slice" (mirrors absent JS property).
    pub slice_size: u64,
}

/// OpenCL backend allocation for a GfProc.
#[derive(Debug, Clone)]
pub struct OclSpec {
    pub platform: i32,
    pub device: i32,
    pub slice_size: u64,
    pub slice_offset: u64,
    pub method: i32,
    /// 0 inherits the CPU checksum method.
    pub cksum_method: i32,
    pub input_batchsize: i32,
    pub input_minbatchsize: i32,
    pub target_iters: u32,
    pub target_grouping: u32,
}

/// Events delivered by a processor (replaces JS callbacks).
/// Completed buffers are returned inside the events, mirroring the
/// Buffer argument the JS callbacks received.
#[derive(Debug)]
pub enum ProcEvent {
    /// An input slice was consumed; the submission buffer is returned.
    InputDone {
        token: u64,
        input_index: i64,
        buf: SliceBuf,
    },
    /// end_input processing finished; outputs may now be fetched.
    Ended { token: u64 },
    /// Output fetch completed; the output buffer is returned filled.
    OutputDone {
        token: u64,
        cksum_valid: bool,
        buf: SliceBuf,
    },
    /// Progress: number of inputs currently staged.
    Progress { num_inputs: i64 },
    /// Asynchronous error.
    Error { token: u64, message: String },
}

/// Result of an add_input call.
#[derive(Debug)]
pub enum AddResult {
    /// Input accepted; an InputDone event will arrive for the token.
    Accepted,
    /// Staging areas full; the buffer is returned to the caller,
    /// who should wait for progress and retry.
    Full(SliceBuf),
}

/// One compute backend instance (equivalent to addon GfProc).
pub trait GfProc: Send {
    /// Opaque id identifying this proc in wait_event results.
    fn proc_id(&self) -> usize;

    /// Returns the actual number of threads in use.
    fn set_num_threads(&mut self, threads: i32) -> Result<i32>;
    fn set_current_slice_size(&mut self, slice_size: u64) -> Result<()>;
    /// exponents == None => all exponents default to 1 (C++ setRecoverySlices).
    fn set_recovery_slices(&mut self, exponents: Option<&[u16]>, count: usize) -> Result<()>;
    fn free_mem(&mut self) -> Result<()>;

    /// Submit one input slice. On Accepted the backend owns the buffer until
    /// the InputDone event returns it; on Full it is returned immediately.
    fn add_input(&mut self, buf: SliceBuf, input_index: i32, token: u64) -> Result<AddResult>;
    fn end_input(&mut self, token: u64) -> Result<()>;
    fn get_output(&mut self, output_index: i32, out: SliceBuf, token: u64) -> Result<()>;
    fn set_progress(&mut self, on: bool) -> Result<()>;
    fn info(&mut self) -> Result<GfInfo>;
    /// Deinit: wait for in-flight work, release native resources.
    fn close(&mut self) -> Result<()>;
}

/// Factory + global capability surface for a compute backend.
pub trait ComputeBackend {
    type Proc: GfProc;
    type InputHasher: HasherInput;
    type OutputHasher: HasherOutput;

    fn new_proc(
        &mut self,
        slice_size: u64,
        cpu: Option<&CpuSpec>,
        ocl: &[OclSpec],
        staging_areas: i32,
    ) -> Result<Self::Proc>;

    /// Wait for the next event from any processor owned by this backend.
    /// Returns (proc_id, event). None only if the backend is torn down.
    fn wait_event(&mut self, timeout_ms: i32) -> Option<(usize, ProcEvent)>;

    fn gf_info(method: i32) -> Result<MethodInfo>
    where
        Self: Sized;
    fn set_inhash_method(method: i32) -> bool
    where
        Self: Sized;
    fn set_outhash_method(level: i32)
    where
        Self: Sized;
    fn inhash_method_name() -> &'static str
    where
        Self: Sized;
    fn outhash_method_name() -> &'static str
    where
        Self: Sized;

    /// Enumerate OpenCL platforms. Err when the runtime cannot load.
    fn opencl_platforms(&mut self) -> Result<Vec<OclPlatform>>;
    fn opencl_default_platform(&mut self) -> i32;
    /// `PAR2ProcOCL::defaultDeviceId` - first available GPU on the platform,
    /// else first available device, else -1.
    fn opencl_default_device(&mut self, platform: i32) -> i32;
    /// `opencl_device_info` - resolves -1 defaults; returns (resolved
    /// platform id, device info) or None when invalid (JS returns undefined).
    fn opencl_device_info(
        &mut self,
        platform: i32,
        device: i32,
    ) -> Result<Option<(i32, OclDevice)>>;

    fn new_input_hasher(
        &mut self,
        slice_size: u64,
        block_out: Box<[u8]>,
    ) -> Result<Self::InputHasher>;
    fn new_output_hasher(&mut self, regions: u32) -> Result<Self::OutputHasher>;
}

/// Per-file input hasher: per-slice MD5+CRC32 records + final file MD5.
/// Mirrors addon HasherInput (slice-boundary splitting is internal).
/// block_out was captured at construction and receives 20-byte records;
/// `end` hands it back.
pub trait HasherInput: Send {
    /// Feed data; must be serialised by the caller per hasher.
    fn update(&mut self, buf: &[u8]) -> Result<()>;
    /// Writes the 16-byte file MD5 to `out`, then returns the block_out
    /// buffer (with all per-slice records written). The hasher is finished.
    fn end(&mut self, out: &mut [u8]) -> Result<Box<[u8]>>;
    fn reset(&mut self) -> Result<()>;
    /// Deposit a file MD5 computed outside the hasher (lane-split read
    /// thread). Backends that hash the file lane internally ignore it.
    fn set_file_md5(&mut self, _md5: &[u8; 16]) {}
}

/// Multi-region output hasher (MD5Multi): hashes N equal-length buffers.
/// Mirrors addon HasherOutput.
pub trait HasherOutput: Send {
    /// `bufs.len()` must equal the region count; all bufs must be equal len.
    fn update(&mut self, bufs: &[&[u8]]) -> Result<()>;
    /// Writes regions*16 bytes of MD5s to out.
    fn get(&mut self, out: &mut [u8]) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
}

impl GfProc for Box<dyn GfProc> {
    fn proc_id(&self) -> usize {
        (**self).proc_id()
    }
    fn set_num_threads(&mut self, threads: i32) -> Result<i32> {
        (**self).set_num_threads(threads)
    }
    fn set_current_slice_size(&mut self, slice_size: u64) -> Result<()> {
        (**self).set_current_slice_size(slice_size)
    }
    fn set_recovery_slices(&mut self, exponents: Option<&[u16]>, count: usize) -> Result<()> {
        (**self).set_recovery_slices(exponents, count)
    }
    fn free_mem(&mut self) -> Result<()> {
        (**self).free_mem()
    }
    fn add_input(&mut self, buf: SliceBuf, input_index: i32, token: u64) -> Result<AddResult> {
        (**self).add_input(buf, input_index, token)
    }
    fn end_input(&mut self, token: u64) -> Result<()> {
        (**self).end_input(token)
    }
    fn get_output(&mut self, output_index: i32, out: SliceBuf, token: u64) -> Result<()> {
        (**self).get_output(output_index, out, token)
    }
    fn set_progress(&mut self, on: bool) -> Result<()> {
        (**self).set_progress(on)
    }
    fn info(&mut self) -> Result<GfInfo> {
        (**self).info()
    }
    fn close(&mut self) -> Result<()> {
        (**self).close()
    }
}

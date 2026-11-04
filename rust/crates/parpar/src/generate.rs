//! Port of `lib/par2gen.js` - the solver/layout half of `PAR2Gen`:
//! slice sizing, input/recovery slice calculations, memory limits,
//! chunking/pass layout, device distribution and recovery-volume layout.
//! The reader/writer pipeline that consumes this layout is elsewhere.

use crate::args::{CriticalRedundancy, MethodRef, SpecExpr, SpecRounding, SpecTerm, SpecUnit};
use crate::cliparse::{ThrottleMode, ThrottleSpec};
use crate::compat;
use crate::fileinfo::FileInfo;
use crate::par2::{
    method_num, method_num_ref, CharOpt, GfCtx, GfOpts, Par2, Par2File, ProcOpts, GFOCL_METHODS,
    GF_METHODS,
};
use crate::sysinfo;
use parpar_backend::{CpuSpec, OclSpec};

pub type JsResult<T> = Result<T, String>;

/// `friendlySize` from par2gen.js - used in solver error messages.
pub fn friendly_size(mut s: f64) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut i = 0;
    while i < UNITS.len() {
        if s < 10000.0 {
            break;
        }
        s /= 1024.0;
        i += 1;
    }
    // JS indexes units[i] past the array end for huge values -> 'undefined'
    let unit = if i < UNITS.len() {
        UNITS[i]
    } else {
        "undefined"
    };
    format!("{} {}", compat::js_math_round(s * 100.0) / 100.0, unit)
}

/// JS truthiness for numbers (0 and NaN are falsy).
fn truthy(v: f64) -> bool {
    v != 0.0 && !v.is_nan()
}

fn fmt_num(v: f64) -> String {
    compat::fmt_num(v)
}

// ---------------------------------------------------------------------------
// slice-count helpers (par2gen.js lines 77-189)
// ---------------------------------------------------------------------------

/// `calcNumSlicesForFiles`.
pub fn calc_num_slices_for_files(files: &[FileInfo], slice_size: f64) -> f64 {
    files
        .iter()
        .map(|f| (f.size as f64 / slice_size).ceil())
        .sum()
}

/// `calcSliceSizeForFiles` - binary search, returns (lbound, ubound).
pub fn calc_slice_size_for_files(num_slices: f64, files: &[FileInfo], mult: f64) -> (f64, f64) {
    let mut lbound = mult;
    let mut ubound = files.iter().fold(0f64, |max, f| max.max(f.size as f64));
    ubound = (ubound / mult).ceil() * mult;
    while lbound < ubound - mult {
        let mid = ((ubound + lbound) / (mult * 2.0)).floor() * mult;
        if num_slices >= calc_num_slices_for_files(files, mid) {
            ubound = mid;
        } else {
            lbound = mid;
        }
    }
    (lbound, ubound)
}

/// `_calcNumSlicesCommon` - file_values are file sizes or per-file slice
/// counts (numbers); `None` when the caller passed no files.
fn calc_num_slices_common(
    s: &SpecTerm,
    base_value: f64,
    file_values: Option<Vec<f64>>,
) -> JsResult<f64> {
    match s.unit {
        SpecUnit::Power | SpecUnit::Log | SpecUnit::ILog => {
            if base_value == 0.0 {
                return Ok(0.0);
            }
            match s.unit {
                SpecUnit::Power => Ok(base_value.powf(s.value)),
                SpecUnit::Log => {
                    let val = base_value.ln() / s.value.ln();
                    if !val.is_finite() {
                        // RangeError in JS
                        return Err("Invalid logarithm value".into());
                    }
                    Ok(val)
                }
                _ => {
                    let base_value = base_value * s.value;
                    Ok(base_value
                        / compat::js_math_max(base_value.ln() / std::f64::consts::LN_2, 1.0))
                }
            }
        }
        SpecUnit::LargestFiles | SpecUnit::SmallestFiles if file_values.is_some() => {
            let mut sorted = file_values.unwrap();
            if s.unit == SpecUnit::LargestFiles {
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            // JS tries to drop zero-size entries for 'smallest_files' via
            // `sorted[i].size == 0`, but sorted holds numbers (no `.size`)
            // so the loop never advances - dead code, nothing is removed.
            let abs_value = s.value.abs();
            let amt = abs_value.ceil();
            let mut selected: Vec<f64> = sorted.into_iter().take(amt.max(0.0) as usize).collect();
            if amt != abs_value && amt <= selected.len() as f64 {
                let idx = amt as usize - 1;
                selected[idx] *= abs_value - abs_value.floor();
            }
            Ok(selected.iter().sum::<f64>() * if s.value < 0.0 { -1.0 } else { 1.0 })
        }
        _ => Ok(s.value),
    }
}

fn spec_scale(s: &SpecTerm) -> f64 {
    if truthy(s.scale) {
        s.scale
    } else {
        1.0
    }
}

/// JS `if(spec.method)` - a method name/number is truthy unless '' or 0.
fn method_truthy(m: &MethodRef) -> bool {
    match m {
        MethodRef::Name(s) => !s.is_empty(),
        MethodRef::Index(n) => truthy(*n),
    }
}

/// `calcNumInputSlices` - returns the byte size, or a *negative* number
/// encoding an exact slice count.
pub fn calc_num_input_slices(
    spec: &SpecTerm,
    files: &[FileInfo],
    total_size: f64,
    multiple: f64,
) -> JsResult<f64> {
    if spec.value < 0.0 {
        // RangeError in JS
        return Err("Input slice value cannot be negative".into());
    }
    let scale = spec_scale(spec);
    if spec.unit == SpecUnit::Count {
        return Ok(compat::js_math_round(scale * -spec.value));
    }
    if spec.unit == SpecUnit::Bytes && scale == 1.0 {
        return Ok(scale * spec.value);
    }
    let sizes: Vec<f64> = files.iter().map(|f| f.size as f64).collect();
    let mut amount = scale * calc_num_slices_common(spec, total_size, Some(sizes))?;
    if amount < 1.0 {
        amount = 1.0;
    }
    if amount % multiple != 0.0 {
        amount = (amount / multiple).ceil() * multiple;
    }
    Ok(amount)
}

/// `calcNumRecoverySlices` - sums amounts across spec terms. `files`
/// decides whether file-values are available (`None` for the critical
/// redundancy calls, which pass only 3 args).
pub fn calc_num_recovery_slices(
    spec: &SpecExpr,
    slice_size: f64,
    in_slices: f64,
    files: Option<&[FileInfo]>,
) -> JsResult<f64> {
    spec.iter()
        .map(|s| {
            let scale = spec_scale(s);
            match s.unit {
                SpecUnit::Ratio => Ok(scale * in_slices * s.value),
                SpecUnit::Bytes => Ok(scale * (s.value / slice_size)),
                _ => {
                    let counts = files.map(|fs| {
                        fs.iter()
                            .map(|f| (f.size as f64 / slice_size).ceil())
                            .collect()
                    });
                    Ok(scale * calc_num_slices_common(s, in_slices, counts)?)
                }
            }
        })
        .try_fold(0f64, |sum, a: JsResult<f64>| a.map(|v| sum + v))
}

// ---------------------------------------------------------------------------
// throttle validation (createThrottle - object creation deferred to readers)
// ---------------------------------------------------------------------------

/// `createThrottle` - performs the JS validation; returns the throttle
/// parameters as (mode, size_or_count, time_ms) for later construction.
pub fn create_throttle(spec: &ThrottleSpec) -> JsResult<(ThrottleMode, i32, i32)> {
    let throttle_time0 = compat::to_int32(spec.time);
    match spec.mode {
        ThrottleMode::Delay => {
            if throttle_time0 <= 0 {
                return Err(format!("Invalid throttle delay ({})", throttle_time0));
            }
            Ok((spec.mode, 0, throttle_time0))
        }
        ThrottleMode::Rate => {
            let throttle_time = if throttle_time0 == 0 {
                1000
            } else {
                throttle_time0
            };
            if throttle_time <= 0 {
                return Err(format!("Invalid throttle time ({})", throttle_time));
            }
            let throttle_size = compat::to_int32(spec.size);
            if throttle_size <= 0 {
                return Err(format!("Invalid throttle size ({})", throttle_size));
            }
            Ok((spec.mode, throttle_size, throttle_time))
        }
        ThrottleMode::ReqRate => {
            let throttle_time = if throttle_time0 == 0 {
                1000
            } else {
                throttle_time0
            };
            if throttle_time <= 0 {
                return Err(format!("Invalid throttle time ({})", throttle_time));
            }
            let count_f = compat::js_number(&spec.count);
            let throttle_count = compat::to_int32(count_f);
            if throttle_count <= 0 || (throttle_count as f64) != count_f {
                return Err(format!("Invalid throttle count ({})", throttle_count));
            }
            Ok((spec.mode, throttle_count, throttle_time))
        }
    }
}

// ---------------------------------------------------------------------------
// par2Ext volume naming
// ---------------------------------------------------------------------------

/// `par2Ext` - the `.volNNNNN+NNNNN.par2` / `.volNNNNN-NNNNN.par2` suffix.
pub fn par2_ext(
    num_slices: i64,
    slice_offset: i64,
    total_slices: i64,
    alt_scheme: bool,
) -> JsResult<String> {
    if num_slices == 0 {
        return Ok(".par2".into());
    }
    let slice_end = slice_offset + num_slices;
    let s_offs = slice_offset.to_string();
    let s_end = (if alt_scheme { num_slices } else { slice_end }).to_string();
    let digits = if total_slices != 0 {
        if slice_end > total_slices {
            return Err("Invalid slice values".into());
        }
        total_slices.to_string().len().max(2)
    } else {
        s_end.len().max(2)
    };
    Ok(format!(
        ".vol{:0>w$}{}{:0>w$}.par2",
        s_offs,
        if alt_scheme { '+' } else { '-' },
        s_end,
        w = digits
    ))
}

// ---------------------------------------------------------------------------
// option + packet types
// ---------------------------------------------------------------------------

/// One `o.openclDevices` entry - JS defaults applied:
/// `{platform: null, device: null, ratio: null, memoryLimit: null, method:
///   null, input_batchsize: 0, target_iters: 0, target_grouping: 0,
///   minChunkSize: 32768}`.
#[derive(Debug, Clone)]
pub struct OclDevOpt {
    pub platform: i32,
    pub device: i32,
    /// None = JS null (unset); resolved during distribution.
    pub ratio: Option<f64>,
    /// None = JS null (auto); Some(0) = no limit.
    pub memory_limit: Option<f64>,
    /// GFOCL method (name or index), resolved for the backend spec when
    /// JS-truthy.
    pub method: Option<MethodRef>,
    pub input_batchsize: i32,
    pub target_iters: i32,
    pub target_grouping: i32,
    pub min_chunk_size: f64,
    // solver outputs
    pub slice_offset: u64,
    pub slice_size: u64,
    /// `cksum_method = gfInfo.id` - resolved GF method index.
    pub cksum_method: i32,
}

impl Default for OclDevOpt {
    fn default() -> Self {
        OclDevOpt {
            platform: -1,
            device: -1,
            ratio: None,
            memory_limit: None,
            method: None,
            input_batchsize: 0,
            target_iters: 0,
            target_grouping: 0,
            min_chunk_size: 32768.0,
            slice_offset: 0,
            slice_size: 0,
            cksum_method: 0,
        }
    }
}

/// `criticalRedundancyScheme` - a spec expression, 'pow2' (back-compat =>
/// `{log:2}`) or the string 'none'.
#[derive(Debug, Clone, PartialEq)]
pub enum CritScheme {
    Spec(SpecExpr),
    None,
}

impl CritScheme {
    pub fn from_cli(v: &CriticalRedundancy) -> CritScheme {
        match v {
            CriticalRedundancy::Named(s) if s == "none" => CritScheme::None,
            CriticalRedundancy::Named(s) if s == "pow2" => CritScheme::Spec(vec![SpecTerm {
                unit: SpecUnit::Log,
                value: 2.0,
                scale: 1.0,
            }]),
            // other strings parse as spec expressions via CLI parsing
            CriticalRedundancy::Named(s) => CritScheme::Spec(vec![SpecTerm {
                unit: SpecUnit::Count,
                value: compat::js_number(s),
                scale: 1.0,
            }]),
            CriticalRedundancy::Expr(e) => CritScheme::Spec(e.clone()),
        }
    }
}

/// The `o` options object after `_extend` - JS defaults in Default impl.
/// Resolved values that JS writes back into `o` are `*_out` fields.
#[derive(Debug, Clone)]
pub struct GenOpts {
    pub output_base: String,                     // ''
    pub min_slice_size: Option<SpecTerm>,        // null
    pub max_slice_size: Option<SpecTerm>,        // null
    pub slice_size_multiple: f64,                // 4
    pub recovery_slices: SpecExpr,               // {slices:0}
    pub min_recovery_slices: Option<SpecExpr>,   // null
    pub max_recovery_slices: Option<SpecExpr>,   // {slices:65536}
    pub recovery_offset: i32,                    // 0
    pub recovery_exponents: Option<Vec<String>>, // null
    /// Validated/converted exponents (`exp|0` results).
    pub recovery_exponents_int: Option<Vec<i32>>,
    pub memory_limit: Option<f64>,       // null=>auto, 0=none
    pub min_chunk_size: f64,             // 128*1024
    pub process_batch_size: Option<f64>, // null
    pub hash_batch_size: f64,            // 8
    pub rec_data_size: Option<f64>,      // null
    pub comments: Vec<String>,
    pub creator: String,
    pub unicode: Option<bool>, // null=auto
    pub output_overwrite: bool,
    pub output_sync: bool,
    pub output_index: bool,                                      // true
    pub output_size_scheme: String,                              // 'pow2'
    pub output_first_file_slices: Option<SpecExpr>,              // null
    pub output_first_file_slices_rounding: Option<SpecRounding>, // 'round'
    pub output_file_max_slices: SpecExpr,                        // {slices:65535}
    pub output_file_max_slices_rounding: Option<SpecRounding>,   // 'round'
    pub output_file_count: f64,                                  // 0
    pub critical_redundancy_scheme: CritScheme,
    pub min_critical_redundancy: f64,              // 1
    pub max_critical_redundancy: f64,              // 0
    pub output_alt_naming_scheme: bool,            // true
    pub display_name_format: String,               // 'common'
    pub display_name_base: String,                 // '.'
    pub seq_read_size: f64,                        // 4MB
    pub chunk_read_threads: f64,                   // 2
    pub seq_read_throttle: Option<ThrottleSpec>,   // null
    pub chunk_read_throttle: Option<ThrottleSpec>, // null
    pub read_buffers: f64,                         // 8
    pub read_hash_queue: f64,                      // 5
    pub num_threads: Option<i32>,                  // null
    pub gf_method: Option<MethodRef>,              // null => auto
    pub loop_tile_size: f64,                       // 0
    pub opencl_devices: Vec<OclDevOpt>,
    pub cpu_min_chunk_size: f64, // 65536
    // values the JS solver writes back into `o`
    pub slice_size: f64,                           // o.sliceSize
    pub recovery_slices_res: f64,                  // o.recoverySlices
    pub output_first_file_slices_res: Option<f64>, // o.outputFirstFileSlices
    pub output_file_max_slices_res: f64,           // o.outputFileMaxSlices
    pub output_file_count_res: i32,                // o.outputFileCount |0
}

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts {
            output_base: String::new(),
            min_slice_size: None,
            max_slice_size: None,
            slice_size_multiple: 4.0,
            recovery_slices: vec![SpecTerm {
                unit: SpecUnit::Count,
                value: 0.0,
                scale: 1.0,
            }],
            min_recovery_slices: None,
            max_recovery_slices: Some(vec![SpecTerm {
                unit: SpecUnit::Count,
                value: 65536.0,
                scale: 1.0,
            }]),
            recovery_offset: 0,
            recovery_exponents: None,
            recovery_exponents_int: None,
            memory_limit: None,
            min_chunk_size: 128.0 * 1024.0,
            process_batch_size: None,
            hash_batch_size: 8.0,
            rec_data_size: None,
            comments: Vec::new(),
            creator: format!(
                "ParPar (library) v{} [https://github.com/animetosho/parpar]",
                env!("CARGO_PKG_VERSION")
            ),
            unicode: None,
            output_overwrite: false,
            output_sync: false,
            output_index: true,
            output_size_scheme: "pow2".into(),
            output_first_file_slices: None,
            output_first_file_slices_rounding: None,
            output_file_max_slices: vec![SpecTerm {
                unit: SpecUnit::Count,
                value: 65535.0,
                scale: 1.0,
            }],
            output_file_max_slices_rounding: None,
            output_file_count: 0.0,
            critical_redundancy_scheme: CritScheme::Spec(vec![SpecTerm {
                unit: SpecUnit::Log,
                value: 2.0,
                scale: 1.0,
            }]),
            min_critical_redundancy: 1.0,
            max_critical_redundancy: 0.0,
            output_alt_naming_scheme: true,
            display_name_format: "common".into(),
            display_name_base: ".".into(),
            seq_read_size: 4.0 * 1048576.0,
            chunk_read_threads: 2.0,
            seq_read_throttle: None,
            chunk_read_throttle: None,
            read_buffers: 8.0,
            read_hash_queue: 5.0,
            num_threads: None,
            gf_method: None,
            loop_tile_size: 0.0,
            opencl_devices: Vec::new(),
            cpu_min_chunk_size: 65536.0,
            slice_size: 0.0,
            recovery_slices_res: 0.0,
            output_first_file_slices_res: None,
            output_file_max_slices_res: 0.0,
            output_file_count_res: 0,
        }
    }
}

/// `Math[rounding]` - 'round' (JS default) / 'floor' / 'ceil'.
fn apply_rounding(rounding: Option<SpecRounding>, v: f64) -> f64 {
    match rounding {
        Some(SpecRounding::Floor) => v.floor(),
        Some(SpecRounding::Ceil) => v.ceil(),
        None => compat::js_math_round(v),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PacketType {
    FileDesc,
    FileChk,
    Comment,
    Main,
    Creator,
    Recovery,
}

/// `PAR2GenPacket` - `data` holds the packet's data buffers once the
/// writer phase fills them (JS `data` = Buffer[] | null).
#[derive(Debug, Clone)]
pub struct GenPacket {
    pub pkt_type: PacketType,
    pub size: u64,
    /// Recovery exponent or file index; -1 for main/creator (JS null).
    pub index: i64,
    /// JS `dataChunkOffset` - offset of `data` within the packet body.
    pub data_chunk_offset: u64,
    /// JS `data` - attached buffers; empty = JS null.
    pub data: Vec<Vec<u8>>,
}

impl GenPacket {
    /// JS `setData(data, offset)`.
    pub fn set_data(&mut self, data: Vec<Vec<u8>>, offset: u64) {
        self.data = data;
        self.data_chunk_offset = offset;
    }
    /// JS `takeData()`.
    pub fn take_data(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.data)
    }
    /// JS `dataLen()`.
    pub fn data_len(&self) -> usize {
        self.data.iter().map(|d| d.len()).sum()
    }
}

/// `PAR2OutFile` - layout only (fd machinery is a later phase).
#[derive(Debug, Clone)]
pub struct RecoveryFile {
    pub name: String,
    pub recovery_slices: i64,
    pub recovery_index: i64,
    pub packets: Vec<GenPacket>,
    pub total_size: u64,
}

/// Resolved OpenCL device info - the fields par2gen reads:
/// `{platform_id, id, memory_global}`.
#[derive(Debug, Clone)]
pub struct OclDevInfo {
    pub platform_id: i32,
    pub id: i32,
    pub memory_global: u64,
}

/// `Par2.gf_info` + `Par2.opencl_device_info` - backend lookups the solver
/// needs, abstracted for testing.
pub trait SolverInfo {
    fn gf_info(&mut self, method: i32) -> JsResult<parpar_backend::MethodInfo>;
    fn ocl_device_info(&mut self, platform: i32, device: i32) -> Option<OclDevInfo>;
}

impl<B: parpar_backend::ComputeBackend> SolverInfo for crate::par2::SharedBackend<B> {
    fn gf_info(&mut self, method: i32) -> JsResult<parpar_backend::MethodInfo> {
        // when the native proc is active, `auto` resolves through the
        // native kernel table so proc_cpu.method names what it will run
        let method = if method == 0 && self.native_proc() {
            parpar_gf16::kernel::auto_id()
        } else {
            method
        };
        B::gf_info(method).map_err(|e| e.message)
    }
    fn ocl_device_info(&mut self, platform: i32, device: i32) -> Option<OclDevInfo> {
        self.with_backend(|b| {
            b.opencl_device_info(platform, device)
                .ok()?
                .map(|(platform_id, d)| OclDevInfo {
                    platform_id,
                    id: d.id,
                    memory_global: d.memory_global,
                })
        })
    }
}

/// `os`/`process`/`buffer` facts used by the solver. `constrained_mem`
/// carries the raw `process.constrainedMemory()` value - including the
/// (uint64_t)-1 sentinel, which JS treats as truthy => constrained.
#[derive(Debug, Clone)]
pub struct SysEnv {
    pub total_mem: u64,
    pub free_mem: u64,
    pub constrained_mem: u64,
    pub is_64b: bool,
    /// `buffer.kMaxLength` - MAX_BUFFER_SIZE subtracts 1024+68.
    pub k_max_length: u64,
}

impl SysEnv {
    pub fn max_buffer_size(&self) -> u64 {
        self.k_max_length.saturating_sub(1024 + 68)
    }

    /// Values for the current machine (Node-compatible definitions).
    pub fn detect() -> SysEnv {
        SysEnv {
            total_mem: sysinfo::total_mem(),
            free_mem: sysinfo::free_mem(),
            constrained_mem: sysinfo::constrained_memory(),
            is_64b: sysinfo::is_64b_platform(),
            k_max_length: sysinfo::K_MAX_LENGTH,
        }
    }
}

// ---------------------------------------------------------------------------
// Par2Gen
// ---------------------------------------------------------------------------

/// The solver + layout result. `gen.par2.files` is the sorted file list
/// (JS `this.files = par.getFiles()`).
pub struct Par2Gen<C: GfCtx> {
    pub opts: GenOpts,
    pub par2: Par2<C>,
    pub total_size: f64,
    pub input_slices: f64,
    pub recovery_files: Vec<RecoveryFile>,
    pub crit_packets: Vec<GenPacket>,
    pub creator_pkt: GenPacket,
    pub passes: f64,
    /// `chunks` - advisory chunk passes.
    pub chunks: f64,
    pub slices_per_pass: f64,
    /// `_chunkSize`.
    pub chunk_size: f64,
    /// `_sliceNums` - JS `Array(recoverySlices)` with per-packet exponents
    /// filled in order; `None` = JS undefined hole.
    pub slice_nums: Vec<Option<i64>>,
    pub read_size: f64,
    /// `procInStagingBufferCount`.
    pub proc_in_staging_buffer_count: f64,
    /// The `stagingCount` local (passed to the PAR2 proc).
    pub staging_count: i32,
    /// `cpuRatio` - CPU's share after OpenCL distribution.
    pub cpu_ratio: f64,
    /// `minMemoryLimit` - tightest device memory limit (0 = none).
    pub min_memory_limit: f64,
    /// Resolved `o.openclDevices` (post distribution).
    pub opencl_devices: Vec<OclDevOpt>,
    /// `procCpu` spec (None when OpenCL covers the whole chunk).
    pub proc_cpu: Option<CpuSpec>,
    /// `proc_ocl` - backend specs for the devices kept.
    pub proc_ocl: Vec<OclSpec>,
    /// `processBatchSize` after the JS rescale (used for dump parity).
    pub process_batch_size: f64,
    /// `recDataSize` after clamping (memory-accounting value).
    pub rec_data_size: f64,
}

impl<C: GfCtx> Par2Gen<C> {
    /// `new PAR2Gen(fileInfo, inputSliceDef, ppo)` - `slice_size_def` is the
    /// parseSizeOrNum-style spec; `host` resolves `gf_info` and OpenCL
    /// device info; `sys` carries the machine facts JS reads live.
    pub fn new(
        files: &mut [FileInfo],
        slice_size_def: &SpecTerm,
        mut o: GenOpts,
        ctx: C,
        ascii_charset: &str,
        host: &mut dyn SolverInfo,
        sys: &SysEnv,
    ) -> JsResult<Par2Gen<C>> {
        if let Some(exps_raw) = o.recovery_exponents.clone() {
            o.output_index = false;
            o.output_size_scheme = "equal".into();
            o.output_first_file_slices = None;
            o.output_file_max_slices = vec![SpecTerm {
                unit: SpecUnit::Count,
                value: 65535.0,
                scale: 1.0,
            }];
            o.output_file_count = 0.0;
            o.recovery_offset = 0;

            let mut used = std::collections::HashSet::new();
            let mut exps = Vec::with_capacity(exps_raw.len());
            for e in &exps_raw {
                // JS `exp|0` on the string form (e.g. "5" -> 5, "x" -> 0)
                let exp = compat::to_int32(compat::js_number(e));
                if !(0..=65534).contains(&exp) {
                    return Err(format!("Invalid recovery exponent {}", exp));
                }
                if !used.insert(exp) {
                    return Err(format!("Duplicate recovery exponent {}", exp));
                }
                exps.push(exp);
            }
            o.recovery_exponents_int = Some(exps.clone());
            // JS: o.recoverySlices = o.recoveryExponents.length (a number)
            o.recovery_slices_res = exps.len() as f64;
        }

        // JS only guards `!fileInfo || typeof fileInfo != 'object'` - an
        // empty array proceeds (inputSlices=0 -> downstream errors).
        let mut total_size = 0f64;
        let mut data_files = 0f64;
        for f in files.iter() {
            if f.size == 0 {
                continue;
            }
            total_size += f.size as f64;
            data_files += 1.0;
        }

        if data_files > 32768.0 {
            return Err(format!(
                "Cannot have more than 32768 non-empty files in a single PAR2 recovery set ({} non-empty files given)",
                fmt_num(data_files)
            ));
        }
        if o.slice_size_multiple % 4.0 != 0.0 || o.slice_size_multiple == 0.0 {
            return Err(format!(
                "Slice size multiple ({}) must be a multiple of 4",
                fmt_num(o.slice_size_multiple)
            ));
        }

        // ---- complex sliceSize definitions (lines 322-465) ----
        let ss_unit = slice_size_def.unit;
        let mut slice_size =
            calc_num_input_slices(slice_size_def, files, total_size, o.slice_size_multiple)?;
        if slice_size >= 0.0 && slice_size % o.slice_size_multiple != 0.0 {
            return Err(format!(
                "Slice size ({}) is not a multiple of {}",
                fmt_num(slice_size),
                fmt_num(o.slice_size_multiple)
            ));
        }

        let mut adj_ssmin = false;
        let mut adj_ssmax = false;
        // resolved min/max (raw numbers post calcNumInputSlices)
        let mut min_res = 0.0f64;
        let mut max_res = 0.0f64;

        if let Some(m) = o.min_slice_size.clone() {
            let same_unit_min = ss_unit == m.unit;
            min_res = calc_num_input_slices(&m, files, total_size, o.slice_size_multiple)?;
            if (slice_size < 0.0) == (min_res <= 0.0) && slice_size.abs() < min_res.abs() {
                if same_unit_min {
                    return Err("Specified slice size/count is below specified minimum".into());
                }
                adj_ssmin = true;
            }
            if min_res > 0.0 && min_res % o.slice_size_multiple != 0.0 {
                return Err(format!(
                    "Specified min slice size ({}) is not a multiple of specified slice size multiple ({})",
                    fmt_num(min_res),
                    fmt_num(o.slice_size_multiple)
                ));
            }
        }
        if let Some(m) = o.max_slice_size.clone() {
            let same_unit_max = ss_unit == m.unit;
            max_res = calc_num_input_slices(&m, files, total_size, o.slice_size_multiple)?;
            if (slice_size < 0.0) == (max_res <= 0.0) && slice_size.abs() > max_res.abs() {
                if same_unit_max {
                    return Err("Specified slice size/count is above specified maximum".into());
                }
                adj_ssmax = true;
            }
            if max_res > 0.0 && max_res % o.slice_size_multiple != 0.0 {
                return Err(format!(
                    "Specified max slice size ({}) is not a multiple of specified slice size multiple ({})",
                    fmt_num(max_res),
                    fmt_num(o.slice_size_multiple)
                ));
            }
        }
        if o.min_slice_size.is_some() && o.max_slice_size.is_some() {
            if (min_res <= 0.0) == (max_res <= 0.0) && max_res.abs() < min_res.abs() {
                if min_res <= 0.0 {
                    return Err(format!(
                        "Specified min/max slice count range is invalid or cannot be satisfied: {} (min) > {} (max)",
                        fmt_num(-min_res),
                        fmt_num(-max_res)
                    ));
                }
                return Err(format!(
                    "Specified min/max slice size range is invalid or cannot be satisfied: {} (min) > {} (max)",
                    friendly_size(min_res),
                    friendly_size(max_res)
                ));
            }
        }
        if o.min_slice_size.is_none() {
            min_res = slice_size;
        } else if adj_ssmin {
            slice_size = min_res;
        }
        if o.max_slice_size.is_none() {
            max_res = slice_size;
        } else if adj_ssmax {
            slice_size = max_res;
        }

        if slice_size < 0.0 {
            // specifies number of blocks to make
            if -slice_size < data_files {
                return Err(format!(
                    "Cannot select slice size to satisfy required number of slices, as there are more non-empty files ({}) than the specified number of slices ({})",
                    fmt_num(data_files),
                    fmt_num(-slice_size)
                ));
            }
            if data_files == 0.0 {
                return Err("Cannot have any input slices when there is no input data".into());
            }

            let bounds = calc_slice_size_for_files(-slice_size, files, o.slice_size_multiple);
            let lbound_slices = calc_num_slices_for_files(files, bounds.0);
            let ubound_slices = calc_num_slices_for_files(files, bounds.1);
            if lbound_slices == -slice_size {
                o.slice_size = bounds.0;
            } else if ubound_slices == -slice_size {
                o.slice_size = bounds.1;
            } else {
                // target count unreachable - auto-scale may still fix it
                let use_lbound = ((min_res <= 0.0 && lbound_slices >= -min_res)
                    || (min_res > 0.0 && bounds.0 >= min_res))
                    && ((max_res <= 0.0 && lbound_slices <= -max_res)
                        || (max_res > 0.0 && bounds.0 <= max_res));
                let use_ubound = ((min_res <= 0.0 && ubound_slices >= -min_res)
                    || (min_res > 0.0 && bounds.1 >= min_res))
                    && ((max_res <= 0.0 && ubound_slices <= -max_res)
                        || (max_res > 0.0 && bounds.1 <= max_res));
                // JS picks lbound when either bound works
                if use_lbound || use_ubound {
                    o.slice_size = if use_lbound { bounds.0 } else { bounds.1 };
                } else {
                    return Err(format!(
                        "Cannot determine a slice size to satisfy required number of slices ({}): using a slice size of {} bytes would produce {} slice(s), whilst a size of {} bytes would produce {} slice(s)",
                        fmt_num(-slice_size),
                        fmt_num(bounds.0),
                        fmt_num(lbound_slices),
                        fmt_num(bounds.1),
                        fmt_num(ubound_slices)
                    ));
                }
            }

            // auto-scale size
            if min_res > 0.0 && o.slice_size < min_res {
                o.slice_size = min_res;
            }
            if max_res > 0.0 && o.slice_size > max_res {
                o.slice_size = max_res;
            }
        } else {
            o.slice_size = slice_size;

            // min/max *sizes* ignored here; min/max *counts* auto-scale
            let mut num_slices = calc_num_slices_for_files(files, slice_size);
            if min_res <= 0.0 && num_slices < -min_res {
                let bounds = calc_slice_size_for_files(-min_res, files, o.slice_size_multiple);
                let lbound_slices = calc_num_slices_for_files(files, bounds.0);
                if lbound_slices >= -min_res {
                    o.slice_size = bounds.0;
                    num_slices = lbound_slices;
                } else {
                    o.slice_size = bounds.1;
                    num_slices = calc_num_slices_for_files(files, bounds.1);
                }
            }
            if max_res <= 0.0 && num_slices > -max_res {
                let bounds = calc_slice_size_for_files(-max_res, files, o.slice_size_multiple);
                let ubound_slices = calc_num_slices_for_files(files, bounds.1);
                if ubound_slices <= -max_res {
                    o.slice_size = bounds.1;
                    num_slices = ubound_slices;
                } else {
                    o.slice_size = bounds.0;
                    num_slices = calc_num_slices_for_files(files, bounds.0);
                }
            }
            if min_res <= 0.0 && num_slices < -min_res {
                return Err(
                    "Could not find an appropriate slice size based on supplied constraints".into(),
                );
            }
        }
        if o.slice_size < 1.0 {
            return Err("Invalid slice size specified".into());
        }
        let mut input_slices = calc_num_slices_for_files(files, o.slice_size);
        if min_res > 0.0 && min_res < o.slice_size {
            // shrink the slice size whilst retaining the slice count
            let bounds = calc_slice_size_for_files(input_slices, files, o.slice_size_multiple);
            let lbound_slices = calc_num_slices_for_files(files, bounds.0);
            let ubound_slices = calc_num_slices_for_files(files, bounds.1);
            if lbound_slices == input_slices {
                o.slice_size = bounds.0.max(min_res.max(0.0));
            } else if ubound_slices == input_slices {
                o.slice_size = bounds.1.max(min_res.max(0.0));
            }
            // else: abandon the optimization (JS comment says can't happen)
        }

        if input_slices > 32768.0 {
            return Err(format!(
                "Too many input slices: {} exceeds maximum of 32768. Please consider increasing the slice size, reducing the amount of input data, or using the `--auto-slice-size` option",
                fmt_num(input_slices)
            ));
        }
        input_slices = calc_num_slices_for_files(files, o.slice_size);
        if (min_res <= 0.0 && input_slices < -min_res)
            || (min_res > 0.0 && o.slice_size < min_res)
            || (max_res <= 0.0 && input_slices > -max_res)
            || (max_res > 0.0 && o.slice_size > max_res)
        {
            return Err("Could not satisfy specified min/max slice size/count constraints".into());
        }

        let max_buffer_size_mod2 = ((sys.max_buffer_size() as f64) / 2.0).floor() * 2.0;
        if o.min_chunk_size > max_buffer_size_mod2 {
            return Err(format!(
                "Minimum chunk size ({}) exceeds maximum size supported by this version of Node.js of {} bytes",
                fmt_num(o.min_chunk_size),
                fmt_num(max_buffer_size_mod2)
            ));
        }
        if truthy(o.min_chunk_size) && o.min_chunk_size % 2.0 != 0.0 {
            return Err(format!(
                "Minimum chunk size ({}) must be a multiple of 2 bytes",
                fmt_num(o.min_chunk_size)
            ));
        }
        if o.seq_read_size > max_buffer_size_mod2 {
            return Err(format!(
                "Read buffer size ({}) exceeds maximum size supported by this version of Node.js of {} bytes",
                fmt_num(o.seq_read_size),
                fmt_num(max_buffer_size_mod2)
            ));
        }

        // ---- recovery slice count (lines 483-499) ----
        if o.recovery_exponents_int.is_none() {
            let mut recovery_slices = calc_num_recovery_slices(
                &o.recovery_slices,
                o.slice_size,
                input_slices,
                Some(files),
            )?;
            let mut min_rec = recovery_slices.ceil();
            let mut max_rec = recovery_slices.floor();
            if let Some(m) = &o.min_recovery_slices {
                min_rec =
                    calc_num_recovery_slices(m, o.slice_size, input_slices, Some(files))?.ceil();
            }
            if let Some(m) = &o.max_recovery_slices {
                max_rec =
                    calc_num_recovery_slices(m, o.slice_size, input_slices, Some(files))?.floor();
            }
            recovery_slices =
                compat::js_math_min(compat::js_math_max(recovery_slices, min_rec), max_rec);
            recovery_slices = compat::js_math_round(recovery_slices);
            if recovery_slices < min_rec || recovery_slices > max_rec {
                return Err(
                    "Could not satisfy specified min/max recovery slice count constraints".into(),
                );
            }
            if recovery_slices < 0.0 || !recovery_slices.is_finite() {
                return Err("Invalid number of recovery slices".into());
            }
            if recovery_slices + o.recovery_offset as f64 > 65535.0 {
                return Err(format!(
                    "Cannot generate specified number of recovery slices: {} exceeds maximum of 65535",
                    fmt_num(recovery_slices + o.recovery_offset as f64)
                ));
            }
            o.recovery_slices_res = recovery_slices;
        }

        if input_slices < 1.0 && o.recovery_slices_res > 0.0 {
            return Err("Cannot generate recovery from empty input data".into());
        }

        o.output_file_max_slices_res = compat::js_math_min(
            compat::js_math_max(
                apply_rounding(
                    o.output_file_max_slices_rounding,
                    calc_num_recovery_slices(
                        &o.output_file_max_slices,
                        o.slice_size,
                        input_slices,
                        Some(files),
                    )?,
                ),
                1.0,
            ),
            65535.0,
        );
        if o.output_first_file_slices.is_some() {
            o.output_first_file_slices_res = Some(compat::js_math_min(
                compat::js_math_max(
                    apply_rounding(
                        o.output_first_file_slices_rounding,
                        calc_num_recovery_slices(
                            o.output_first_file_slices.as_ref().unwrap(),
                            o.slice_size,
                            input_slices,
                            Some(files),
                        )?,
                    ),
                    1.0,
                ),
                o.recovery_slices_res,
            ));
        }

        if o.min_critical_redundancy < 1.0 {
            return Err("Must have at least one copy of critical packets".into());
        }
        if truthy(o.max_critical_redundancy)
            && o.max_critical_redundancy < o.min_critical_redundancy
        {
            return Err("Maximum critical packet redundancy cannot be below the minimum".into());
        }

        // JS `getMethodNum(GF_METHODS, null)` -> 'Unknown method "null"'
        let gf_method = match &o.gf_method {
            Some(m) => method_num_ref(GF_METHODS, m)?,
            None => method_num(GF_METHODS, "null")?,
        };
        let gf_info = host.gf_info(gf_method)?;

        if truthy(o.process_batch_size.unwrap_or(0.0))
            && !(1.0..=32768.0).contains(&o.process_batch_size.unwrap())
        {
            return Err("Invalid processing batch size".into());
        }
        if o.chunk_read_threads < 1.0 || o.chunk_read_threads > 32768.0 {
            return Err("Invalid number of chunk read threads".into());
        }
        if o.read_buffers < 1.0 || o.read_buffers > 32768.0 {
            return Err("Invalid number of read buffers".into());
        }
        if o.chunk_read_threads > o.read_buffers {
            return Err(format!(
                "Number of chunk read threads ({}) cannot exceed the number of read buffers ({})",
                fmt_num(o.chunk_read_threads),
                fmt_num(o.read_buffers)
            ));
        }
        if o.hash_batch_size < 1.0 || o.hash_batch_size > 65535.0 {
            return Err("Invalid hash batch size".into());
        }
        if o.read_hash_queue < 1.0 || o.read_hash_queue > 32768.0 {
            return Err("Invalid read hash queue size".into());
        }
        if o.read_hash_queue > o.read_buffers {
            return Err(format!(
                "Read hash queue size ({}) cannot exceed number of read buffers ({})",
                fmt_num(o.read_hash_queue),
                fmt_num(o.read_buffers)
            ));
        }
        let mut rec_data_size = match o.rec_data_size {
            Some(r) if truthy(r) => {
                if r < 1.0 || r > 65535.0 {
                    return Err("Invalid recovery buffer count".into());
                }
                r
            }
            _ => (o.hash_batch_size * 1.5).ceil(),
        };
        if o.loop_tile_size < 0.0 {
            return Err("Invalid loop tiling size".into());
        }
        rec_data_size = rec_data_size.min(o.recovery_slices_res);

        let mut staging_count = 2;
        let mut process_batch_size = if truthy(o.process_batch_size.unwrap_or(0.0)) {
            o.process_batch_size.unwrap()
        } else {
            let multiple = gf_info.target_grouping as f64;
            let mut pbs = compat::js_math_round(12.0 / multiple) * multiple;
            if input_slices > 0.0 {
                let batches = (input_slices / pbs).ceil();
                pbs = (input_slices / batches).ceil();
            }
            pbs
        };

        let mut memory_limit = o.memory_limit;
        if o.recovery_slices_res < 1.0 {
            memory_limit = Some(0.0);
        }

        if input_slices < process_batch_size * staging_count as f64 {
            if input_slices < 2.0 {
                process_batch_size = input_slices;
                staging_count = 1;
            } else {
                process_batch_size = (input_slices / staging_count as f64).ceil();
            }
        }

        if memory_limit.is_none() {
            // autodetect from total/free RAM
            let mut total_mem = sys.total_mem as f64;
            // the (uint64_t)-1 sentinel is truthy => constrained path
            let constrained_mem = sys.constrained_mem as f64;
            if constrained_mem != 0.0 {
                total_mem = constrained_mem;
            }
            if total_mem <= 1048576.0 {
                memory_limit = Some(256.0 * 1048576.0);
            } else {
                let mut free_mem = sys.free_mem as f64;
                let mut ml;
                if constrained_mem != 0.0 {
                    // constrained environment - care less about total memory
                    free_mem = free_mem.min(constrained_mem);
                    ml = (free_mem * 0.75).max(total_mem * 0.2);
                } else {
                    ml = ((total_mem * 0.5)
                        * 1.33f64.min(((total_mem / (2048.0 * 1048576.0)) + 0.33).max(0.67)))
                    .min((free_mem * 0.75).max(total_mem * 0.2));
                }
                ml = ml.max(64.0 * 1048576.0);
                if sys.is_64b {
                    ml = ml.min(8192.0 * 1048576.0);
                } else {
                    ml = ml.min(512.0 * 1048576.0);
                }
                memory_limit = Some(ml);
            }
        }

        if truthy(memory_limit.unwrap_or(0.0)) {
            memory_limit = Some(memory_limit.unwrap().min(if sys.is_64b {
                9007199254740991.0
            } else {
                (2048.0 - 64.0) * 1048576.0
            }));
        }

        if o.cpu_min_chunk_size < 2.0 || o.cpu_min_chunk_size % 2.0 != 0.0 {
            return Err(format!(
                "CPU min chunk size ({}) must be even and at least 2 bytes",
                fmt_num(o.cpu_min_chunk_size)
            ));
        }

        // ---- OpenCL device distribution (lines 615-733) ----
        let mut cpu_ratio = 1.0f64;
        let mut min_memory_limit = memory_limit.unwrap_or(0.0);
        if o.cpu_min_chunk_size < o.slice_size
            && !o.opencl_devices.is_empty()
            && o.recovery_slices_res > 0.0
        {
            let mut total_min_chunk = o.cpu_min_chunk_size;
            let mut ocl_used = std::collections::HashSet::new();
            let mut unspecified_ratio_mem: Option<f64> = None;
            for ocl_dev in o.opencl_devices.iter_mut() {
                let dev_info = host
                    .ocl_device_info(ocl_dev.platform, ocl_dev.device)
                    .ok_or("Unable to obtain OpenCL device info")?;
                ocl_dev.platform = dev_info.platform_id;
                ocl_dev.device = dev_info.id;

                let key = format!("{}:{}", ocl_dev.platform, ocl_dev.device);
                if !ocl_used.insert(key.clone()) {
                    return Err(format!("OpenCL device {} listed more than once", key));
                }

                if ocl_dev.memory_limit.is_none() {
                    let mut ml = ((dev_info.memory_global as f64 * 0.8).floor())
                        .max(ocl_dev.min_chunk_size)
                        .max(8.0 * 1048576.0);
                    ml = if sys.is_64b {
                        ml.min(4096.0 * 1048576.0)
                    } else {
                        ml.min(512.0 * 1048576.0)
                    };
                    ocl_dev.memory_limit = Some(ml);
                }

                if truthy(ocl_dev.input_batchsize as f64)
                    && (ocl_dev.input_batchsize < 0 || ocl_dev.input_batchsize > 32768)
                {
                    return Err(format!(
                        "Invalid OpenCL device input batch size ({})",
                        ocl_dev.input_batchsize
                    ));
                }
                if truthy(ocl_dev.target_iters as f64) && ocl_dev.target_iters < 0 {
                    return Err(format!(
                        "Invalid OpenCL device iteration count ({})",
                        ocl_dev.target_iters
                    ));
                }
                if truthy(ocl_dev.target_grouping as f64)
                    && (ocl_dev.target_grouping < 0 || ocl_dev.target_grouping > 65535)
                {
                    return Err(format!(
                        "Invalid OpenCL device grouping size ({})",
                        ocl_dev.target_grouping
                    ));
                }

                // JS `if(oclDev.ratio)` - falsy (0/null) goes to the
                // memory-based pool
                if ocl_dev.ratio.map(truthy).unwrap_or(false) {
                    let r = ocl_dev.ratio.unwrap();
                    if r > 1.0 || r <= 0.0 {
                        return Err(format!(
                            "Invalid OpenCL device processing ratio ({}%)",
                            fmt_num(r * 100.0)
                        ));
                    }
                    if ocl_dev.min_chunk_size / r > o.slice_size {
                        ocl_dev.ratio = Some(ocl_dev.min_chunk_size / o.slice_size);
                    }
                    cpu_ratio -= ocl_dev.ratio.unwrap();
                } else if !truthy(ocl_dev.memory_limit.unwrap_or(0.0)) {
                    return Err("Must specify ratio or memory limit for OpenCL processing".into());
                }

                if ocl_dev.min_chunk_size < 2.0 || ocl_dev.min_chunk_size % 2.0 != 0.0 {
                    return Err(
                        "OpenCL device min chunk size must be even and at least 2 bytes".into(),
                    );
                }
                if let Some(ml) = ocl_dev.memory_limit {
                    if truthy(ml) && ocl_dev.min_chunk_size > ml {
                        return Err(format!(
                            "OpenCL device's memory limit ({}) must be greater than the device's minimum chunk size ({})",
                            friendly_size(ml),
                            friendly_size(ocl_dev.min_chunk_size)
                        ));
                    }
                }
                total_min_chunk += ocl_dev.min_chunk_size;
            }
            if cpu_ratio < 0.0 {
                return Err(format!(
                    "Total OpenCL device processing ratio(s) ({}%) exceed 100%",
                    fmt_num((1.0 - cpu_ratio) * 100.0)
                ));
            }

            if total_min_chunk > o.slice_size {
                // disable devices from the end until it fits
                while let Some(d) = o.opencl_devices.pop() {
                    total_min_chunk -= d.min_chunk_size;
                    if total_min_chunk <= o.slice_size {
                        break;
                    }
                }
            }

            cpu_ratio = 1.0;
            for ocl_dev in &o.opencl_devices {
                if ocl_dev.ratio.map(truthy).unwrap_or(false) {
                    cpu_ratio -= ocl_dev.ratio.unwrap();
                } else {
                    // JS `unspecifiedOclRatioMemory += oclDev.memoryLimit`
                    // (null stays null until first add)
                    unspecified_ratio_mem = Some(
                        unspecified_ratio_mem.unwrap_or(0.0) + ocl_dev.memory_limit.unwrap_or(0.0),
                    );
                }
            }

            if let Some(mut urm) = unspecified_ratio_mem {
                if !truthy(memory_limit.unwrap_or(0.0)) {
                    return Err(
                        "Need to specify CPU memory limit if OpenCL device ratio(s) unspecified"
                            .into(),
                    );
                }
                urm += memory_limit.unwrap();
                loop {
                    let mut min_found = false;
                    for ocl_dev in o.opencl_devices.iter_mut() {
                        if !ocl_dev.ratio.map(truthy).unwrap_or(false) {
                            let test_ratio =
                                (ocl_dev.memory_limit.unwrap_or(0.0) / urm) * cpu_ratio;
                            let min_ratio = ocl_dev.min_chunk_size / o.slice_size;
                            if test_ratio <= min_ratio {
                                ocl_dev.ratio = Some(min_ratio);
                                urm -= ocl_dev.memory_limit.unwrap_or(0.0);
                                cpu_ratio -= min_ratio;
                                min_found = true;
                            }
                        }
                    }
                    if !min_found {
                        break;
                    }
                }
                if cpu_ratio < 0.0 {
                    return Err(
                        "Unable to allocate processing across OpenCL devices based on specified ratios, memory limits and device minimum chunk sizes. Consider specifying a processing ratio across all OpenCL devices"
                            .into(),
                    );
                }
                for ocl_dev in o.opencl_devices.iter_mut() {
                    if !ocl_dev.ratio.map(truthy).unwrap_or(false) {
                        ocl_dev.ratio =
                            Some((ocl_dev.memory_limit.unwrap_or(0.0) / urm) * cpu_ratio);
                    }
                }
                cpu_ratio *= memory_limit.unwrap() / urm;
            }

            if !truthy(min_memory_limit) {
                min_memory_limit = f64::MAX;
            }
            for ocl_dev in &o.opencl_devices {
                let ml = match ocl_dev.memory_limit {
                    Some(m) if truthy(m) => m,
                    _ => continue,
                };
                let dev_scaled = (ml / ocl_dev.ratio.unwrap_or(0.0)).floor();
                min_memory_limit = compat::js_math_min(min_memory_limit, dev_scaled);
            }
            if min_memory_limit == f64::MAX {
                min_memory_limit = 0.0;
            }
        }

        let proc_in_staging_buffer_count = process_batch_size * staging_count as f64;
        let max_chunk_size = o.seq_read_size;
        if truthy(o.min_chunk_size) {
            o.min_chunk_size = compat::js_math_min(o.min_chunk_size, o.slice_size);
            for ocl_dev in &o.opencl_devices {
                let mut ratio_chunk =
                    (ocl_dev.min_chunk_size / ocl_dev.ratio.unwrap_or(0.0)).ceil();
                ratio_chunk += ratio_chunk % 2.0;
                o.min_chunk_size = compat::js_math_max(o.min_chunk_size, ratio_chunk);
            }
            if o.min_chunk_size > o.seq_read_size {
                return Err(format!(
                    "Minimum chunk size ({}) cannot exceed read buffer size ({})",
                    friendly_size(o.min_chunk_size),
                    friendly_size(o.seq_read_size)
                ));
            }
        }

        if let Some(t) = &o.seq_read_throttle {
            create_throttle(t)?;
        }
        if let Some(t) = &o.chunk_read_throttle {
            create_throttle(t)?;
        }

        if let Some(ml) = memory_limit {
            if truthy(ml) {
                let cpu_min_chunk = (cpu_ratio * o.min_chunk_size / 2.0).ceil() * 2.0;
                if truthy(o.min_chunk_size) && (rec_data_size + 1.0) * o.min_chunk_size > ml {
                    return Err(format!(
                        "Cannot accommodate target memory limit ({}) with target minimum chunk size ({}). At least {} buffers of the minimum chunk size need to be allocated.",
                        friendly_size(ml),
                        friendly_size(o.min_chunk_size),
                        fmt_num(rec_data_size + 1.0)
                    ));
                }
                if cpu_min_chunk * (proc_in_staging_buffer_count + 1.0) > ml {
                    return Err(format!(
                        "Cannot accommodate target memory limit ({}) with target minimum CPU chunk size ({}). At least {} buffers of the minimum chunk size need to be allocated.",
                        friendly_size(ml),
                        friendly_size(cpu_min_chunk),
                        fmt_num(proc_in_staging_buffer_count + 1.0)
                    ));
                }
            }
        }

        // ---- chunk / pass layout (lines 762-809) ----
        let req_mem = o.slice_size * o.recovery_slices_res
            + (rec_data_size * o.slice_size)
                .max(proc_in_staging_buffer_count * (o.slice_size * cpu_ratio / 2.0).ceil() * 2.0);
        let mut passes = 1.0f64;
        let mut chunks = 1.0f64;
        let mut slices_per_pass = o.recovery_slices_res;
        let chunk_size;
        let mut n_passes = if truthy(min_memory_limit) {
            (req_mem / min_memory_limit).ceil()
        } else {
            1.0
        };
        let min_passes = (o.slice_size / max_chunk_size).ceil();
        n_passes = compat::js_math_max(n_passes, min_passes);
        if n_passes > 1.0 && o.recovery_slices_res > 0.0 {
            let mut cs = (o.slice_size / n_passes).ceil() - 1.0;
            cs += cs % 2.0;
            let mut min_chunk = if o.min_chunk_size <= 0.0 {
                compat::js_math_min(o.slice_size, sys.max_buffer_size() as f64)
            } else {
                o.min_chunk_size
            };
            if !truthy(min_memory_limit) && cs < min_chunk {
                if truthy(o.min_chunk_size) {
                    min_chunk = cs; // ignore underflowing requested minimum
                } else {
                    return Err(format!(
                        "Cannot disable chunking as maximum chunk size ({}) is less than the slice size ({})",
                        friendly_size(max_chunk_size),
                        friendly_size(o.slice_size)
                    ));
                }
            }
            if cs < min_chunk {
                // partial recovery generation - multiple passes needed
                chunks = (o.slice_size / min_chunk).ceil();
                cs = (o.slice_size / chunks).ceil();
                cs += cs % 2.0;
                slices_per_pass = (min_memory_limit / cs).floor();
                if let Some(ml) = memory_limit {
                    if truthy(ml) {
                        let overhead =
                            (proc_in_staging_buffer_count * (cs * cpu_ratio / 2.0).ceil() * 2.0)
                                .max(rec_data_size * cs);
                        let cpu_slices = ((ml - overhead) / cs).floor();
                        if cpu_slices < 1.0 {
                            return Err(format!(
                                "Cannot accommodate memory limit ({}); a chunk size of {} was chosen, but a minimum of 1 recovery + {} processing buffer needs to be held in memory",
                                friendly_size(ml),
                                friendly_size(cs),
                                friendly_size(overhead)
                            ));
                        }
                        slices_per_pass = compat::js_math_min(slices_per_pass, cpu_slices);
                    }
                }
                if slices_per_pass < 1.0 {
                    return Err(format!(
                        "Cannot accommodate target memory limit ({}); a minimum of 1 chunk of size {} needs to be held in memory",
                        friendly_size(min_memory_limit),
                        friendly_size(cs)
                    ));
                }
                if cs > max_chunk_size {
                    return Err("Cannot accommodate minimum/maximum chunk size".into());
                }
                passes = (o.recovery_slices_res / slices_per_pass).ceil();
            } else {
                chunks = n_passes;
            }
            chunk_size = cs;
        } else {
            chunk_size = o.slice_size;
        }

        // ---- break up chunks across devices (lines 811-825) ----
        let mut proc_cpu = CpuSpec {
            method: gf_info.id,
            input_batchsize: process_batch_size as u32,
            input_minbatchsize: 0,
            chunk_size: o.loop_tile_size as u32,
            slice_offset: 0,
            slice_size: 0,
        };
        let mut slice_offset = 0u64;
        let mut kept_devs: Vec<OclDevOpt> = Vec::new();
        for ocl_dev in o.opencl_devices.iter_mut() {
            ocl_dev.slice_offset = slice_offset;
            ocl_dev.slice_size =
                (compat::js_math_round(ocl_dev.ratio.unwrap_or(0.0) * chunk_size / 2.0) * 2.0)
                    as u64;
            slice_offset += ocl_dev.slice_size;
            ocl_dev.cksum_method = gf_info.id;
            if ocl_dev.slice_size > 0 {
                kept_devs.push(ocl_dev.clone());
            }
        }
        proc_cpu.slice_offset = slice_offset;
        // rounded device ratios can sum past chunk_size - JS goes negative
        // and skips the CPU spec entirely
        let proc_cpu_size = (chunk_size as u64).saturating_sub(slice_offset);
        proc_cpu.slice_size = proc_cpu_size;
        let proc_cpu_opt = if proc_cpu_size < 1 {
            None
        } else {
            Some(proc_cpu)
        };
        let mut proc_ocl = Vec::new();
        for d in &kept_devs {
            proc_ocl.push(OclSpec {
                platform: d.platform,
                device: d.device,
                slice_size: d.slice_size,
                slice_offset: d.slice_offset,
                // JS `if(spec.method) spec.method = getMethodNum(...)` -
                // falsy method (null/''/0) stays raw; binding treats 0 as
                // GF16OCL_AUTO
                method: match &d.method {
                    Some(m) if method_truthy(m) => method_num_ref(GFOCL_METHODS, m)?,
                    _ => 0,
                },
                cksum_method: d.cksum_method,
                input_batchsize: d.input_batchsize,
                input_minbatchsize: 0,
                target_iters: d.target_iters.max(0) as u32,
                target_grouping: d.target_grouping.max(0) as u32,
            });
        }

        // ---- display filenames (lines 827-888) ----
        if o.display_name_format == "outrel" {
            o.display_name_format = "path".into();
            o.display_name_base = path_dirname(&o.output_base);
        }
        match o.display_name_format.as_str() {
            "basename" => {
                for f in files.iter_mut() {
                    if f.display_name.is_none() {
                        f.display_name = Some(path_basename(&f.name));
                    }
                }
            }
            "keep" => {
                for f in files.iter_mut() {
                    if f.display_name.is_none() {
                        f.display_name = Some(sysinfo::path_to_par2(&f.name));
                    }
                }
            }
            "path" => {
                let base = path_resolve(&o.display_name_base);
                for f in files.iter_mut() {
                    if f.display_name.is_none() {
                        let rel = path_relative(&base, &f.name);
                        let mut dn = sysinfo::path_to_par2(&rel);
                        if dn.is_empty() {
                            dn = ".".into();
                        }
                        f.display_name = Some(dn);
                    }
                }
            }
            // 'common' (default) + anything unrecognised
            _ => {
                let mut common_root: Option<Vec<String>> = None;
                for f in files.iter_mut() {
                    let full = path_resolve(&f.name);
                    let dir = path_dirname(&sysinfo::path_normalize(&full));
                    // path_dirname always returns '/'-separated paths (even
                    // on Windows) - split on '/', not the platform sep
                    let parts: Vec<String> = dir.split('/').map(|s| s.to_string()).collect();
                    match &mut common_root {
                        None => common_root = Some(parts),
                        Some(root) => {
                            let mut i = 0;
                            while i < root.len() && i < parts.len() && root[i] == parts[i] {
                                i += 1;
                            }
                            root.truncate(i);
                        }
                    }
                    f.full_path = Some(full);
                }
                if let Some(mut root) = common_root {
                    if !root.is_empty() {
                        if root.last().map(|s| s.is_empty()) == Some(true) {
                            root.pop(); // fix for root directories
                        }
                        let prefix = format!("{}/", root.join("/"));
                        for f in files.iter_mut() {
                            if f.display_name.is_none() {
                                let full = f.full_path.as_deref().unwrap_or("");
                                // JS takes substring(stripLen) where stripLen
                                // is the root prefix + one separator
                                let dn = full
                                    .strip_prefix(&prefix)
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| {
                                        // fall back to unit-index slicing
                                        // (should not happen)
                                        full.chars().skip(prefix.chars().count()).collect()
                                    });
                                f.display_name = Some(sysinfo::path_to_par2(&dn));
                            }
                        }
                    }
                }
            }
        }

        // ---- PAR2 + critical packets (lines 890-911) ----
        let par_files: Vec<([u8; 16], u64, String, String, Option<std::path::PathBuf>)> = files
            .iter()
            .map(|f| {
                let name = f
                    .display_name
                    .clone()
                    .unwrap_or_else(|| path_basename(&f.name));
                (f.md5_16k, f.size, f.name.clone(), name, f.fs_path.clone())
            })
            .collect();
        // JS PAR2 gets no recDataSize - it defaults to ceil(hb*1.5)
        let gf_opts = GfOpts {
            hash_batch_size: o.hash_batch_size as usize,
            rec_data_size: (o.hash_batch_size * 1.5).ceil() as usize,
        };
        let proc_opts = ProcOpts {
            cpu: proc_cpu_opt.clone(),
            ocl: proc_ocl.clone(),
            staging: staging_count,
            threads: o.num_threads,
        };
        let par2 = Par2::new(
            par_files,
            o.slice_size as u64,
            ascii_charset.to_string(),
            ctx,
            gf_opts,
            proc_opts,
        )?;

        let unicode = match o.unicode {
            Some(true) => Some(CharOpt::Both),
            Some(false) => Some(CharOpt::Ascii),
            None => None,
        };
        let mut crit_packets = Vec::new();
        for (i, f) in par2.files.iter().enumerate() {
            crit_packets.push(GenPacket {
                pkt_type: PacketType::FileDesc,
                size: f.packet_description_size(unicode, ascii_charset) as u64,
                index: i as i64,
                data_chunk_offset: 0,
                data: Vec::new(),
            });
            if f.size > 0 {
                crit_packets.push(GenPacket {
                    pkt_type: PacketType::FileChk,
                    size: f.packet_checksums_size() as u64,
                    index: i as i64,
                    data_chunk_offset: 0,
                    data: Vec::new(),
                });
            }
        }
        for (i, cmt) in o.comments.iter().enumerate() {
            crit_packets.push(GenPacket {
                pkt_type: PacketType::Comment,
                size: par2.packet_comment_size(cmt, unicode)? as u64,
                index: i as i64,
                data_chunk_offset: 0,
                data: Vec::new(),
            });
        }
        crit_packets.push(GenPacket {
            pkt_type: PacketType::Main,
            size: par2.packet_main_size() as u64,
            index: -1,
            data_chunk_offset: 0,
            data: Vec::new(),
        });
        let creator_pkt = GenPacket {
            pkt_type: PacketType::Creator,
            size: par2.packet_creator_size(&o.creator) as u64,
            index: -1,
            data_chunk_offset: 0,
            data: Vec::new(),
        };

        // JS resolves o.memoryLimit in place (null->auto); write it back
        // as a number (0 = no limit, matching the falsy JS value)
        o.memory_limit = Some(memory_limit.unwrap_or(0.0));
        let mut gen = Par2Gen {
            opts: o,
            par2,
            total_size,
            input_slices,
            recovery_files: Vec::new(),
            crit_packets,
            creator_pkt,
            passes,
            chunks,
            slices_per_pass,
            chunk_size,
            slice_nums: Vec::new(),
            read_size: 0.0,
            proc_in_staging_buffer_count,
            staging_count,
            cpu_ratio,
            min_memory_limit,
            opencl_devices: kept_devs,
            proc_cpu: proc_cpu_opt,
            proc_ocl,
            process_batch_size,
            rec_data_size,
        };

        if gen.opts.output_index {
            gen.rf_push(0, SliceRef::Offset(0))?;
        }

        if gen.opts.output_file_count < 0.0
            || gen.opts.output_file_count.is_nan()
            || !gen.opts.output_file_count.is_finite()
        {
            return Err("Invalid number of recovery files".into());
        }
        gen.opts.output_file_count_res = compat::to_int32(gen.opts.output_file_count);
        if gen.opts.output_file_count_res as f64 > gen.opts.recovery_slices_res {
            return Err(format!(
                "Cannot allocate {} recovery slices to {} volumes as there aren't enough slices",
                fmt_num(gen.opts.recovery_slices_res),
                gen.opts.output_file_count_res
            ));
        }

        if gen.total_size > 0.0 {
            if let Some(exps) = gen.opts.recovery_exponents_int.clone() {
                let n = exps.len() as i64;
                gen.rf_push(n, SliceRef::Exponents(exps))?;
            } else if gen.opts.output_size_scheme == "pow2" {
                let start_slices = match gen.opts.output_first_file_slices_res {
                    Some(v) if truthy(v) => v,
                    _ => 1.0,
                };
                let total_slices_i =
                    (gen.opts.recovery_slices_res + gen.opts.recovery_offset as f64) as i64;
                let recovery_offset = gen.opts.recovery_offset as i64;
                let max_slices = gen.opts.output_file_max_slices_res as i64;
                let get_list = |slices0: i64| -> Vec<(i64, i64)> {
                    let mut result = Vec::new();
                    let mut slices = slices0;
                    let mut i = 0i64;
                    // JS exits via `i += slices` going NaN (NaN < total is
                    // false); a NaN maxSlices casts to 0 here, so guard on
                    // slices going non-positive (also terminates the JS
                    // pathological negative-slices hang)
                    while i < total_slices_i && slices > 0 {
                        let f_slices = slices.min(total_slices_i - i);
                        if i + f_slices > recovery_offset {
                            if recovery_offset > i {
                                result.push((f_slices - (recovery_offset - i), recovery_offset));
                            } else {
                                result.push((f_slices, i));
                            }
                        }
                        i += slices;
                        slices = (slices * 2).min(max_slices);
                    }
                    result
                };
                let slice_files_list;
                if gen.opts.output_file_count_res != 0 {
                    let mut found = None;
                    let mut s: i64 = 1;
                    while s <= max_slices {
                        let test_list = get_list(s);
                        if test_list.len() as i64 == gen.opts.output_file_count_res as i64 {
                            found = Some(test_list);
                            break;
                        }
                        if (test_list.len() as i64) < gen.opts.output_file_count_res as i64 {
                            break;
                        }
                        s *= 2;
                    }
                    match found {
                        Some(l) => slice_files_list = l,
                        None => {
                            return Err(format!(
                            "Unable to find a set of parameters to generate {} recovery volume(s)",
                            gen.opts.output_file_count_res
                        ))
                        }
                    }
                } else {
                    slice_files_list = get_list(start_slices as i64);
                }
                for (n, off) in slice_files_list {
                    gen.rf_push(n, SliceRef::Offset(off))?;
                }
            } else if gen.opts.output_size_scheme == "uniform" {
                let mut num_files = if gen.opts.output_file_count_res != 0 {
                    gen.opts.output_file_count_res as f64
                } else {
                    (gen.opts.recovery_slices_res / gen.opts.output_file_max_slices_res).ceil()
                };
                let mut slice_pos = 0f64;
                loop {
                    // while(numFiles--) - NaN is falsy in JS and exits;
                    // `!(x > 0)` covers NaN and non-positive alike
                    if !(num_files > 0.0) {
                        break;
                    }
                    num_files -= 1.0;
                    let n_slices =
                        ((gen.opts.recovery_slices_res - slice_pos) / (num_files + 1.0)).ceil();
                    gen.rf_push(
                        n_slices as i64,
                        SliceRef::Offset((slice_pos + gen.opts.recovery_offset as f64) as i64),
                    )?;
                    slice_pos += n_slices;
                }
            } else {
                // 'equal' - JS checks truthiness of the option (a resolved
                // 0 is falsy and doesn't produce a first file)
                if let Some(ff) = gen.opts.output_first_file_slices_res.filter(|&v| truthy(v)) {
                    gen.rf_push(ff as i64, SliceRef::Offset(gen.opts.recovery_offset as i64))?;
                }
                let mut slices_per_file = gen.opts.output_file_max_slices_res;
                if gen.opts.output_file_count_res != 0 {
                    if let Some(ff) = gen.opts.output_first_file_slices_res.filter(|&v| truthy(v)) {
                        let remaining_slices = gen.opts.recovery_slices_res - ff;
                        if gen.opts.output_file_count_res < 2 {
                            if remaining_slices != 0.0 {
                                return Err(format!(
                                    "Cannot allocate slices: only one recovery volume requested, which is allocated {} slice(s), leaving {} slice(s) unallocated",
                                    fmt_num(ff),
                                    fmt_num(remaining_slices)
                                ));
                            }
                            // JS writes `slicesPerFile = 0` here ("no
                            // slices to allocate") then always overwrites
                            // it below - dead store, dropped.
                        }
                        if gen.opts.output_file_count_res as f64 - 1.0 > remaining_slices {
                            return Err(format!(
                                "Cannot allocate {} recovery slice(s) amongst {} recovery volume(s)",
                                fmt_num(remaining_slices),
                                fmt_num(gen.opts.output_file_count_res as f64 - 1.0)
                            ));
                        }
                        slices_per_file = (remaining_slices
                            / (gen.opts.output_file_count_res as f64 - 1.0))
                            .ceil();
                    } else {
                        slices_per_file = (gen.opts.recovery_slices_res
                            / gen.opts.output_file_count_res as f64)
                            .ceil();
                    }
                }
                // JS `o.outputFirstFileSlices || 0` - NaN is falsy -> 0
                let mut i = gen
                    .opts
                    .output_first_file_slices_res
                    .filter(|&v| truthy(v))
                    .unwrap_or(0.0);
                while i < gen.opts.recovery_slices_res {
                    gen.rf_push(
                        // Math.min propagates NaN (NaN -> falsy 0-slice file)
                        compat::js_math_min(slices_per_file, gen.opts.recovery_slices_res - i)
                            as i64,
                        SliceRef::Offset((i + gen.opts.recovery_offset as f64) as i64),
                    )?;
                    i += slices_per_file;
                }
            }
        }

        if gen.recovery_files.is_empty() {
            return Err(
                "Nothing to generate; need to generate an index or at least one recovery slice"
                    .into(),
            );
        }
        if gen.recovery_files.len() == 1 && gen.opts.recovery_offset == 0 {
            gen.recovery_files[0].name = format!("{}.par2", gen.opts.output_base);
        }

        // `_sliceNums` - JS `Array(o.recoverySlices)` sparsely filled:
        // trailing entries stay undefined when packets < recoverySlices
        let mut slice_nums = vec![None; gen.opts.recovery_slices_res.max(0.0) as usize];
        let mut i = 0usize;
        for rf in &gen.recovery_files {
            for p in &rf.packets {
                if p.pkt_type == PacketType::Recovery && i < slice_nums.len() {
                    slice_nums[i] = Some(p.index);
                    i += 1;
                }
            }
        }
        gen.slice_nums = slice_nums;

        // ---- readSize (lines 1013-1034) ----
        let read_size = if gen.chunks > 1.0 {
            if gen.opts.seq_read_size < gen.chunk_size {
                return Err(format!(
                    "Read size ({}) cannot be smaller than chunking size ({})",
                    friendly_size(gen.opts.seq_read_size),
                    friendly_size(gen.chunk_size)
                ));
            }
            gen.opts.seq_read_size
        } else if gen.opts.slice_size <= gen.opts.seq_read_size {
            (gen.opts.seq_read_size / gen.opts.slice_size).floor() * gen.opts.slice_size
        } else if gen.opts.recovery_slices_res < 1.0 {
            gen.opts.seq_read_size
        } else {
            if truthy(gen.min_memory_limit) && truthy(gen.opts.min_chunk_size) {
                return Err("Cannot read a slice with a single read call".into());
            }
            gen.chunk_size
        };
        gen.read_size = read_size;

        Ok(gen)
    }

    /// `_unicodeOpt`.
    pub fn unicode_opt(&self) -> Option<CharOpt> {
        match self.opts.unicode {
            Some(true) => Some(CharOpt::Both),
            Some(false) => Some(CharOpt::Ascii),
            None => None,
        }
    }

    /// `this.files` = `par2.getFiles()` alias.
    pub fn files(&self) -> &[Par2File] {
        &self.par2.files
    }

    /// `_rfPush` - build a recovery volume's packet list + push it.
    fn rf_push(&mut self, num_slices: i64, slice_ref: SliceRef) -> JsResult<()> {
        let mut crit_total_size: u64 = self.crit_packets.iter().map(|p| p.size).sum();
        let recv_size = if num_slices != 0 {
            self.par2.packet_recovery_size()
        } else {
            0
        };
        let mut packets: Vec<GenPacket> = Vec::new();

        if self.opts.critical_redundancy_scheme != CritScheme::None {
            if num_slices != 0 {
                // repeat critical packets pow2-style, spread amongst recovery
                let scheme = match &self.opts.critical_redundancy_scheme {
                    CritScheme::Spec(s) => s,
                    CritScheme::None => unreachable!(),
                };
                let mut crit_copies = compat::js_math_round(calc_num_recovery_slices(
                    scheme,
                    crit_total_size as f64,
                    num_slices as f64,
                    None,
                )?);
                crit_copies = compat::js_math_max(crit_copies, self.opts.min_critical_redundancy);
                if truthy(self.opts.max_critical_redundancy) {
                    crit_copies =
                        compat::js_math_min(crit_copies, self.opts.max_critical_redundancy);
                }
                // JS sizes `packets = Array(numSlices + critNum + 1)` -
                // Array() throws RangeError("Invalid array length") on
                // n > 2^32-1, non-integer, or negative lengths
                if !(crit_copies >= 0.0 && crit_copies.fract() == 0.0) {
                    return Err("Invalid array length".into());
                }
                let crit_num = (crit_copies as usize) * self.crit_packets.len();
                if num_slices as f64 + crit_num as f64 + 1.0 > 4294967295.0 {
                    return Err("Invalid array length".into());
                }
                let crit_ratio = crit_num as f64 / num_slices as f64;

                let mut crit_written = 0usize;
                for i in 0..num_slices {
                    let exp = match &slice_ref {
                        SliceRef::Exponents(e) => e[i as usize] as i64,
                        SliceRef::Offset(off) => i + *off,
                    };
                    packets.push(GenPacket {
                        pkt_type: PacketType::Recovery,
                        size: recv_size,
                        index: exp,
                        data_chunk_offset: 0,
                        data: Vec::new(),
                    });
                    while (crit_written as f64)
                        < compat::js_math_round(crit_ratio * (i as f64 + 1.0))
                    {
                        let p = self.crit_packets[crit_written % self.crit_packets.len()].clone();
                        packets.push(p);
                        crit_written += 1;
                    }
                }
                packets.push(self.creator_pkt.clone());
                crit_total_size *= crit_copies as u64;
            } else {
                // JS `packets = Array(numCritPackets + 1)` - RangeError
                // on length > 2^32-1, NaN, negative, or non-integer
                let ncrit_len =
                    self.opts.min_critical_redundancy * self.crit_packets.len() as f64 + 1.0;
                if !(ncrit_len >= 0.0 && ncrit_len <= 4294967295.0 && ncrit_len.fract() == 0.0) {
                    return Err("Invalid array length".into());
                }
                let num_crit = self.opts.min_critical_redundancy as usize * self.crit_packets.len();
                for _ in 0..self.opts.min_critical_redundancy as usize {
                    for p in &self.crit_packets {
                        packets.push(p.clone());
                    }
                }
                packets.push(self.creator_pkt.clone());
                crit_total_size *= self.opts.min_critical_redundancy as u64;
                let _ = num_crit;
            }
        } else {
            // no critical repetition - single copy at the end
            for i in 0..num_slices {
                let exp = match &slice_ref {
                    SliceRef::Exponents(e) => e[i as usize] as i64,
                    SliceRef::Offset(off) => i + *off,
                };
                packets.push(GenPacket {
                    pkt_type: PacketType::Recovery,
                    size: recv_size,
                    index: exp,
                    data_chunk_offset: 0,
                    data: Vec::new(),
                });
            }
            for p in &self.crit_packets {
                packets.push(p.clone());
            }
            packets.push(self.creator_pkt.clone());
        }

        let recovery_index = self
            .recovery_files
            .last()
            .map(|f| f.recovery_index + f.recovery_slices)
            .unwrap_or(0);
        let suffix = match &slice_ref {
            SliceRef::Exponents(_) => String::new(),
            SliceRef::Offset(off) => par2_ext(
                num_slices,
                *off,
                (self.opts.recovery_slices_res + self.opts.recovery_offset as f64) as i64,
                self.opts.output_alt_naming_scheme,
            )?,
        };
        self.recovery_files.push(RecoveryFile {
            name: format!("{}{}", self.opts.output_base, suffix),
            recovery_slices: num_slices,
            recovery_index,
            packets,
            total_size: crit_total_size + self.creator_pkt.size + recv_size * num_slices as u64,
        });
        Ok(())
    }

    /// `numChunksThisChunkPass` - input slices remaining at `chunk_offset`.
    pub fn num_chunks_this_chunk_pass(&self, chunk_offset: f64) -> f64 {
        self.par2.files.iter().fold(0.0, |count, f| {
            if f.size == 0 || (f.size as f64) <= chunk_offset {
                count
            } else {
                count + ((f.size as f64 - chunk_offset) / self.opts.slice_size).ceil()
            }
        })
    }

    /// `totalNumInputChunks`.
    pub fn total_num_input_chunks(&self) -> f64 {
        self.par2.files.iter().fold(0.0, |count, f| {
            if f.size > 0 {
                let non_last = f.num_slices as f64 - 1.0;
                let mut c = count + non_last * self.chunks;
                let last_slice_size = f.size as f64 - non_last * self.opts.slice_size;
                c += (last_slice_size / self.chunk_size).ceil();
                c
            } else {
                count
            }
        })
    }
}

/// `_rfPush`'s `sliceOffsetOrExponents`.
enum SliceRef {
    Offset(i64),
    Exponents(Vec<i32>),
}

// ---------------------------------------------------------------------------
// Node `path` (POSIX) ports - displayNameFormat handling
// ---------------------------------------------------------------------------

/// `path.resolve` for POSIX paths - resolves the argument against cwd and
/// normalizes dot segments. On Windows, `\` is folded to `/` and
/// drive-letter (`C:/`) + UNC (`//`) forms count as absolute.
fn path_resolve(p: &str) -> String {
    let cwd = std::env::current_dir()
        .map(|d| d.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| "/".to_string());
    #[cfg(windows)]
    let p = &*p.replace('\\', "/");
    #[cfg(windows)]
    let absolute =
        p.starts_with('/') || (p.len() >= 3 && p.as_bytes()[1] == b':' && p.as_bytes()[2] == b'/');
    #[cfg(not(windows))]
    let absolute = p.starts_with('/');
    let joined = if absolute {
        p.to_string()
    } else {
        format!("{}/{}", cwd.trim_end_matches('/'), p)
    };
    path_normalize_posix(&joined)
}

/// `path.relative` (POSIX) - Node's full algorithm on resolved paths.
fn path_relative(from: &str, to: &str) -> String {
    let from = path_resolve(from);
    let to = path_resolve(to);
    if from == to {
        return String::new();
    }
    let from_parts: Vec<&str> = from.trim_start_matches('/').split('/').collect();
    let to_parts: Vec<&str> = to.trim_start_matches('/').split('/').collect();
    let mut i = 0;
    while i < from_parts.len() && i < to_parts.len() && from_parts[i] == to_parts[i] {
        i += 1;
    }
    let mut out = String::new();
    for _ in i..from_parts.len() {
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str("..");
    }
    for p in &to_parts[i..] {
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str(p);
    }
    out
}

/// `path.basename` (POSIX) - trailing slashes ignored.
fn path_basename(p: &str) -> String {
    #[cfg(windows)]
    let p = &*p.replace('\\', "/");
    let s = p.trim_end_matches('/');
    match s.rfind('/') {
        Some(i) => s[i + 1..].to_string(),
        None => s.to_string(),
    }
}

/// `path.dirname` (POSIX).
fn path_dirname(p: &str) -> String {
    #[cfg(windows)]
    let p = &*p.replace('\\', "/");
    if p == "/" {
        return "/".to_string();
    }
    let s = p.trim_end_matches('/');
    match s.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => s[..i].to_string(),
        None => ".".to_string(),
    }
}

/// `path.normalize` (POSIX) - resolve `.`/`..`, collapse separators.
fn path_normalize_posix(p: &str) -> String {
    let abs = p.starts_with('/');
    let mut segs: Vec<&str> = Vec::new();
    for part in p.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if !segs.is_empty() && *segs.last().unwrap() != ".." {
                    segs.pop();
                } else if !abs {
                    segs.push("..");
                }
            }
            s => segs.push(s),
        }
    }
    let mut out = String::new();
    if abs {
        out.push('/');
    }
    out.push_str(&segs.join("/"));
    if out.is_empty() {
        out.push('.');
    }
    out
}

impl GenOpts {
    /// `CliConfig` -> solver `o` (`_extend(optsDefault, ppo)` semantics:
    /// every `Some`/`present` field overrides the default).
    pub fn from_cli(cfg: &crate::args::CliConfig) -> Self {
        use crate::args::CliConfig as C;
        let ppo = &cfg.ppo;
        let mut o = GenOpts::default();
        let C {
            min_slice_size,
            max_slice_size,
            recovery_slices,
            min_recovery_slices,
            max_recovery_slices,
            output_file_max_slices,
            output_file_max_slices_rounding,
            output_first_file_slices,
            output_first_file_slices_rounding,
            ..
        } = cfg;
        o.output_base = ppo.output_base.clone();
        o.min_slice_size = min_slice_size.clone();
        o.max_slice_size = max_slice_size.clone();
        // JS `false` (unparseable) coerces to 0 in arithmetic
        o.slice_size_multiple = ppo.slice_size_multiple.unwrap_or(0.0);
        if let Some(e) = recovery_slices {
            o.recovery_slices = e.clone();
        }
        o.min_recovery_slices = min_recovery_slices.clone();
        // GenOpts default is {slices:65536} - only an explicit
        // --max-recovery-slices may replace it; assigning None would drop
        // the ceiling and fail any non-integral -r spec
        if let Some(m) = max_recovery_slices {
            o.max_recovery_slices = Some(m.clone());
        }
        if let Some(v) = ppo.recovery_offset {
            o.recovery_offset = v;
        }
        o.recovery_exponents = ppo.recovery_exponents.clone();
        if let Some(v) = ppo.memory_limit {
            o.memory_limit = Some(v);
        }
        if let Some(v) = ppo.min_chunk_size {
            o.min_chunk_size = v;
        }
        if let Some(v) = ppo.process_batch_size {
            o.process_batch_size = Some(v as f64);
        }
        if let Some(v) = ppo.hash_batch_size {
            o.hash_batch_size = v as f64;
        }
        if let Some(v) = ppo.rec_data_size {
            o.rec_data_size = Some(v as f64);
        }
        o.comments = ppo.comments.clone().unwrap_or_default();
        o.creator = ppo.creator.clone();
        o.unicode = ppo.unicode;
        if let Some(v) = ppo.output_overwrite {
            o.output_overwrite = v;
        }
        if let Some(v) = ppo.output_sync {
            o.output_sync = v;
        }
        if let Some(v) = ppo.output_index {
            o.output_index = v;
        }
        if let Some(v) = &ppo.output_size_scheme {
            o.output_size_scheme = v.clone();
        }
        if let Some(v) = output_first_file_slices {
            o.output_first_file_slices = Some(v.clone());
            o.output_first_file_slices_rounding = *output_first_file_slices_rounding;
        }
        if let Some(v) = output_file_max_slices {
            o.output_file_max_slices = v.clone();
            o.output_file_max_slices_rounding = *output_file_max_slices_rounding;
        }
        if let Some(v) = ppo.output_file_count {
            o.output_file_count = v as f64;
        }
        if let Some(v) = &ppo.critical_redundancy_scheme {
            o.critical_redundancy_scheme = CritScheme::from_cli(v);
        }
        if let Some(v) = ppo.min_critical_redundancy {
            o.min_critical_redundancy = v as f64;
        }
        if let Some(v) = ppo.max_critical_redundancy {
            o.max_critical_redundancy = v as f64;
        }
        if let Some(v) = ppo.output_alt_naming_scheme {
            o.output_alt_naming_scheme = v;
        }
        if let Some(v) = &ppo.display_name_format {
            o.display_name_format = v.clone();
        }
        if let Some(v) = &ppo.display_name_base {
            o.display_name_base = v.clone();
        }
        if let Some(v) = ppo.seq_read_size {
            o.seq_read_size = v;
        }
        if let Some(v) = ppo.chunk_read_threads {
            o.chunk_read_threads = v as f64;
        }
        if let Some(v) = &ppo.seq_read_throttle {
            o.seq_read_throttle = v.clone();
        }
        if let Some(v) = &ppo.chunk_read_throttle {
            o.chunk_read_throttle = v.clone();
        }
        if let Some(v) = ppo.read_buffers {
            o.read_buffers = v as f64;
        }
        if let Some(v) = ppo.read_hash_queue {
            o.read_hash_queue = v as f64;
        }
        if let Some(v) = ppo.num_threads {
            o.num_threads = Some(v);
        }
        if let Some(v) = &ppo.gf_method {
            o.gf_method = Some(MethodRef::Name(v.clone()));
        }
        if let Some(v) = ppo.loop_tile_size {
            o.loop_tile_size = v;
        }
        if let Some(devs) = &ppo.opencl_devices {
            o.opencl_devices = devs
                .iter()
                .map(|d| OclDevOpt {
                    platform: d.platform.unwrap_or(-1),
                    device: d.device.unwrap_or(-1),
                    ratio: d.ratio,
                    memory_limit: d.memory_limit,
                    method: d.method.as_ref().map(|m| MethodRef::Name(m.clone())),
                    input_batchsize: d.input_batchsize.unwrap_or(0),
                    target_iters: d.target_iters.unwrap_or(0),
                    target_grouping: d.target_grouping.unwrap_or(0),
                    min_chunk_size: d.min_chunk_size.unwrap_or(32768.0),
                    ..Default::default()
                })
                .collect();
        }
        if let Some(v) = ppo.cpu_min_chunk_size {
            o.cpu_min_chunk_size = v;
        }
        o
    }
}

impl<C: GfCtx> Par2Gen<C> {
    /// JSON dump of the solver layout for differential comparison with
    /// `test/dump_gen.js`. Numbers are JS-number strings via `fmt_num` to
    /// preserve f64 formatting; strings/nulls map to JSON natively.
    pub fn dump(&self) -> serde_json::Value {
        use serde_json::json;
        let num = |v: f64| -> serde_json::Value { fmt_num(v).into() };
        let opt_num = |v: Option<f64>| -> serde_json::Value {
            v.map(|x| fmt_num(x).into())
                .unwrap_or(serde_json::Value::Null)
        };
        let files: Vec<_> = self
            .recovery_files
            .iter()
            .map(|f| {
                json!({
                    "name": f.name,
                    "recoverySlices": f.recovery_slices,
                    "recoveryIndex": f.recovery_index,
                    "totalSize": f.total_size,
                    "packets": f.packets.iter().map(|p| json!({
                        "type": match p.pkt_type {
                            PacketType::FileDesc => "filedesc",
                            PacketType::FileChk => "filechk",
                            PacketType::Comment => "comment",
                            PacketType::Main => "main",
                            PacketType::Creator => "creator",
                            PacketType::Recovery => "recovery",
                        },
                        "size": p.size,
                        "index": p.index,
                        "dataChunkOffset": p.data_chunk_offset,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        // `gfopts.proc_ocl` IS `o.openclDevices` (same array): PAR2's
        // `if(spec.method) spec.method = getMethodNum(...)` resolves names
        // in place; falsy methods stay raw. Emit one shared entry list.
        let ocl: Vec<_> = self
            .opencl_devices
            .iter()
            .zip(self.proc_ocl.iter())
            .map(|(d, s)| {
                let method: serde_json::Value = match &d.method {
                    None => serde_json::Value::Null,
                    Some(m) if method_truthy(m) => fmt_num(s.method as f64).into(),
                    Some(MethodRef::Index(n)) => fmt_num(*n).into(),
                    Some(MethodRef::Name(name)) => name.clone().into(),
                };
                json!({
                    "platform": d.platform,
                    "device": d.device,
                    "ratio": opt_num(d.ratio),
                    "memoryLimit": opt_num(d.memory_limit),
                    "method": method,
                    "input_batchsize": d.input_batchsize,
                    "target_iters": d.target_iters,
                    "target_grouping": d.target_grouping,
                    "minChunkSize": num(d.min_chunk_size),
                    "slice_offset": d.slice_offset,
                    "slice_size": d.slice_size,
                    "cksum_method": s.cksum_method,
                })
            })
            .collect();
        let display_names: Vec<serde_json::Value> = self
            .par2
            .files()
            .iter()
            .map(|f| f.display_name.clone().into())
            .collect();
        json!({
            "sliceSize": num(self.opts.slice_size),
            "inputSlices": num(self.input_slices),
            "recoverySlices": num(self.opts.recovery_slices_res),
            "chunkSize": num(self.chunk_size),
            "chunks": num(self.chunks),
            "passes": num(self.passes),
            "slicesPerPass": num(self.slices_per_pass),
            "readSize": num(self.read_size),
            "procInStagingBufferCount": num(self.proc_in_staging_buffer_count),
            "stagingCount": self.staging_count,
            "cpuRatio": num(self.cpu_ratio),
            "minMemoryLimit": num(self.min_memory_limit),
            "processBatchSize": num(self.process_batch_size),
            "recDataSize": num(self.rec_data_size),
            "outputFileMaxSlices": num(self.opts.output_file_max_slices_res),
            "outputFirstFileSlices": opt_num(self.opts.output_first_file_slices_res),
            "outputFileCount": self.opts.output_file_count_res,
            "memoryLimit": opt_num(self.opts.memory_limit),
            "sliceNums": self.slice_nums,
            "recoveryFiles": files,
            "openclDevices": ocl,
            "procCpu": self.proc_cpu.as_ref().map(|c| json!({
                "method": c.method,
                "input_batchsize": c.input_batchsize,
                "chunk_size": c.chunk_size,
                "slice_offset": c.slice_offset,
                "slice_size": c.slice_size,
            })).unwrap_or(serde_json::Value::Null),
            "procOcl": ocl,
            "displayNames": display_names,
        })
    }
}

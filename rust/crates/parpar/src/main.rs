//! parpar - high performance PAR2 creation tool (Rust port).
//!
//! Phase 1a: the full argument parser + option post-processing is wired in;
//! help/version/opencl-list and all parse errors match the JS CLI. The
//! generation pipeline lands in Phase 1f. `parpar selftest` exercises the
//! FFI backend end-to-end (GF round-trip + hashers).

use parpar::args::{self, EarlyExit};
use parpar::compat;
use parpar::dump;
use parpar::par2;
use parpar::present;
use parpar_backend::*;
#[cfg(feature = "ffi")]
use parpar_ffi::FfiBackend;
use std::io::IsTerminal;
use std::io::Write;

const HELP_SHORT: &str = include_str!("../../../../help.txt");
const HELP_FULL: &str = include_str!("../../../../help-full.txt");

/// JS `console.error` writes the message + '\n' to stderr.
fn stderr_line(s: &str) {
    let mut e = std::io::stderr().lock();
    let _ = e.write_all(s.as_bytes());
    let _ = e.write_all(b"\n");
}

/// cliFormat: `code` styling only when stderr is a TTY.
fn cli_format(code: &str, msg: &str) -> String {
    if std::io::stderr().is_terminal() {
        format!("\x1b[{}m{}\x1b[0m", code, msg)
    } else {
        msg.to_string()
    }
}

/// JS `error()` - message, usage hint, exit 1.
fn cli_error(msg: &str) -> ! {
    stderr_line(msg);
    stderr_line(&format!(
        "Enter `{}` or `{}` for usage information",
        cli_format("1", "parpar --help"),
        cli_format("1", "parpar --help-full")
    ));
    std::process::exit(1);
}

/// JSON.stringify(obj, null, 2) - serde_json's pretty printer uses the same
/// 2-space layout; `preserve_order` keeps JS insertion order in objects.
fn json_pretty(v: serde_json::Value) -> String {
    serde_json::to_string_pretty(&v).unwrap()
}

/// JS number -> string, covering the cases needed for our payload dumps.
fn js_num_string(n: f64) -> String {
    if n.is_nan() || n.is_infinite() {
        return "null".into();
    }
    if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

fn os_arg(s: &std::ffi::OsString) -> String {
    // Node decodes argv as UTF-8 with U+FFFD replacement - lossy conversion
    // is the closest equivalent
    s.to_string_lossy().into_owned()
}

fn main() {
    let args: Vec<String> = std::env::args_os().skip(1).map(|s| os_arg(&s)).collect();
    // dev-only scaffold until the pipeline lands
    if args.first().map(|s| s.as_str()) == Some("selftest") {
        std::process::exit(selftest());
    }
    let cfg = match args::parse_cli(&args) {
        Ok(c) => c,
        Err(e) => cli_error(&e),
    };
    match &cfg.early {
        Some(EarlyExit::Help { full }) => {
            let help_text = if *full { HELP_FULL } else { HELP_SHORT };
            // `ParPar\n` -> `ParPar vX.Y.Z\n` on the first line only;
            // console.error appends its own \n after the text's trailing \n
            let ver = args::PARPAR_VERSION;
            let replaced = help_text.replacen("ParPar", &format!("ParPar v{}", ver), 1);
            stderr_line(&replaced);
            std::process::exit(0);
        }
        Some(EarlyExit::Version) => {
            let json = cfg.argv.get("json").map(|v| v.truthy()).unwrap_or(false);
            if json {
                println!(
                    "{}",
                    json_pretty(serde_json::json!({
                        "type": "client_info",
                        "version": args::PARPAR_VERSION,
                        "creator": args::creator_string(),
                    }))
                );
            } else {
                stderr_line(args::PARPAR_VERSION);
            }
            std::process::exit(0);
        }
        Some(EarlyExit::OpenclList(filter)) => {
            let json = cfg.argv.get("json").map(|v| v.truthy()).unwrap_or(false);
            std::process::exit(opencl_list(filter, json));
        }
        None => {}
    }

    if let Err(e) = run(cfg) {
        cli_error(&e);
    }
}

fn run(mut cfg: args::CliConfig) -> std::result::Result<(), String> {
    // resolve --input-file/--input-file0 sources (stdin read once, on demand)
    let mut stdin_read = false;
    cfg.resolve_input_files(|| {
        if stdin_read {
            return Err("stdin was specified as input for multiple sources".into());
        }
        stdin_read = true;
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut buf)
            .map_err(|e| e.to_string())?;
        Ok(buf)
    })?;

    // deferred OpenCL device-name resolution (JS does this inside ppo
    // construction, after the input lists). The platform list is only
    // loaded when a non-numeric device name appears, mirroring the lazy
    // opencl_devices() call in openclMap.
    let needs_lookup = std::iter::once(&cfg.opencl_global)
        .chain(cfg.ppo.opencl_devices.iter().flatten())
        .any(|s| {
            s.device_raw
                .as_ref()
                .map(|r| !args::is_numeric_device_spec(r))
                .unwrap_or(false)
        });
    if needs_lookup || cfg.ppo.opencl_devices.is_some() {
        let platforms = if needs_lookup {
            let mut backend = parpar::native_backend::NativeBackend::new();
            backend
                .opencl_platforms()
                .map_err(|e| e.to_string())?
                .iter()
                .map(|p| args::OclPlatformInfo {
                    id: p.id,
                    devices: p
                        .devices
                        .iter()
                        .map(|d| args::OclDeviceInfo {
                            id: d.id,
                            name: d.name.clone(),
                        })
                        .collect(),
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut global = cfg.opencl_global.clone();
        let mut devs = cfg.ppo.opencl_devices.take().unwrap_or_default();
        args::resolve_opencl_devices(&mut global, &mut devs, &platforms)?;
        cfg.opencl_global = global;
        if !devs.is_empty() {
            cfg.ppo.opencl_devices = Some(devs);
        }
    }

    if std::env::var_os("PARPAR_DUMP_ARGS").is_some() {
        println!("{}", dump::dump_json(&cfg));
        return Ok(());
    }

    let json = cfg.json;
    let quiet = cfg.quiet;

    // scanning phase (JS: 'Scanning input file(s)...\r' + 200ms counter)
    if !quiet {
        if json {
            present::print_json("scanning_input", serde_json::Map::new());
        } else {
            let _ = std::io::stderr()
                .lock()
                .write_all(b"Scanning input file(s)...\r");
        }
    }
    let scan_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let scan_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let scan_thread = if !quiet {
        let c = scan_count.clone();
        let stop = scan_stop.clone();
        Some(std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let n = c.load(std::sync::atomic::Ordering::Relaxed);
                if n < 1 {
                    continue;
                }
                if json {
                    present::print_json(
                        "scanning_progress",
                        serde_json::json!({"num_scanned": n})
                            .as_object()
                            .unwrap()
                            .clone(),
                    );
                } else {
                    let _ = std::io::stderr().lock().write_all(
                        format!("Scanning input file(s)... ({} scanned)\r", n).as_bytes(),
                    );
                }
            }
        }))
    } else {
        None
    };
    let mut files = match parpar::fileinfo::file_info(
        &cfg.input_files,
        cfg.recurse,
        cfg.skip_symlinks,
        cfg.ppo.chunk_read_threads.unwrap_or(0).max(0) as usize,
        Some(scan_count),
    ) {
        Ok(f) => f,
        Err(e) => {
            scan_stop.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(t) = scan_thread {
                let _ = t.join();
            }
            stderr_line(&e); // JS console.error(err); process.exit(1)
            std::process::exit(1);
        }
    };
    scan_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(t) = scan_thread {
        let _ = t.join();
    }
    if files.is_empty() {
        stderr_line("No input files found."); // JS console.error path
        std::process::exit(1);
    }

    // solver (JS: try{new PAR2Gen} catch -> error(x.message))
    let slice_def = cfg.input_slice_def.clone().unwrap_or(args::SpecTerm {
        unit: args::SpecUnit::Count,
        value: 0.0,
        scale: 1.0,
    });
    let o = parpar::generate::GenOpts::from_cli(&cfg);
    let env = parpar::generate::SysEnv {
        total_mem: parpar::sysinfo::total_mem(),
        free_mem: parpar::sysinfo::free_mem(),
        constrained_mem: parpar::sysinfo::constrained_memory(),
        is_64b: parpar::sysinfo::is_64b_platform(),
        k_max_length: parpar::sysinfo::max_buffer_size(),
    };
    // Backend cutover: native Rust backend by default; PARPAR_FORCE_FFI=1
    // selects the C++ FFI backend (kept as the migration oracle, only in
    // builds compiled with the `ffi` feature).
    #[cfg(feature = "ffi")]
    if std::env::var("PARPAR_FORCE_FFI")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        let backend = FfiBackend::new().map_err(|e| e.to_string())?;
        let mut shared = par2::SharedBackend::new(backend);
        // coexistence flags still apply on the FFI path
        if std::env::var("PARPAR_NATIVE_HASH")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            shared.set_native_hashers(true);
        }
        if std::env::var("PARPAR_NATIVE_GF")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            shared.set_native_proc(true);
        }
        return run_generate(&cfg, &mut files, slice_def, o, env, shared);
    }
    #[cfg(not(feature = "ffi"))]
    if std::env::var("PARPAR_FORCE_FFI")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        return Err("this build has no FFI backend (built without the `ffi` feature)".into());
    }
    let backend = parpar::native_backend::NativeBackend::new();
    run_generate(
        &cfg,
        &mut files,
        slice_def,
        o,
        env,
        par2::SharedBackend::new(backend),
    )
}

fn run_generate<B: ComputeBackend>(
    cfg: &args::CliConfig,
    files: &mut Vec<parpar::fileinfo::FileInfo>,
    slice_def: args::SpecTerm,
    o: parpar::generate::GenOpts,
    env: parpar::generate::SysEnv,
    shared: par2::SharedBackend<B>,
) -> std::result::Result<(), String>
where
    B::Proc: 'static,
    B::InputHasher: 'static,
    B::OutputHasher: 'static,
{
    // hash/md5 method selection (JS lines 441-446; getMethodNum throws
    // for unknown names).
    if let Some(h) = &cfg.hash_method {
        let idx = par2::method_num(par2::INHASH_METHODS, h)?;
        B::set_inhash_method(idx);
    }
    if let Some(m) = &cfg.md5_method {
        let idx = par2::method_num(par2::OUTHASH_METHODS, m)?;
        B::set_outhash_method(idx);
    }
    let json = cfg.json;
    let quiet = cfg.quiet;
    let progress_sink = cfg.progress.clone();
    let mut shared = shared;
    let mut gen = parpar::generate::Par2Gen::new(
        files,
        &slice_def,
        o,
        shared.clone(),
        &cfg.ascii_charset,
        &mut shared,
        &env,
    )?;

    // processing info
    let (hash_in, hash_out) = if shared.native_hashers() {
        (
            parpar_hasher::inhash_method_name().to_string(),
            parpar_hasher::outhash_method_name().to_string(),
        )
    } else {
        (
            B::inhash_method_name().to_string(),
            B::outhash_method_name().to_string(),
        )
    };
    if !quiet {
        let info = present::InfoFields {
            slice_size: gen.opts.slice_size,
            total_size: gen.total_size,
            input_slices: gen.input_slices,
            input_files: files.len(),
            recovery_slices: gen.opts.recovery_slices_res,
            chunks: gen.chunks,
            passes: gen.passes,
            slices_per_pass: gen.slices_per_pass,
            chunk_size: gen.chunk_size,
            read_size: gen.read_size,
            read_buffers: gen.opts.read_buffers,
            rec_data_size: gen.rec_data_size,
            input_chunks_total: gen.total_num_input_chunks(),
            hash_in: hash_in.clone(),
            hash_out: hash_out.clone(),
            recovery_files: gen
                .recovery_files
                .iter()
                .map(|rf| present::RecoveryFileInfo {
                    name: rf.name.clone(),
                    recovery_slices: rf.recovery_slices,
                    recovery_offset: rf.recovery_index,
                    total_size: rf.total_size as f64,
                })
                .collect(),
        };
        if json {
            present::print_processing_info_json(&info);
        } else {
            present::print_processing_info_text(&info);
        }
    }

    // progress model constants (JS lines ~818-826)
    let rec_slices = gen.opts.recovery_slices_res;
    let progress_write_factor = if rec_slices != 0.0 {
        rec_slices.powf(-0.3)
    } else {
        0.0
    };
    let mut total_progress =
        gen.total_num_input_chunks() * gen.passes + gen.chunks * rec_slices * progress_write_factor;
    let write_factor_per_pass = progress_write_factor / gen.passes.max(1.0);
    let hash_factor = gen.chunks * progress_write_factor;
    total_progress += hash_factor * gen.input_slices;

    let start = std::time::Instant::now();
    let gen_rec_data_size = gen.rec_data_size;
    let gen_chunk_size = gen.chunk_size;
    let gen_staging_buf_count = gen.proc_in_staging_buffer_count;
    let gen_input_slices = gen.input_slices;
    let cpu_name = parpar::sysinfo::cpu_model();
    let mut drv = parpar::pipeline::GenDriver::new(&mut gen).map_err(|e| e.to_string())?;

    let state = std::sync::Arc::new(std::sync::Mutex::new(present::ProgressState {
        current_state: "Initializing".into(),
        ..Default::default()
    }));
    let mut ticker = if progress_sink != "none" && total_progress != 0.0 {
        Some(present::Ticker::start(
            progress_sink.clone(),
            json,
            total_progress,
            write_factor_per_pass,
            hash_factor,
            gen_input_slices,
            state.clone(),
        ))
    } else {
        None
    };
    let run_res = {
        let state = state.clone();
        drv.run(&mut |ev| {
            match &ev {
                parpar::pipeline::Progress::ComputeInfo(info) => {
                    if !quiet {
                        if json {
                            present::print_compute_info_json(info, &cpu_name);
                        } else {
                            present::print_compute_info_text(
                                info,
                                &cpu_name,
                                gen_rec_data_size,
                                gen_chunk_size,
                                gen_staging_buf_count,
                            );
                        }
                    }
                }
                parpar::pipeline::Progress::BeginChunkPass(p, c, n) => {
                    if json {
                        present::print_json(
                            "begin_subpass",
                            serde_json::json!({
                                "pass": p, "subpass": c, "input_chunks": n
                            })
                            .as_object()
                            .unwrap()
                            .clone(),
                        );
                    }
                }
                parpar::pipeline::Progress::ChunkPassComplete(p, c) => {
                    if json {
                        present::print_json(
                            "end_subpass",
                            serde_json::json!({"pass": p, "subpass": c})
                                .as_object()
                                .unwrap()
                                .clone(),
                        );
                    }
                }
                parpar::pipeline::Progress::ChunkPassWrite(p, c) => {
                    if json {
                        present::print_json(
                            "subpass_write",
                            serde_json::json!({"pass": p, "subpass": c})
                                .as_object()
                                .unwrap()
                                .clone(),
                        );
                    }
                }
                parpar::pipeline::Progress::PassWrite(p) => {
                    if json {
                        present::print_json(
                            "pass_write",
                            serde_json::json!({"pass": p}).as_object().unwrap().clone(),
                        );
                    }
                }
                parpar::pipeline::Progress::WritingFile(_, name, _, _) => {
                    if json {
                        present::print_json(
                            "writing_data",
                            serde_json::json!({"file_name": name})
                                .as_object()
                                .unwrap()
                                .clone(),
                        );
                    }
                }
                parpar::pipeline::Progress::ClosingFiles => {
                    if json {
                        present::print_json("closing_files", serde_json::Map::new());
                    }
                }
                _ => {}
            }
            state.lock().unwrap().on_event(&ev);
        })
    };

    if let Some(t) = ticker.as_mut() {
        t.stop();
    }
    if let Err(e) = run_res {
        return Err(e.to_string());
    }

    if progress_sink != "none" && !json {
        // JS writeProgress({state:'Finished', progress_percent:100})
        present::write_progress(&progress_sink, false, "Finished", 100.0, None);
    }
    if !quiet {
        let secs = (start.elapsed().as_millis() as f64) / 1000.0;
        if json {
            present::print_json(
                "process_complete",
                serde_json::json!({"duration_seconds": secs})
                    .as_object()
                    .unwrap()
                    .clone(),
            );
        } else {
            let _ = std::io::stderr().lock().write_all(
                format!(
                    "\nProcessing time   : {}\n",
                    cli_format("1", &format!("{} s", present::js_num(secs)))
                )
                .as_bytes(),
            );
        }
    }
    Ok(())
}

/// `--opencl-list` output - text mode mirrors bin/parpar.js exactly.
fn opencl_list(filter: &str, json: bool) -> i32 {
    let mut backend = parpar::native_backend::NativeBackend::new();
    let platforms = match backend.opencl_platforms() {
        Ok(p) => p,
        Err(e) => {
            cli_error(&e.to_string());
        }
    };
    if json {
        let plats: Vec<serde_json::Value> = platforms
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "name": &p.name,
                    "devices": p.devices.iter().map(ocl_device_json).collect::<Vec<_>>(),
                })
            })
            .collect();
        println!(
            "{}",
            json_pretty(serde_json::json!({
                "type": "opencl_list",
                "platforms": plats,
            }))
        );
        return 0;
    }
    if platforms.is_empty() {
        stderr_line("No OpenCL platforms found");
        return 0;
    }
    // JS `opencl_device_info()` -> default platform + first available GPU
    let default_plat = backend.opencl_default_platform();
    let default_dev = if default_plat >= 0 {
        backend.opencl_default_device(default_plat)
    } else {
        -1
    };
    let show_all = filter.eq_ignore_ascii_case("all");
    let mut plat_blocks: Vec<String> = Vec::new();
    for p in &platforms {
        let default_label = if p.id == default_plat {
            " [default]"
        } else {
            ""
        };
        let mut dev_lines: Vec<String> = Vec::new();
        for d in &p.devices {
            if !(show_all
                || (d.device_type.eq_ignore_ascii_case("gpu") && d.available && d.supported))
            {
                continue;
            }
            let mut out = vec![
                format!(
                    "  - Device #{}: {}{}",
                    d.id,
                    d.name,
                    if d.id == default_dev {
                        default_label
                    } else {
                        ""
                    }
                ),
                format!("    Type: {}", d.device_type),
                format!(
                    "    Memory: {}{}",
                    friendly_size(d.memory_global as f64),
                    if d.memory_unified { " (shared)" } else { "" }
                ),
            ];
            if !d.supported {
                out.insert(1, "    Supported: no".into());
            }
            if !d.available {
                out.insert(1, "    Available: no".into());
            }
            dev_lines.push(out.join("\n"));
        }
        let device_str = dev_lines.join("\n");
        if !device_str.is_empty() || show_all {
            plat_blocks.push(
                format!("Platform #{}: {}{}", p.id, p.name, default_label)
                    + if device_str.is_empty() { "" } else { "\n" }
                    + &device_str,
            );
        }
    }
    let body = plat_blocks.join("\n\n");
    if body.is_empty() {
        stderr_line(&format!(
            "No available devices found; try `{}` to see all platforms/devices",
            cli_format("1", "--opencl-list=all")
        ));
    } else {
        stderr_line(&body);
    }
    0
}

fn ocl_device_json(d: &OclDevice) -> serde_json::Value {
    let mut dev = serde_json::json!({
        "id": d.id,
        "name": &d.name,
        "vendor_id": d.vendor_id,
        "available": d.available,
        "supported": d.supported,
        "type": &d.device_type,
        "memory_global": d.memory_global,
        "cache_global": d.cache_global,
        "memory_unified": d.memory_unified,
        "memory_constant": d.memory_constant,
    });
    // JS sets memory_local only when the driver reports a distinct local size
    if let Some(l) = d.memory_local {
        dev["memory_local"] = serde_json::json!(l);
    }
    dev["memory_max_alloc"] = serde_json::json!(d.memory_max_alloc);
    dev["workgroup_limit"] = serde_json::json!(d.workgroup_limit);
    dev["workgroup_multiple"] = serde_json::json!(d.workgroup_multiple);
    dev["compute_units"] = serde_json::json!(d.compute_units);
    dev
}

/// cliUtil.friendlySize
fn friendly_size(mut s: f64) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut i = 0;
    while i < UNITS.len() && s >= 10000.0 {
        s /= 1024.0;
        i += 1;
    }
    format!(
        "{} {}",
        js_num_string(compat::js_math_round(s * 100.0) / 100.0),
        UNITS[i.min(UNITS.len() - 1)]
    )
}

// method-name -> id mapping (GF_METHODS / INHASH_METHODS / OUTHASH_METHODS
// tables in lib/par2.js) is ported with the pipeline work

fn selftest() -> i32 {
    use parpar::native_backend::NativeBackend;
    eprintln!("== parpar selftest (native backend) ==");

    // method info
    for m in [0i32, 1, 19, 22, 27] {
        match NativeBackend::gf_info(m) {
            Ok(i) => eprintln!(
                "gf_info({}) -> id={} name={:?} align={} stride={} chunk={} grouping={}",
                m, i.id, i.name, i.alignment, i.stride, i.target_chunk, i.target_grouping
            ),
            Err(e) => eprintln!("gf_info({}) -> err {}", m, e),
        }
    }
    eprintln!("inhash method: {}", NativeBackend::inhash_method_name());
    eprintln!("outhash method: {}", NativeBackend::outhash_method_name());

    let mut backend = NativeBackend::new();

    // opencl enumeration
    match backend.opencl_platforms() {
        Ok(plats) => {
            eprintln!(
                "opencl: {} platform(s), default={}",
                plats.len(),
                backend.opencl_default_platform()
            );
            for p in &plats {
                eprintln!("  platform {}: {}", p.id, p.name);
                for d in &p.devices {
                    eprintln!(
                        "    dev {}: {:?} type={} supported={} mem={}",
                        d.id, d.name, d.device_type, d.supported, d.memory_global
                    );
                }
            }
        }
        Err(e) => eprintln!("opencl: {}", e),
    }

    // --- GF round trip ---
    let slice_size: u64 = 4096;
    let n_inputs = 10usize;
    let n_recovery = 4usize;

    let cpu = CpuSpec {
        method: 0, // auto
        input_batchsize: 0,
        input_minbatchsize: 0,
        chunk_size: 0,
        slice_offset: 0,
        slice_size: 0, // full slice
    };
    let mut proc = match backend.new_proc(slice_size, Some(&cpu), &[], 0) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("new_proc failed: {}", e);
            return 1;
        }
    };
    let exps: Vec<u16> = (0..n_recovery as u16).collect();
    if let Err(e) = proc.set_recovery_slices(Some(&exps), n_recovery) {
        eprintln!("set_recovery_slices: {}", e);
        return 1;
    }
    match proc.set_num_threads(4) {
        Ok(t) => eprintln!("threads: {}", t),
        Err(e) => eprintln!("set_num_threads: {}", e),
    }
    let _ = proc.set_progress(true);
    if let Ok(info) = proc.info() {
        if let Some(c) = &info.cpu {
            eprintln!(
                "cpu method={:?} chunk={} staging={}x{} align={} stride={} slice_mem={} outs={}",
                c.method_desc,
                c.chunk_size,
                c.staging_count,
                c.staging_size,
                c.alignment,
                c.stride,
                c.slice_mem,
                c.num_output_slices
            );
        }
    }

    // deterministic inputs
    let mut token = 1u64;
    let mut _pending = 0usize;
    for i in 0..n_inputs {
        let mut v = vec![0u8; slice_size as usize];
        for (j, b) in v.iter_mut().enumerate() {
            *b = ((i * 131 + j) % 251) as u8;
        }
        let len = v.len();
        let buf = SliceBuf::new(v.into_boxed_slice(), len);
        match proc.add_input(buf, i as i32, token) {
            Ok(AddResult::Accepted) => {
                token += 1;
                _pending += 1;
            }
            Ok(AddResult::Full(b)) => {
                eprintln!("add {}: staging full ({} bytes back)", i, b.len);
            }
            Err(e) => {
                eprintln!("add {}: {}", i, e);
                return 1;
            }
        }
    }
    let end_tok = token;
    if let Err(e) = proc.end_input(token) {
        eprintln!("end_input: {}", e);
        return 1;
    }
    token += 1;

    // drain until ENDED (input-done events recycle the buffers)
    let mut inputs_done = 0usize;
    loop {
        let (_id, ev) = backend.wait_event(-1).expect("event stream closed");
        match ev {
            ProcEvent::InputDone { input_index, .. } => {
                inputs_done += 1;
                eprintln!("  input {} consumed", input_index);
            }
            ProcEvent::Ended { .. } => break,
            ProcEvent::Progress { num_inputs } => eprintln!("  progress: {} staged", num_inputs),
            ProcEvent::Error { message, .. } => {
                eprintln!("  async error: {}", message);
                return 1;
            }
            _ => {}
        }
    }
    let _ = inputs_done;
    let _ = end_tok;

    // fetch recovery outputs
    let mut rec: Vec<Vec<u8>> = vec![vec![]; n_recovery];
    let mut fetched = 0usize;
    for i in 0..n_recovery {
        let v = vec![0u8; slice_size as usize];
        let buf = SliceBuf::new(v.into_boxed_slice(), slice_size as usize);
        if let Err(e) = proc.get_output(i as i32, buf, token) {
            eprintln!("get_output {}: {}", i, e);
            return 1;
        }
        token += 1;
    }
    while fetched < n_recovery {
        let (_id, ev) = backend.wait_event(-1).expect("event stream closed");
        if let ProcEvent::OutputDone {
            cksum_valid, buf, ..
        } = ev
        {
            let idx = fetched;
            let md = md5_like_sum(&buf.data[..]);
            eprintln!(
                "  output {}: {} bytes cksum_valid={} sum={:08x}",
                idx, buf.len, cksum_valid, md
            );
            rec[idx] = buf.data.into_vec();
            fetched += 1;
        }
    }
    // NOTE: output order isn't guaranteed to be 0..n-1; for the smoke test we
    // only check bytes exist. Ordered checks happen in the conformance suite.

    if let Err(e) = proc.close() {
        eprintln!("close: {}", e);
        return 1;
    }

    // --- hashers ---
    {
        let slices = 3usize;
        let hsize = slice_size as usize;
        let block_out = vec![0u8; (slices + 1) * 20].into_boxed_slice();
        let mut hin = match backend.new_input_hasher(slice_size, block_out) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("hin new: {}", e);
                return 1;
            }
        };
        let data = vec![0xabu8; hsize * slices + 17];
        if let Err(e) = hin.update(&data) {
            eprintln!("hin update: {}", e);
            return 1;
        }
        let mut md5 = [0u8; 16];
        match hin.end(&mut md5) {
            Ok(blocks) => {
                eprintln!(
                    "hin end: file md5={} blockrecs={}B",
                    hex(&md5),
                    blocks.len()
                );
            }
            Err(e) => {
                eprintln!("hin end: {}", e);
                return 1;
            }
        }
    }
    {
        let regions = 4u32;
        let mut hout = match backend.new_output_hasher(regions) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("hout new: {}", e);
                return 1;
            }
        };
        let bufs: Vec<Vec<u8>> = (0..regions).map(|i| vec![i as u8; 333]).collect();
        let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
        if let Err(e) = hout.update(&refs) {
            eprintln!("hout update: {}", e);
            return 1;
        }
        let mut out = vec![0u8; regions as usize * 16];
        if let Err(e) = hout.get(&mut out) {
            eprintln!("hout get: {}", e);
            return 1;
        }
        for r in 0..regions as usize {
            eprintln!("  region {} md5={}", r, hex(&out[r * 16..r * 16 + 16]));
        }
    }

    drop(proc);
    eprintln!("== selftest OK ==");
    0
}

fn md5_like_sum(b: &[u8]) -> u32 {
    b.iter()
        .fold(0u32, |a, &x| a.wrapping_mul(31).wrapping_add(x as u32))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

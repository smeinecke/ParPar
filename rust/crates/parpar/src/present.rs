//! CLI presentation: `bin/parpar.js`'s scanning/processing info text,
//! progress ticker, JSON event output and compute-info block.

use std::io::IsTerminal;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::compat;
use crate::pipeline::Progress;

/// f64 -> JSON value using JS number semantics (integral -> int).
pub fn jnum(n: f64) -> serde_json::Value {
    if n.fract() == 0.0 && n.abs() < 9.0e15 {
        serde_json::json!(n as i64)
    } else {
        serde_json::json!(n)
    }
}

/// `print_json` - `{type, ...obj}` pretty printed to stdout.
pub fn print_json(t: &str, obj: serde_json::Map<String, serde_json::Value>) {
    let mut o = serde_json::Map::new();
    o.insert("type".into(), serde_json::json!(t));
    for (k, v) in obj {
        o.insert(k, v);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(o)).unwrap()
    );
}

/// `cliFormat(code, msg)` - ANSI only when stderr is a TTY.
pub fn cli_format(code: &str, msg: &str) -> String {
    if std::io::stderr().is_terminal() {
        format!("\x1b[{}m{}\x1b[0m", code, msg)
    } else {
        msg.to_string()
    }
}

/// `pluralDisp(n, unit, suffix)`.
fn plural_disp(n: f64, unit: &str, suffix: &str) -> String {
    if n == 1.0 {
        format!("{} {}", cli_format("1", "1"), unit)
    } else {
        format!("{} {}{}", cli_format("1", &js_num(n)), unit, suffix)
    }
}

fn size_disp(val: f64) -> String {
    cli_format("1", &friendly_size(val))
}

/// `friendlySize` from cliUtil.
pub fn friendly_size(mut s: f64) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut i = 0;
    while i < UNITS.len() && s >= 10000.0 {
        s /= 1024.0;
        i += 1;
    }
    format!(
        "{} {}",
        js_num(compat::js_math_round(s * 100.0) / 100.0),
        // i can reach UNITS.len() on huge/inf inputs (JS prints
        // 'undefined'); clamp rather than index out of bounds
        UNITS[i.min(UNITS.len() - 1)]
    )
}

/// JS number -> display string (integer text when integral).
pub fn js_num(n: f64) -> String {
    // `n as i64` saturates for |n| > i64::MAX (integral f64 up to 1e21
    // still prints full digits in JS, and Rust's {} does the same)
    if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

fn stderr_write(s: &str) {
    let _ = std::io::stderr().lock().write_all(s.as_bytes());
}

/// The `processing_info` text block printed before generation starts.
pub struct InfoFields {
    pub slice_size: f64,
    pub total_size: f64,
    pub input_slices: f64,
    pub input_files: usize,
    pub recovery_slices: f64,
    pub chunks: f64,
    pub passes: f64,
    pub slices_per_pass: f64,
    pub chunk_size: f64,
    pub read_size: f64,
    pub read_buffers: f64,
    pub rec_data_size: f64,
    pub input_chunks_total: f64,
    pub hash_in: String,
    pub hash_out: String,
    pub recovery_files: Vec<RecoveryFileInfo>,
}

pub struct RecoveryFileInfo {
    pub name: String,
    pub recovery_slices: i64,
    pub recovery_offset: i64,
    pub total_size: f64,
}

/// `processing_info` JSON object.
pub fn print_processing_info_json(f: &InfoFields) {
    let mut o = serde_json::Map::new();
    o.insert("input_size".into(), jnum(f.total_size));
    o.insert("input_slices".into(), jnum(f.input_slices));
    o.insert("input_chunks".into(), jnum(f.input_chunks_total));
    o.insert("input_files".into(), jnum(f.input_files as f64));
    o.insert("recovery_slices".into(), jnum(f.recovery_slices));
    o.insert("slice_size".into(), jnum(f.slice_size));
    o.insert("passes".into(), jnum(f.passes));
    o.insert("subpasses".into(), jnum(f.chunks));
    o.insert("pass_slices".into(), jnum(f.slices_per_pass));
    o.insert("subpass_chunk_size".into(), jnum(f.chunk_size));
    o.insert("read_buffer_size".into(), jnum(f.read_size));
    o.insert("read_buffers".into(), jnum(f.read_buffers));
    o.insert("recovery_buffers".into(), jnum(f.rec_data_size));
    // JS rf.recoveryOffset is always undefined -> the key is omitted
    let rfs: Vec<serde_json::Value> = f
        .recovery_files
        .iter()
        .map(|rf| {
            serde_json::json!({
                "name": rf.name,
                "recovery_slices": rf.recovery_slices,
                "size": jnum(rf.total_size),
            })
        })
        .collect();
    o.insert("recovery_files".into(), serde_json::Value::Array(rfs));
    o.insert("hash_input_method".into(), serde_json::json!(f.hash_in));
    o.insert("hash_recovery_method".into(), serde_json::json!(f.hash_out));
    print_json("processing_info", o);
}

/// The text info block (JS lines ~788-813).
pub fn print_processing_info_text(f: &InfoFields) {
    if f.slice_size > 1024.0 * 1048576.0 {
        stderr_write(&format!(
            "{}: selected slice size ({}) is larger than 1GB, which is beyond what a number of PAR2 clients support. Consider increasing the number of slices or reducing the slice size so that it is under 1GB\n",
            cli_format("33", "Warning"),
            friendly_size(f.slice_size)
        ));
    } else if f.slice_size > 100.0 * 1e6 && f.total_size <= 32768.0 * 100.0 * 1e6 {
        stderr_write(&format!(
            "{}: selected slice size ({}) may be too large to be compatible with QuickPar\n",
            cli_format("33", "Warning"),
            friendly_size(f.slice_size)
        ));
    }
    stderr_write(&format!(
        "Input data        : {} ({} from {})\n",
        size_disp(f.total_size),
        plural_disp(f.input_slices, "slice", "s"),
        plural_disp(f.input_files as f64, "file", "s")
    ));
    if f.recovery_slices != 0.0 {
        stderr_write(&format!(
            "Recovery data     : {} ({})\n",
            size_disp(f.recovery_slices * f.slice_size),
            plural_disp(
                f.recovery_slices,
                &format!("* {} slice", size_disp(f.slice_size)),
                "s"
            )
        ));
        stderr_write(&format!(
            "Input pass(es)    : {}, processing {} per pass\n",
            cli_format("1", &js_num(f.chunks * f.passes)),
            plural_disp(
                f.slices_per_pass,
                &format!("* {} chunk", size_disp(f.chunk_size)),
                "s"
            )
        ));
    }
    stderr_write(&format!(
        "Read buffer size  : {} * max {}\n",
        size_disp(f.read_size),
        plural_disp(f.read_buffers, "buffer", "s")
    ));
    stderr_write(&format!(
        "Hash method       : {} (input){}\n",
        cli_format("1", &f.hash_in),
        if f.recovery_slices != 0.0 {
            format!(", {} (recovery)", cli_format("1", &f.hash_out))
        } else {
            String::new()
        }
    ));
}

/// `compute_info` (JSON) - printed on first 'begin_chunk_pass'.
pub fn print_compute_info_json(info: &parpar_backend::GfInfo, cpu_name: &str) {
    let cpu = match &info.cpu {
        Some(c) => serde_json::json!({
            "device_name": cpu_name.trim(),
            "threads": c.threads,
            "method": c.method_desc,
            "tile_size": c.chunk_size,
            "batch_slices": c.staging_size,
            "batches": c.staging_count,
            "slice_size": c.slice_mem,
            "recovery_slices": c.num_output_slices,
        }),
        None => serde_json::json!(null),
    };
    let ocl: Vec<serde_json::Value> = info
        .ocl
        .iter()
        .map(|d| {
            serde_json::json!({
                "device_name": d.device_name.trim(),
                "method": d.method_desc,
                "output_grouping": d.output_chunks,
                "tile_size": d.chunk_size,
                "batch_slices": d.staging_size,
                "batches": d.staging_count,
                "slice_size": d.slice_mem,
                "recovery_slices": d.num_output_slices,
            })
        })
        .collect();
    print_json(
        "compute_info",
        serde_json::json!({"cpu": cpu, "opencl": ocl})
            .as_object()
            .unwrap()
            .clone(),
    );
}

/// compute_info text block (JS lines ~898-917).
pub fn print_compute_info_text(
    info: &parpar_backend::GfInfo,
    cpu_name: &str,
    rec_data_size: f64,
    chunk_size: f64,
    proc_in_staging_buffer_count: f64,
) {
    if let Some(c) = &info.cpu {
        if c.threads != 0 {
            let mut name = cpu_name.trim().to_string();
            if name == "unknown" {
                name = "CPU".into();
            }
            stderr_write(&format!("\n{}\n", cli_format("4", &name)));
            stderr_write(&format!(
                "  Multiply method : {} with {} loop tiling, {}\n",
                cli_format("1", &c.method_desc),
                size_disp(c.chunk_size as f64),
                plural_disp(c.threads as f64, "thread", "s")
            ));
            stderr_write(&format!(
                "  Input batching  : {}, {}\n",
                plural_disp(c.staging_size as f64, "chunk", "s"),
                plural_disp(c.staging_count as f64, "batch", "es")
            ));
            let trans_mem =
                (rec_data_size * chunk_size).max(c.slice_mem as f64 * proc_in_staging_buffer_count);
            stderr_write(&format!(
                "  Memory Usage    : {} ({} + {} transfer buffer)\n",
                size_disp(c.slice_mem as f64 * c.num_output_slices as f64 + trans_mem),
                plural_disp(
                    c.num_output_slices as f64,
                    &format!("* {} chunk", size_disp(c.slice_mem as f64)),
                    "s"
                ),
                size_disp(trans_mem)
            ));
        } else {
            stderr_write(&format!(
                "Transfer Buffer   : {}\n",
                size_disp(rec_data_size * chunk_size)
            ));
        }
    } else {
        stderr_write(&format!(
            "Transfer Buffer   : {}\n",
            size_disp(rec_data_size * chunk_size)
        ));
    }
    for d in &info.ocl {
        stderr_write(&format!("\n{}\n", cli_format("4", d.device_name.trim())));
        stderr_write(&format!(
            "  Multiply method : {}, split into {} * {} workgroups\n",
            cli_format("1", &d.method_desc),
            cli_format("1", &js_num(d.output_chunks as f64)),
            size_disp(d.chunk_size as f64)
        ));
        stderr_write(&format!(
            "  Input batching  : {}, {}\n",
            plural_disp(d.staging_size as f64, "chunk", "s"),
            plural_disp(d.staging_count as f64, "batch", "es")
        ));
        stderr_write(&format!(
            "  Memory Usage    : {} ({})\n",
            size_disp(d.slice_mem as f64 * d.num_output_slices as f64),
            plural_disp(
                d.num_output_slices as f64,
                &format!("* {} chunk", size_disp(d.slice_mem as f64)),
                "s"
            )
        ));
    }
    stderr_write("\n");
}

/// Progress state updated by generation events (JS g.run callback's
/// counters) and sampled by the 200ms ticker.
#[derive(Default)]
pub struct ProgressState {
    pub current_state: String,
    pub current_slice: f64,
    pub current_slice_frac: f64,
    pub cur_pass_slice: u64,
    pub rec_slices_written: f64,
    pub prg_last_rec_file_slices: f64,
    pub rec_slices_written_frac: f64,
    /// last WritingFile's (recoverySlices, totalSize) for the pos frac
    pub cur_rf_slices: f64,
    pub cur_rf_total: f64,
    pub pass: u32,
    pub subpass: u32,
    pub info_shown: bool,
}

/// `Progress` event -> state counters (mirrors the JS callback).
impl ProgressState {
    pub fn on_event(&mut self, ev: &Progress) {
        match ev {
            Progress::BeginChunkPass(p, c, _) => {
                self.current_state = "Calculating".into();
                self.pass = *p;
                self.subpass = *c;
            }
            Progress::BeginPass(p) => {
                self.pass = *p;
            }
            Progress::ProcessingSlice(_, _) => {
                self.current_slice += 1.0;
                self.cur_pass_slice += 1;
                self.current_slice_frac = 0.0;
            }
            Progress::ProcessingSlicePart(frac) => {
                self.current_slice_frac = *frac;
            }
            Progress::ChunkPassWrite(_, _) | Progress::PassWrite(_) => {
                self.current_state = "Writing".into()
            }
            Progress::WritingFile(_, _, rsl, ts) => {
                self.rec_slices_written += self.prg_last_rec_file_slices;
                self.prg_last_rec_file_slices = *rsl as f64;
                self.cur_rf_slices = *rsl as f64;
                self.cur_rf_total = *ts as f64;
                self.rec_slices_written_frac = 0.0;
            }
            Progress::WritingFilePos(_, pos) => {
                self.rec_slices_written_frac = if self.cur_rf_total != 0.0 {
                    (*pos as f64 / self.cur_rf_total) * self.cur_rf_slices
                } else {
                    0.0
                };
            }
            Progress::ChunkPassComplete(_, _) => {
                self.rec_slices_written += self.prg_last_rec_file_slices;
                self.prg_last_rec_file_slices = 0.0;
                self.rec_slices_written_frac = 0.0;
                self.cur_pass_slice = 0;
            }
            Progress::ClosingFiles => self.current_state = "Finalizing".into(),
            _ => {}
        }
    }
}

/// Shared progress-ticker state.
pub struct Ticker {
    pub state: Arc<Mutex<ProgressState>>,
    pub stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Ticker {
    /// Start the 200ms ticker; `total_progress`/`hash_factor`/
    /// `write_factor_per_pass`/`input_slices` are the JS formula terms.
    /// `sink` is 'stderr'|'stdout'|'none' (JS --progress).
    pub fn start(
        sink: String,
        json: bool,
        total_progress: f64,
        write_factor_per_pass: f64,
        hash_factor: f64,
        input_slices: f64,
        state: Arc<Mutex<ProgressState>>,
    ) -> Ticker {
        let stop = Arc::new(AtomicBool::new(false));
        let st2 = state.clone();
        let stop2 = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                let s = st2.lock().unwrap();
                let mut current_progress = s.current_slice
                    + s.current_slice_frac
                    + (s.rec_slices_written + s.rec_slices_written_frac) * write_factor_per_pass;
                if s.pass == 0 && s.subpass == 0 {
                    current_progress += hash_factor * (s.current_slice + s.current_slice_frac);
                } else {
                    current_progress += hash_factor * input_slices;
                }
                let perc = (current_progress / total_progress * 10000.0).floor() / 100.0;
                write_progress(
                    &sink,
                    json,
                    &s.current_state,
                    perc.min(99.99),
                    Some((s.pass, s.subpass, s.cur_pass_slice)),
                );
                drop(s);
            }
        });
        Ticker {
            state,
            stop,
            handle: Some(handle),
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// `writeProgress` - JSON event or 'State: NN.NN%\r'.
pub fn write_progress(
    sink: &str,
    json: bool,
    state: &str,
    percent: f64,
    extra: Option<(u32, u32, u64)>,
) {
    if json {
        let mut o = serde_json::Map::new();
        o.insert("state".into(), serde_json::json!(state));
        o.insert("progress_percent".into(), jnum(percent));
        if let Some((pass, subpass, slice)) = extra {
            o.insert("pass".into(), serde_json::json!(pass));
            o.insert("subpass".into(), serde_json::json!(subpass));
            o.insert("slice".into(), serde_json::json!(slice));
        }
        print_json("progress", o);
        return;
    }
    if sink != "stdout" && sink != "stderr" {
        return;
    }
    // JS: parts = pct.toLocaleString().match(/^([0-9]+)([.,][0-9]+)?$/);
    // parts[1]=lpad(int,3,' '), parts[2]=rpad(frac||'.',3,'0'),
    // state=rpad(state,18,' ')
    let pct_str = js_num(percent);
    let (int_part, frac_part) = match pct_str.split_once('.') {
        Some((i, f)) => (i.to_string(), format!(".{}", f)),
        None => (pct_str, ".".to_string()),
    };
    let int_part = format!("{:>3}", int_part);
    let frac_part = format!(
        "{:<3}",
        frac_part
            .replace('.', ".")
            .chars()
            .chain(std::iter::repeat('0'))
            .take(3)
            .collect::<String>()
    );
    let state = format!("{:<18}", state);
    let pct = format!("{}{}", int_part, frac_part);
    let is_tty = match sink {
        "stdout" => std::io::stdout().is_terminal(),
        _ => std::io::stderr().is_terminal(),
    };
    let line = if is_tty {
        format!("{}: \x1b[1m{}%\x1b[0m\x1b[0G", state, pct)
    } else {
        format!("{}: {}%\r", state, pct)
    };
    if sink == "stdout" {
        let mut h = std::io::stdout().lock();
        let _ = h.write_all(line.as_bytes());
        let _ = h.flush();
        return;
    }
    let mut h = std::io::stderr().lock();
    let _ = h.write_all(line.as_bytes());
    let _ = h.flush();
}

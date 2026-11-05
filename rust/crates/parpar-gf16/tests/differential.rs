#![cfg(target_env = "gnu")] // FFI oracle only links on gnu targets

use parpar_backend::{AddResult, ComputeBackend, CpuSpec, GfProc, ProcEvent, SliceBuf};
use parpar_ffi::FfiBackend;
use parpar_gf16::{EventQueues, NativeGfProc};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

fn cpu(batchsize: i32, method: i32) -> CpuSpec {
    CpuSpec {
        method,
        input_batchsize: batchsize as u32,
        input_minbatchsize: 1,
        chunk_size: 0,
        slice_offset: 0,
        slice_size: 0,
    }
}

fn queues() -> EventQueues {
    Arc::new(Mutex::new(std::collections::HashMap::new()))
}

fn fill(len: usize, seed: u64) -> SliceBuf {
    let mut v = vec![0u8; len];
    let mut x = seed | 1;
    for b in v.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = x as u8;
    }
    SliceBuf::new(v.into_boxed_slice(), len)
}

fn drain_ffi(be: &mut FfiBackend, want_ended: u64) -> Vec<(i64, SliceBuf)> {
    let mut done = Vec::new();
    loop {
        let (_pid, ev) = be.wait_event(-1).expect("event stream closed");
        match ev {
            ProcEvent::Ended { token } if token == want_ended => return done,
            ProcEvent::InputDone {
                input_index, buf, ..
            } => done.push((input_index, buf)),
            ProcEvent::Error { message, .. } => panic!("ffi async error: {}", message),
            _ => {}
        }
    }
}

/// Blocking pop: the native proc emits events from its worker thread, so
/// the queue can legitimately be empty between polls.
fn pop_event(q: &EventQueues, pid: usize) -> ProcEvent {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let ev = {
            let mut m = q.lock().unwrap();
            m.get_mut(&pid).and_then(VecDeque::pop_front)
        };
        if let Some(ev) = ev {
            return ev;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "native event queue timed out"
        );
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

fn drain_native(q: &EventQueues, pid: usize, want_ended: u64) -> Vec<(i64, SliceBuf)> {
    let mut done = Vec::new();
    loop {
        match pop_event(q, pid) {
            ProcEvent::Ended { token } if token == want_ended => return done,
            ProcEvent::InputDone {
                input_index, buf, ..
            } => done.push((input_index, buf)),
            ProcEvent::Error { message, .. } => panic!("native error: {}", message),
            _ => {}
        }
    }
}

fn fetch_output_ffi(
    be: &mut FfiBackend,
    p: &mut impl GfProc,
    idx: i32,
    cap: usize,
) -> (Vec<u8>, bool) {
    p.get_output(
        idx,
        SliceBuf::new(vec![0; cap].into_boxed_slice(), cap),
        0xF0,
    )
    .unwrap();
    loop {
        let (_pid, ev) = be.wait_event(-1).expect("event stream closed");
        match ev {
            ProcEvent::OutputDone {
                cksum_valid, buf, ..
            } => return (buf.data[..buf.len].to_vec(), cksum_valid),
            ProcEvent::Error { message, .. } => panic!("ffi async error: {}", message),
            _ => {}
        }
    }
}

fn fetch_output_native(
    q: &EventQueues,
    p: &mut impl GfProc,
    idx: i32,
    cap: usize,
) -> (Vec<u8>, bool) {
    p.get_output(
        idx,
        SliceBuf::new(vec![0; cap].into_boxed_slice(), cap),
        0xF0,
    )
    .unwrap();
    loop {
        match pop_event(q, p.proc_id()) {
            ProcEvent::OutputDone {
                cksum_valid, buf, ..
            } => return (buf.data[..buf.len].to_vec(), cksum_valid),
            ProcEvent::Error { message, .. } => panic!("native error: {}", message),
            _ => {}
        }
    }
}

/// Feed `inputs` [(index, buf)] through both procs, end_input, then
/// compare every recovery output byte-for-byte.
fn run_case(
    method: i32,
    slice: usize,
    batch: i32,
    exps: Option<Vec<u16>>,
    inputs: &[(i32, usize, u64)],
    label: &str,
) {
    run_case_t(method, slice, batch, exps, inputs, label, 1)
}

fn run_case_t(
    method: i32,
    slice: usize,
    batch: i32,
    exps: Option<Vec<u16>>,
    inputs: &[(i32, usize, u64)],
    label: &str,
    threads: i32,
) {
    let count = exps.as_ref().map(|e| e.len()).unwrap_or(3);
    let mut be = FfiBackend::new().unwrap();
    let mut fp = be
        .new_proc(slice as u64, Some(&cpu(batch, 0)), &[], 2)
        .unwrap();
    let q = queues();
    let mut np =
        NativeGfProc::new(7, q.clone(), slice as u64, Some(&cpu(batch, method)), 2).unwrap();
    np.set_num_threads(threads).unwrap();
    fp.set_recovery_slices(exps.as_deref(), count).unwrap();
    np.set_recovery_slices(exps.as_deref(), count).unwrap();
    let mut tok = 0x100u64;
    let mut early: Vec<(i64, SliceBuf)> = Vec::new();
    let mut nearly: Vec<(i64, SliceBuf)> = Vec::new();
    for &(idx, len, seed) in inputs {
        let bf = fill(len, seed + 0x1000);
        let bn = fill(len, seed + 0x1000);
        tok += 1;
        // FFI staging is asynchronous - Full means wait for a staging-free
        // event (collecting any InputDone) then retry, like the pipeline's
        // pump_adds loop
        let mut buf = bf;
        let mut retries = 0;
        loop {
            match fp.add_input(buf, idx, tok).unwrap() {
                AddResult::Accepted => break,
                AddResult::Full(b) => {
                    buf = b;
                    retries += 1;
                    assert!(retries < 200, "ffi Full persist on {}", label);
                    // timed wait - staging may free without an event
                    match be.wait_event(50) {
                        Some((
                            _pid,
                            ProcEvent::InputDone {
                                input_index,
                                buf: rb,
                                ..
                            },
                        )) => early.push((input_index, rb)),
                        Some((_pid, ProcEvent::Error { message, .. })) => {
                            panic!("ffi async error: {}", message)
                        }
                        _ => {}
                    }
                }
            }
        }
        // native staging is bounded too (staging_limit*batchsize inputs in
        // flight) - retry on staging-free events like the FFI path
        let mut buf = bn;
        let mut retries = 0;
        loop {
            match np.add_input(buf, idx, tok).unwrap() {
                AddResult::Accepted => break,
                AddResult::Full(b) => {
                    buf = b;
                    retries += 1;
                    assert!(retries < 2000, "native Full persist on {}", label);
                    match pop_event(&q, np.proc_id()) {
                        ProcEvent::InputDone {
                            input_index,
                            buf: rb,
                            ..
                        } => nearly.push((input_index, rb)),
                        ProcEvent::Error { message, .. } => {
                            panic!("native error: {}", message)
                        }
                        _ => {}
                    }
                }
            }
        }
        // match the return-buffer stream for both
    }
    tok += 1;
    fp.end_input(tok).unwrap();
    np.end_input(tok).unwrap();
    let mut fdone = drain_ffi(&mut be, tok);
    // merge mid-loop InputDones back in submission order (they arrive first)
    let mut merged = early;
    merged.extend(fdone.drain(..));
    let fdone = merged;
    let mut ndone = drain_native(&q, np.proc_id(), tok);
    let mut nmerged = nearly;
    nmerged.extend(ndone.drain(..));
    let ndone = nmerged;
    assert_eq!(fdone.len(), ndone.len(), "InputDone count {}", label);
    for (i, ((fi, fb), (ni, nb))) in fdone.iter().zip(ndone.iter()).enumerate() {
        assert_eq!(fi, ni, "InputDone index {} {}", i, label);
        assert_eq!(
            &fb.data[..fb.len],
            &nb.data[..nb.len],
            "InputDone buf {} {}",
            i,
            label
        );
    }
    for j in 0..count as i32 {
        let (fo, fc) = fetch_output_ffi(&mut be, &mut fp, j, slice + 64);
        let (no, nc) = fetch_output_native(&q, &mut np, j, slice + 64);
        assert_eq!(fc, nc, "cksum_valid out {} {}", j, label);
        // FFI output includes a trailing checksum block; only the first
        // `slice` bytes are recovery data
        assert_eq!(
            &fo[..slice],
            &no[..slice.min(no.len())],
            "output {} {}",
            j,
            label
        );
    }
}

/// Run each differential case through every method the FFI proc can create
/// on this CPU (plus AUTO) - under emulation this covers the AVX512/GFNI
/// kernels, on NEON hardware the ARM set.
fn for_each_kernel(f: impl Fn(i32)) {
    let mut methods = vec![0];
    for m in 1..32 {
        if parpar_gf16::kernel::pick_explicit(m).is_some() {
            methods.push(m);
        }
    }
    for m in methods {
        f(m);
    }
}

#[test]
fn proc_vs_ffi() {
    for_each_kernel(|method| {
        // (slice, batch, exps, inputs)
        let cases: &[(usize, i32, Option<Vec<u16>>, &[(i32, usize, u64)], &str)] = &[
            (512, 1, None, &[(0, 512, 1)], "single"),
            (
                4096,
                4,
                None,
                &[(0, 4096, 2), (1, 4096, 3), (2, 4000, 4)],
                "basic-3in",
            ),
            (
                8192,
                2,
                Some(vec![0, 5, 17, 100]),
                &[
                    (0, 8192, 5),
                    (3, 8192, 6),
                    (7, 8000, 7),
                    (9, 8192, 8),
                    (12, 8192, 9),
                ],
                "custom-exps-5in",
            ),
            (
                4096,
                3,
                Some(vec![2, 4]),
                &[(1, 4096, 11), (0, 4000, 12)],
                "shuffled-idx",
            ),
            (
                3000,
                2,
                None,
                &[
                    (0, 3000, 21),
                    (1, 1, 22),
                    (2, 2999, 23),
                    (3, 0, 24),
                    (4, 3000, 25),
                ],
                "odd-short-empty",
            ),
        ];
        for &(sl, b, ref e, inputs, l) in cases {
            run_case(
                method,
                sl,
                b,
                e.clone(),
                inputs,
                &format!("m{}-{}", method, l),
            );
        }
    });
}

#[test]
fn accumulate_across_cycles() {
    for_each_kernel(|method| {
        // add->end->get->add->end->get: the chunked-generation pattern
        let slice = 4096usize;
        let mut be = FfiBackend::new().unwrap();
        let mut fp = be.new_proc(slice as u64, Some(&cpu(2, 0)), &[], 2).unwrap();
        let q = queues();
        let mut np =
            NativeGfProc::new(9, q.clone(), slice as u64, Some(&cpu(2, method)), 2).unwrap();
        fp.set_recovery_slices(None, 3).unwrap();
        np.set_recovery_slices(None, 3).unwrap();

        let mut tok = 0x200u64;
        for cycle in 0..3 {
            let incount = [4usize, 2, 3][cycle];
            for i in 0..incount {
                let len = if cycle == 2 { 2048 } else { slice };
                let idx = (cycle * 10 + i) as i32;
                tok += 1;
                // FFI staging is asynchronous - retry on Full like the pipeline
                let mut fbuf = fill(len, tok * 31);
                let mut retries = 0;
                loop {
                    match fp.add_input(fbuf, idx, tok).unwrap() {
                        AddResult::Accepted => break,
                        AddResult::Full(b) => {
                            fbuf = b;
                            retries += 1;
                            assert!(retries < 200, "ffi Full persist in accumulate");
                            let _ = be.wait_event(50);
                        }
                    }
                }
                np.add_input(fill(len, tok * 31), idx, tok).unwrap();
            }
            tok += 1;
            fp.end_input(tok).unwrap();
            np.end_input(tok).unwrap();
            drain_ffi(&mut be, tok);
            drain_native(&q, np.proc_id(), tok);
            for j in 0..3 {
                let (fo, _fc) = fetch_output_ffi(&mut be, &mut fp, j, slice + 64);
                let (no, _nc) = fetch_output_native(&q, &mut np, j, slice + 64);
                assert_eq!(
                    &fo[..slice],
                    &no[..slice.min(no.len())],
                    "cycle {} out {}",
                    cycle,
                    j
                );
            }
        }
    });
}

/// Same work split across N threads must stay byte-identical to FFI.
#[test]
fn multithreaded() {
    // many outputs + a region big enough to stripe
    let exps: Vec<u16> = (0..13).collect();
    let inputs: Vec<(i32, usize, u64)> =
        vec![(0, 20000, 1), (1, 19990, 2), (3, 20000, 3), (5, 3333, 4)];
    for t in [2, 4, 8] {
        for m in [
            parpar_gf16::kernel::method::LOOKUP,
            parpar_gf16::kernel::method::SHUFFLE_AVX2,
            parpar_gf16::kernel::method::XORJIT_AVX2,
            // aarch64 kernels - exercised where the ISA exists, skipped
            // elsewhere (pick_explicit is the arch gate)
            4,
            28,
            29,
        ] {
            if parpar_gf16::kernel::pick_explicit(m).is_none() {
                continue;
            }
            run_case_t(
                m,
                20000,
                4,
                Some(exps.clone()),
                &inputs,
                &format!("mt{}-m{}", t, m),
                t,
            );
        }
    }
}

#[test]
#[ignore]
fn muladd_bench() {
    use parpar_gf16::kernel::{self, GfKernel};
    for method in [0, 19, 12, 18] {
        let k: Box<dyn GfKernel> = match kernel::pick_explicit(method) {
            Some(k) => k,
            None => continue,
        };
        let tile = kernel::method_info(k.id()).target_chunk as usize;
        let src = vec![0xabu8; tile.max(1)];
        let mut dst = vec![0u8; tile.max(1)];
        for _ in 0..1000 {
            k.muladd(&mut dst, &src, 7);
        }
        let iters = (8u64 << 30) as usize / tile.max(1);
        let t = std::time::Instant::now();
        for _ in 0..iters {
            k.muladd(&mut dst, &src, 7);
        }
        let el = t.elapsed().as_secs_f64();
        eprintln!(
            "method={} tile={} muladd {:.2}s -> {:.2} GB/s",
            method,
            tile,
            el,
            (iters * tile) as f64 / el / 1e9
        );
    }
}

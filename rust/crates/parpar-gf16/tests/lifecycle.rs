// Lifecycle regressions: CPU range progress draining (composite deadlock
// fix), proc-close sentinels, and post-close op rejection.
use parpar_backend::{CompositeProc, EventQueues, GfProc, ProcEvent, RangeBackend};
use parpar_gf16::proc::NativeGfProc;
use parpar_gf16::range::CpuRangeBackend;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn queues() -> EventQueues {
    Arc::new(Mutex::new(HashMap::<usize, VecDeque<ProcEvent>>::new()))
}

/// The composite polling path relies on drain_progress reporting freed
/// staging; a CPU backend whose inner proc is asynchronous must surface
/// its InputDone events here or the composite deadlocks on Full retry.
#[test]
fn cpu_range_drain_progress_reports_frees() {
    let mut be = CpuRangeBackend::new(1, 256, None, 4).unwrap();
    be.set_slice_size(256);
    be.set_recovery_slices(Some(&[1]), 1).unwrap();
    be.add_input(&vec![7u8; 256], 0, true).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut freed = 0i64;
    while freed == 0 && Instant::now() < deadline {
        freed += be.drain_progress().unwrap();
        if freed == 0 {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    assert_eq!(freed, 1, "InputDone must surface through drain_progress");
    be.deinit().unwrap();
}

/// drain_progress must propagate worker errors, not silently drop them.
#[test]
fn cpu_range_drain_progress_propagates_error() {
    let mut be = CpuRangeBackend::new(2, 256, None, 4).unwrap();
    be.set_slice_size(256);
    be.deinit().unwrap();
    // after close, draining must surface the close/dead sentinel rather
    // than reporting "no progress" forever
    let deadline = Instant::now() + Duration::from_secs(5);
    let r = loop {
        match be.drain_progress() {
            Err(e) => break Err(e),
            Ok(_) if Instant::now() > deadline => break Ok(()),
            Ok(_) => std::thread::sleep(Duration::from_millis(5)),
        }
    };
    assert!(r.is_err(), "dead proc must surface an error, not hang");
}

/// close() emits a terminal error event so waiters never spin forever.
#[test]
fn closed_proc_emits_sentinel() {
    let q = queues();
    let mut p = NativeGfProc::new(9, q.clone(), 256, None, 2).unwrap();
    p.close().unwrap();
    let ev = p
        .wait_own_event(Duration::from_secs(5))
        .expect("sentinel must be queued");
    match ev {
        ProcEvent::Error { .. } => {}
        other => panic!("expected Error sentinel, got {:?}", other),
    }
}

/// Composite rejects operations after close.
#[test]
fn composite_closed_rejects_ops() {
    let q = queues();
    let be: Box<dyn RangeBackend> = Box::new(CpuRangeBackend::new(5, 256, None, 2).unwrap());
    let mut c = CompositeProc::new(5, q, 256, vec![(be, 0, 256)]).unwrap();
    c.close().unwrap();
    let buf = parpar_backend::SliceBuf::new(vec![0u8; 256].into_boxed_slice(), 256);
    assert!(c.add_input(buf, 0, 1).is_err(), "add after close must fail");
    assert!(c.end_input(2).is_err(), "end after close must fail");
    assert!(
        c.set_recovery_slices(Some(&[1]), 1).is_err(),
        "set_recovery after close must fail"
    );
}

//! Phase 0 lifecycle coverage for the C ABI backend.
//!
//! Exercises the rejection paths, mid-run cancellation and shutdown that
//! the JS addon relies on (gf.cc validation + async teardown). Error
//! strings asserted here mirror the addon's messages so the future CLI
//! surfaces identical diagnostics.

use parpar_backend::{
    ComputeBackend, CpuSpec, GfProc, HasherInput, HasherOutput, ProcEvent, SliceBuf,
};
use parpar_ffi::FfiBackend;

fn cpu_default() -> CpuSpec {
    CpuSpec {
        method: 0,
        input_batchsize: 0,
        input_minbatchsize: 0,
        chunk_size: 0,
        slice_offset: 0,
        slice_size: 0,
    }
}

fn make_proc(be: &mut FfiBackend, slice_size: u64) -> parpar_ffi::FfiGfProc {
    be.new_proc(slice_size, Some(&cpu_default()), &[], 2)
        .expect("proc creation should succeed")
}

fn drain(be: &mut FfiBackend) {
    while be.wait_event(0).is_some() {}
}

// Run one input through end_input and wait for the Ended event, so the
// proc reaches the hasOutput state the addon gates getOutput on.
fn run_to_output(be: &mut FfiBackend, p: &mut parpar_ffi::FfiGfProc, slice: usize) {
    p.add_input(
        SliceBuf::new(vec![5u8; slice].into_boxed_slice(), slice),
        0,
        0x10,
    )
    .unwrap();
    p.end_input(0x11).unwrap();
    loop {
        let (_pid, ev) = be.wait_event(-1).expect("event stream closed");
        match ev {
            ProcEvent::Ended { .. } => return,
            ProcEvent::Error { message, .. } => panic!("unexpected async error: {}", message),
            _ => {}
        }
    }
}

#[test]
fn reject_bad_construction() {
    let mut be = FfiBackend::new().unwrap();
    let cpu = cpu_default();

    for bad in [0u64, 1, 3, 1025] {
        let e = be
            .new_proc(bad, Some(&cpu), &[], 2)
            .expect_err("odd/small slice size must fail");
        assert!(
            e.message.contains("Slice size is invalid"),
            "unexpected error: {}",
            e.message
        );
    }

    let e = be
        .new_proc(1024, None, &[], 2)
        .expect_err("no backends must fail");
    assert!(e
        .message
        .contains("At least the CPU or one OpenCL device must be enabled"));

    let bad_offset = CpuSpec {
        slice_offset: 512,
        slice_size: 1024,
        ..cpu_default()
    };
    let e = be
        .new_proc(1024, Some(&bad_offset), &[], 2)
        .expect_err("offset+size overflow must fail");
    assert!(e
        .message
        .contains("CPU slice offset+size cannot exceed the slice size"));

    let odd_sub = CpuSpec {
        slice_size: 511,
        ..cpu_default()
    };
    let e = be
        .new_proc(1024, Some(&odd_sub), &[], 2)
        .expect_err("odd CPU slice size must fail");
    assert!(e.message.contains("CPU slice size must be a multiple of 2"));

    // staging_areas <= 0 selects the default; only out-of-range counts fail
    let e = be
        .new_proc(1024, Some(&cpu), &[], 40000)
        .expect_err("huge staging count must fail");
    assert!(e.message.contains("Staging area count is invalid"));

    // the addon's setupMethod switch sends unknown ids to default: (Lookup),
    // so an out-of-range method succeeds rather than erroring
    let odd_method = CpuSpec {
        method: 777,
        ..cpu_default()
    };
    let mut p = be
        .new_proc(1024, Some(&odd_method), &[], 2)
        .expect("out-of-range method resolves to default, like the addon");
    p.set_recovery_slices(None, 1).unwrap();
    let info = p.info().unwrap();
    assert_eq!(info.cpu.unwrap().method_desc, "Lookup");
    p.close().unwrap();
    drain(&mut be);
}

#[test]
fn reject_bad_state_calls() {
    let mut be = FfiBackend::new().unwrap();
    let mut p = make_proc(&mut be, 1024);

    let e = p
        .info()
        .expect_err("info before setRecoverySlices must fail");
    assert!(e.message.contains("setRecoverySlices not yet called"));

    let out = SliceBuf::new(vec![0u8; 1024].into_boxed_slice(), 1024);
    let e = p
        .get_output(0, out, 1)
        .expect_err("output before processing must fail");
    assert!(e.message.contains("No finalized output to retrieve"));

    let e = p
        .set_recovery_slices(Some(&[]), 0)
        .expect_err("empty recovery set must fail");
    assert!(e
        .message
        .contains("At least one recovery index must be supplied"));

    let e = p
        .set_recovery_slices(Some(&[65535]), 1)
        .expect_err("exponent > 65534 must fail");
    assert!(e.message.contains("Invalid recovery index supplied"));

    let e = p
        .add_input(
            SliceBuf::new(vec![0u8; 1024].into_boxed_slice(), 1024),
            0,
            2,
        )
        .expect_err("input before setRecoverySlices must fail");
    assert!(e.message.contains("setRecoverySlices not yet called"));

    p.set_recovery_slices(None, 4).unwrap();
    run_to_output(&mut be, &mut p, 1024);

    // now hasOutput is set: index and size validation is reachable
    let e = p
        .get_output(
            4,
            SliceBuf::new(vec![0u8; 1024].into_boxed_slice(), 1024),
            3,
        )
        .expect_err("recovery index out of range must fail");
    assert!(e.message.contains("Recovery index is not valid"));

    let e = p
        .get_output(0, SliceBuf::new(vec![0u8; 8].into_boxed_slice(), 8), 4)
        .expect_err("undersized output buffer must fail");
    assert!(e.message.contains("Output buffer too small"));

    p.close().unwrap();
    let e = p.close().expect_err("double close must fail");
    assert!(e.message.contains("Already closed"));
    drain(&mut be);
}

#[test]
fn cancel_mid_run_and_shutdown() {
    let mut be = FfiBackend::new().unwrap();
    {
        let mut p = make_proc(&mut be, 4096);
        p.set_recovery_slices(None, 3).unwrap();
        p.set_progress(true).unwrap();
        // flood inputs so some are in flight at drop time
        for i in 0..40 {
            let buf = SliceBuf::new(vec![i as u8; 4096].into_boxed_slice(), 4096);
            match p.add_input(buf, i, i as u64) {
                Ok(_) | Err(_) => {}
            }
        }
        // p dropped here with work in flight - must not hang or crash
    }
    // events for the dead proc may still be queued; drain them
    drain(&mut be);

    // free_mem during running is a real rejection; allowed once ended
    let mut p2 = make_proc(&mut be, 2048);
    p2.set_recovery_slices(None, 2).unwrap();
    let buf = SliceBuf::new(vec![7u8; 2048].into_boxed_slice(), 2048);
    let _ = p2.add_input(buf, 0, 100);
    p2.end_input(101).unwrap();
    let e = p2
        .free_mem()
        .expect_err("free_mem whilst running must fail");
    assert!(e.message.contains("Cannot free memory whilst running"));
    run_to_output_drain(&mut be);
    p2.free_mem().unwrap();
    p2.close().unwrap();
    drain(&mut be);
}

fn run_to_output_drain(be: &mut FfiBackend) {
    loop {
        let (_pid, ev) = be.wait_event(-1).expect("event stream closed");
        if let ProcEvent::Ended { .. } = ev {
            return;
        }
    }
}

#[test]
fn ctx_drop_with_live_procs() {
    // dropping the backend while procs are still alive must close them
    // rather than leak the libuv loop thread
    let mut be = FfiBackend::new().unwrap();
    let mut p = make_proc(&mut be, 2048);
    p.set_recovery_slices(None, 1).unwrap();
    let buf = SliceBuf::new(vec![1u8; 2048].into_boxed_slice(), 2048);
    let _ = p.add_input(buf, 0, 7);
    drop(p); // closes via Drop
    drop(be); // ctx teardown drains remaining events
}

#[test]
fn hasher_lifecycle() {
    let mut be = FfiBackend::new().unwrap();

    // input hasher: end() finalises; further calls must fail
    let block_out = vec![0u8; 20 * 8].into_boxed_slice();
    let mut hin = be.new_input_hasher(1024, block_out).unwrap();
    hin.update(b"hello").unwrap();
    hin.update(b" world").unwrap();
    let mut md5 = [0u8; 16];
    let _blocks = hin.end(&mut md5).unwrap();
    // "hello world" md5
    assert_eq!(
        md5.iter().map(|b| format!("{:02x}", b)).collect::<String>(),
        "5eb63bbbe01eeed093cb22bb8f5acdc3"
    );
    let e = hin.update(b"x").expect_err("update after end must fail");
    assert!(e.message.contains("Process already ended"));
    let e = hin.end(&mut md5).expect_err("second end must fail");
    assert!(e.message.contains("Process already ended"));

    // reset revives an active hasher (the addon's end() deinits the hasher
    // entirely, so reset-after-end is rejected rather than revived)
    let block_out = vec![0u8; 20 * 8].into_boxed_slice();
    let mut hin2 = be.new_input_hasher(1024, block_out).unwrap();
    hin2.update(b"junk").unwrap();
    hin2.reset().unwrap();
    hin2.update(b"hello").unwrap();
    hin2.end(&mut md5).unwrap();
    assert_eq!(
        md5.iter().map(|b| format!("{:02x}", b)).collect::<String>(),
        "5d41402abc4b2a76b9719d911017c592"
    );

    // output hasher: wrong region count rejected
    let mut hout = be.new_output_hasher(3).unwrap();
    let a = [0u8; 64];
    let b = [1u8; 64];
    let e = hout
        .update(&[&a[..], &b[..]])
        .expect_err("wrong region count must fail");
    assert!(e.message.contains("Invalid number of array items"));

    // mismatched region lengths rejected
    let c = [2u8; 32];
    let e = hout
        .update(&[&a[..], &b[..], &c[..]])
        .expect_err("mismatched lengths must fail");
    assert!(e.message.contains("All inputs' length must be equal"));

    // happy path still works
    let c = [2u8; 64];
    hout.update(&[&a[..], &b[..], &c[..]]).unwrap();
    let mut sums = [0u8; 48];
    hout.get(&mut sums).unwrap();
    hout.reset().unwrap();
    hout.update(&[&a[..], &b[..], &c[..]]).unwrap();
    let mut sums2 = [0u8; 48];
    hout.get(&mut sums2).unwrap();
    assert_eq!(sums, sums2);
}

#[test]
fn events_carry_tokens_and_buffers() {
    let mut be = FfiBackend::new().unwrap();
    let mut p = make_proc(&mut be, 2048);
    p.set_recovery_slices(None, 2).unwrap();
    p.set_progress(true).unwrap();

    let mut input_done = false;
    let ended;
    let mut outputs = 0;
    let mut progress = false;

    let buf = SliceBuf::new(vec![9u8; 2048].into_boxed_slice(), 2048);
    p.add_input(buf, 42, 0xA1).unwrap();
    p.end_input(0xB2).unwrap();

    // getOutput is rejected whilst running: wait for Ended first
    loop {
        let (_pid, ev) = be.wait_event(-1).expect("event stream closed");
        match ev {
            ProcEvent::InputDone {
                token,
                input_index,
                buf,
            } => {
                assert_eq!(token, 0xA1);
                assert_eq!(input_index, 42);
                assert_eq!(buf.len, 2048);
                assert_eq!(buf.data[0], 9);
                input_done = true;
            }
            ProcEvent::Ended { token } => {
                assert_eq!(token, 0xB2);
                ended = true;
                break;
            }
            ProcEvent::Progress { .. } => progress = true,
            ProcEvent::Error { message, .. } => panic!("unexpected async error: {}", message),
            _ => {}
        }
    }

    p.get_output(
        0,
        SliceBuf::new(vec![0u8; 2048].into_boxed_slice(), 2048),
        0xC3,
    )
    .unwrap();
    p.get_output(
        1,
        SliceBuf::new(vec![0u8; 2048].into_boxed_slice(), 2048),
        0xC4,
    )
    .unwrap();

    while outputs < 2 {
        let (_pid, ev) = be.wait_event(-1).expect("event stream closed");
        match ev {
            ProcEvent::OutputDone {
                token,
                cksum_valid,
                buf,
            } => {
                assert!(token == 0xC3 || token == 0xC4);
                assert!(cksum_valid);
                assert_eq!(buf.data.len(), 2048);
                outputs += 1;
            }
            ProcEvent::Error { message, .. } => panic!("unexpected async error: {}", message),
            _ => {}
        }
    }
    assert!(input_done && ended && progress);
    p.close().unwrap();
}

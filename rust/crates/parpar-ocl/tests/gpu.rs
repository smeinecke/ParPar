// GPU differential test - requires a real OpenCL device; skipped otherwise.
// Runs the OclBackend alone inside CompositeProc over a full-slice range and
// compares outputs against the scalar GF multiply oracle.

use parpar_backend::composite::{CompositeProc, EventQueues, RangeBackend};
use parpar_backend::{GfProc, ProcEvent, SliceBuf};
use parpar_gf16::{gf, CpuRangeBackend};
use parpar_ocl::{enumerate, host::OclBackend};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

fn queues() -> EventQueues {
    Arc::new(Mutex::new(HashMap::<usize, VecDeque<ProcEvent>>::new()))
}

fn drain(qs: &EventQueues, id: usize) -> Vec<ProcEvent> {
    qs.lock()
        .unwrap()
        .entry(id)
        .or_default()
        .drain(..)
        .collect()
}

#[test]
fn ocl_backend_matches_scalar() {
    let plats = enumerate();
    if plats.is_empty() {
        eprintln!("no OpenCL runtime - skipping");
        return;
    }
    let (plat_idx, dev_idx) = 'found: {
        for (pi, p) in plats.iter().enumerate() {
            if !p.devices.is_empty() {
                break 'found (pi as i32, 0i32);
            }
        }
        eprintln!("no OpenCL devices - skipping");
        return;
    };

    let slice_size = 8192usize; // multiple of 64
    let num_outputs = 4usize;
    let exps: Vec<u16> = (0..num_outputs as u16).map(|i| i * 7 + 1).collect();

    let mut be = OclBackend::new(plat_idx, dev_idx, 2).expect("backend create");
    be.set_slice_size(slice_size);
    be.init(1 /* Lookup */, 0, 0, 0, 0, 32).expect("init");

    let qs = queues();
    let proc_id = 77usize;
    let mut proc = CompositeProc::new(
        proc_id,
        qs.clone(),
        slice_size,
        vec![(Box::new(be), 0, slice_size)],
    )
    .expect("composite");
    proc.set_recovery_slices(Some(&exps), num_outputs)
        .expect("recovery");

    // inputs chosen to force full+partial kernel variants
    let inputs: Vec<Vec<u8>> = (0..3u32)
        .map(|i| {
            (0..slice_size)
                .map(|b| ((i * 131 + b as u32).wrapping_mul(2654435761) >> 13) as u8)
                .collect()
        })
        .collect();
    for (i, data) in inputs.iter().enumerate() {
        let tok = 1000 + i as u64;
        let mut buf = SliceBuf::new(data.clone().into_boxed_slice(), data.len());
        loop {
            match proc.add_input(buf, i as i32, tok).unwrap() {
                parpar_backend::AddResult::Accepted => break,
                parpar_backend::AddResult::Full(b) => {
                    buf = b;
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }
    }
    proc.end_input(2000).unwrap();

    // expected: out[j] = sum_i coeff(i, exps[j]) * input[i]
    for (oi, &e) in exps.iter().enumerate() {
        let mut acc = vec![0u8; slice_size];
        for (i, data) in inputs.iter().enumerate() {
            let coeff = gf::gfmat_coeff(i as u16, e);
            gf::muladd_region(&mut acc, data, slice_size, coeff);
        }
        let out = SliceBuf::new(vec![0u8; slice_size].into_boxed_slice(), slice_size);
        proc.get_output(oi as i32, out, 3000 + oi as u64).unwrap();
        let evs = drain(&qs, proc_id);
        let mut got = None;
        for ev in evs {
            if let ProcEvent::OutputDone {
                cksum_valid, buf, ..
            } = ev
            {
                got = Some((cksum_valid, buf));
            }
        }
        let (valid, buf) = got.expect("OutputDone event");
        assert!(valid, "checksum invalid for output {}", oi);
        assert_eq!(&buf.as_bytes()[..], &acc[..], "output {} mismatch", oi);
    }
    proc.close().unwrap();
}

/// Two OclBackends on disjoint ranges of the same device - exercises the
/// composite's split/merge path the same way a CPU+GPU split would.
#[test]
fn ocl_split_range_composite() {
    let plats = enumerate();
    if plats.is_empty() || plats.iter().all(|p| p.devices.is_empty()) {
        eprintln!("no OpenCL device - skipping");
        return;
    }
    let plat_idx = plats.iter().position(|p| !p.devices.is_empty()).unwrap() as i32;

    let slice_size = 8192usize;
    let split = 4096usize;
    let num_outputs = 3usize;
    let exps: Vec<u16> = (0..num_outputs as u16).map(|i| i * 5 + 3).collect();

    let mk = || {
        let mut b = OclBackend::new(plat_idx, 0, 2).expect("backend create");
        b.set_slice_size(slice_size);
        b.init(1, 0, 0, 0, 0, 32).expect("init");
        b
    };
    let qs = queues();
    let proc_id = 78usize;
    let mut proc = CompositeProc::new(
        proc_id,
        qs.clone(),
        slice_size,
        vec![
            (Box::new(mk()), 0, split),
            (Box::new(mk()), split, slice_size - split),
        ],
    )
    .expect("composite");
    proc.set_recovery_slices(Some(&exps), num_outputs)
        .expect("recovery");

    let inputs: Vec<Vec<u8>> = (0..2u32)
        .map(|i| {
            (0..slice_size)
                .map(|b| ((i * 977 + b as u32).wrapping_mul(2246822519) >> 11) as u8)
                .collect()
        })
        .collect();
    for (i, data) in inputs.iter().enumerate() {
        let buf = SliceBuf::new(data.clone().into_boxed_slice(), data.len());
        match proc.add_input(buf, i as i32, 500 + i as u64).unwrap() {
            parpar_backend::AddResult::Accepted => {}
            parpar_backend::AddResult::Full(_) => panic!("unexpected full"),
        }
    }
    proc.end_input(900).unwrap();

    for (oi, &e) in exps.iter().enumerate() {
        let mut acc = vec![0u8; slice_size];
        for (i, data) in inputs.iter().enumerate() {
            let coeff = gf::gfmat_coeff(i as u16, e);
            gf::muladd_region(&mut acc, data, slice_size, coeff);
        }
        let out = SliceBuf::new(vec![0u8; slice_size].into_boxed_slice(), slice_size);
        proc.get_output(oi as i32, out, 800 + oi as u64).unwrap();
        let evs = drain(&qs, proc_id);
        let (valid, buf) = evs
            .into_iter()
            .find_map(|ev| {
                if let ProcEvent::OutputDone {
                    cksum_valid, buf, ..
                } = ev
                {
                    Some((cksum_valid, buf))
                } else {
                    None
                }
            })
            .expect("OutputDone");
        assert!(valid);
        assert_eq!(&buf.as_bytes()[..], &acc[..], "output {} mismatch", oi);
    }
    proc.close().unwrap();
}

/// CPU + GPU backends on disjoint ranges - the real mixed composite path.
#[test]
fn mixed_cpu_ocl_composite() {
    let plats = enumerate();
    if plats.is_empty() || plats.iter().all(|p| p.devices.is_empty()) {
        eprintln!("no OpenCL device - skipping");
        return;
    }
    let plat_idx = plats.iter().position(|p| !p.devices.is_empty()).unwrap() as i32;

    let slice_size = 8192usize;
    let split = 3072usize; // non-half, still even + 64B aligned
    let num_outputs = 3usize;
    let exps: Vec<u16> = (0..num_outputs as u16).map(|i| i * 5 + 3).collect();

    let cpu = CpuRangeBackend::new(900, split, None, 2).expect("cpu backend");
    let mut gpu = OclBackend::new(plat_idx, 0, 2).expect("gpu backend");
    gpu.set_slice_size(slice_size - split);
    gpu.init(1, 0, 0, 0, 0, 32).expect("init");

    let qs = queues();
    let proc_id = 79usize;
    let mut proc = CompositeProc::new(
        proc_id,
        qs.clone(),
        slice_size,
        vec![
            (Box::new(cpu), 0, split),
            (Box::new(gpu), split, slice_size - split),
        ],
    )
    .expect("composite");
    proc.set_recovery_slices(Some(&exps), num_outputs)
        .expect("recovery");

    let inputs: Vec<Vec<u8>> = (0..2u32)
        .map(|i| {
            (0..slice_size)
                .map(|b| ((i * 331 + b as u32).wrapping_mul(374761393) >> 9) as u8)
                .collect()
        })
        .collect();
    for (i, data) in inputs.iter().enumerate() {
        let buf = SliceBuf::new(data.clone().into_boxed_slice(), data.len());
        match proc.add_input(buf, i as i32, 600 + i as u64).unwrap() {
            parpar_backend::AddResult::Accepted => {}
            parpar_backend::AddResult::Full(_) => panic!("unexpected full"),
        }
    }
    proc.end_input(901).unwrap();

    for (oi, &e) in exps.iter().enumerate() {
        let mut acc = vec![0u8; slice_size];
        for (i, data) in inputs.iter().enumerate() {
            let coeff = gf::gfmat_coeff(i as u16, e);
            gf::muladd_region(&mut acc, data, slice_size, coeff);
        }
        let out = SliceBuf::new(vec![0u8; slice_size].into_boxed_slice(), slice_size);
        proc.get_output(oi as i32, out, 850 + oi as u64).unwrap();
        let evs = drain(&qs, proc_id);
        let (valid, buf) = evs
            .into_iter()
            .find_map(|ev| {
                if let ProcEvent::OutputDone {
                    cksum_valid, buf, ..
                } = ev
                {
                    Some((cksum_valid, buf))
                } else {
                    None
                }
            })
            .expect("OutputDone");
        assert!(valid);
        assert_eq!(&buf.as_bytes()[..], &acc[..], "output {} mismatch", oi);
    }
    proc.close().unwrap();
}

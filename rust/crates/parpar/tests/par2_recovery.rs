//! Differential test: RecoveryGen (GFWrapper port) recovery data bytes +
//! packet MD5s vs the JS reference (`test/dump_recdata.js` ->
//! `rust/fixtures/recdata.json`), driven through the FFI backend.

#![cfg(target_env = "gnu")]
use parpar::par2::*;
use parpar_backend::{CpuSpec, SliceBuf};
use parpar_ffi::FfiBackend;
use serde_json::Value;
use std::collections::HashMap;

const SLICE_SIZE: usize = 32768;
const N_IN: usize = 9;
const N_REC: usize = 5;

fn slice_data(num: usize) -> Box<[u8]> {
    let mut b = vec![0u8; SLICE_SIZE];
    for (i, x) in b.iter_mut().enumerate() {
        *x = ((i * 31 + num * 17) & 0xff) as u8;
    }
    b.into_boxed_slice()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

#[test]
fn recovery_data_matches_js() {
    let fx: HashMap<String, Value> = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/recdata.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let md5s: Vec<String> = fx["md5s"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let datas: Vec<String> = fx["datas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let backend = FfiBackend::new().unwrap();
    let ctx = SharedBackend::new(backend);
    let cpu = CpuSpec {
        method: 0,
        input_batchsize: 4,
        input_minbatchsize: 0,
        chunk_size: 0,
        slice_offset: 0,
        slice_size: 0,
    };
    let proc_opts = ProcOpts {
        cpu: Some(cpu),
        ocl: vec![],
        staging: 0,
        threads: Some(4),
    };
    let mut par = Par2::new(
        vec![],
        SLICE_SIZE as u64,
        "utf-8".into(),
        ctx,
        GfOpts::resolve(None, None),
        proc_opts,
    )
    .unwrap();
    par.gen
        .set_recovery_slices((0..N_REC as u16).collect())
        .unwrap();

    for i in 0..N_IN {
        par.gen
            .process_slice(SliceBuf::new(slice_data(i), SLICE_SIZE), i as i64)
            .unwrap();
    }
    par.gen.finish().unwrap();

    for idx in 0..N_REC {
        let got = par.gen.next_recovery().unwrap();
        assert_eq!(got, Some(idx), "recovery idx order");
        let data = par.gen.recovery_data(idx).unwrap().to_vec();
        assert_eq!(hex(&data), datas[idx], "recovery data {}", idx);
        assert_eq!(
            hex(&par.gen.recovery_md5(idx).unwrap()),
            md5s[idx],
            "recovery md5 {}",
            idx
        );
        par.gen.mark_consumed(idx).unwrap();
    }
    assert_eq!(par.gen.next_recovery().unwrap(), None);
    par.gen.wait_recovery_complete().unwrap();
}

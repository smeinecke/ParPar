//! Byte-parity test: packet model output vs the JS reference
//! (`test/dump_par2.js` -> `rust/fixtures/par2_packets.json`).

use parpar::par2::*;
use serde_json::Value;
use std::collections::HashMap;

const SLICE_SIZE: u64 = 65536;

/// Null GfCtx for packet-model tests (no backend needed).
struct NullCtx;
struct NullProc;
impl parpar_backend::GfProc for NullProc {
    fn proc_id(&self) -> usize {
        0
    }
    fn set_num_threads(&mut self, _t: i32) -> parpar_backend::Result<i32> {
        unimplemented!()
    }
    fn set_current_slice_size(&mut self, _s: u64) -> parpar_backend::Result<()> {
        unimplemented!()
    }
    fn set_recovery_slices(&mut self, _e: Option<&[u16]>, _c: usize) -> parpar_backend::Result<()> {
        unimplemented!()
    }
    fn free_mem(&mut self) -> parpar_backend::Result<()> {
        Ok(())
    }
    fn add_input(
        &mut self,
        _b: parpar_backend::SliceBuf,
        _i: i32,
        _t: u64,
    ) -> parpar_backend::Result<parpar_backend::AddResult> {
        unimplemented!()
    }
    fn end_input(&mut self, _t: u64) -> parpar_backend::Result<()> {
        unimplemented!()
    }
    fn get_output(
        &mut self,
        _i: i32,
        _o: parpar_backend::SliceBuf,
        _t: u64,
    ) -> parpar_backend::Result<()> {
        unimplemented!()
    }
    fn set_progress(&mut self, _on: bool) -> parpar_backend::Result<()> {
        Ok(())
    }
    fn info(&mut self) -> parpar_backend::Result<parpar_backend::GfInfo> {
        unimplemented!()
    }
    fn close(&mut self) -> parpar_backend::Result<()> {
        Ok(())
    }
}
impl GfCtx for NullCtx {
    type Proc = NullProc;
    fn new_proc(
        &mut self,
        _s: u64,
        _c: Option<&parpar_backend::CpuSpec>,
        _o: &[parpar_backend::OclSpec],
        _st: i32,
    ) -> parpar_backend::Result<NullProc> {
        unimplemented!()
    }
    fn wait_for(&mut self, _p: usize) -> parpar_backend::Result<parpar_backend::ProcEvent> {
        unimplemented!()
    }
    fn new_output_hasher(
        &mut self,
        _r: u32,
    ) -> parpar_backend::Result<Box<dyn parpar_backend::HasherOutput>> {
        unimplemented!()
    }
    fn new_input_hasher(
        &mut self,
        _s: u64,
        _b: Box<[u8]>,
    ) -> parpar_backend::Result<Box<dyn parpar_backend::HasherInput>> {
        unimplemented!()
    }
}

fn md5_of(s: &str) -> [u8; 16] {
    use md5::Digest;
    let mut h = md5::Md5::new();
    h.update(format!("seed-{}", s).as_bytes());
    h.finalize().into()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn fixture() -> HashMap<String, Value> {
    let s = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/par2_packets.json"
    ))
    .unwrap();
    serde_json::from_str(&s).unwrap()
}

#[test]
fn packet_bytes_match_js() {
    let fx = fixture();
    let get = |k: &str| fx[k].as_str().unwrap().to_string();

    let files: Vec<([u8; 16], u64, String, String, Option<std::path::PathBuf>)> = vec![
        (md5_of("a"), 0, "empty.txt".into(), "empty.txt".into(), None),
        (
            md5_of("b"),
            SLICE_SIZE * 2 + 123,
            "data file.bin".into(),
            "data file.bin".into(),
            None,
        ),
        (
            md5_of("c"),
            SLICE_SIZE,
            "ünïcode fíle.dat".into(),
            "ünïcode fíle.dat".into(),
            None,
        ),
        (
            md5_of("d"),
            100,
            "nul\u{0}name".into(),
            "nul\u{0}name".into(),
            None,
        ),
    ];
    let mut par = Par2::new(
        files,
        SLICE_SIZE,
        "utf-8".to_string(),
        NullCtx,
        GfOpts::resolve(None, None),
        ProcOpts::default(),
    )
    .unwrap();

    assert_eq!(hex(&par.info.set_id), get("setID"), "setID");
    assert_eq!(par.total_slices, fx["totalSlices"].as_u64().unwrap());

    // file order/ids/offsets after reverse-id sort
    let order: Vec<&str> = fx["fileOrder"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let ids: Vec<String> = fx["fileIds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let offs: Vec<u64> = fx["sliceOffsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    for (i, f) in par.files.iter().enumerate() {
        assert_eq!(f.display_name, order[i], "file order");
        assert_eq!(hex(&f.id), ids[i], "file id");
        assert_eq!(f.slice_offset, offs[i], "slice offset");
    }

    assert_eq!(hex(&par.get_packet_main(true)), get("main"), "main pkt");
    assert_eq!(
        hex(&par.make_packet_creator("ParPar v0.4.6 test äöü")),
        get("creator")
    );
    assert_eq!(
        hex(&par.make_packet_creator("plain")),
        get("creatorAsciiOnly")
    );

    let comments = [
        "test comment",
        "ünïcode çømment ✓",
        "emoji \u{1f680} surrogate",
        "",
    ];
    for (i, c) in comments.iter().enumerate() {
        assert_eq!(
            hex(&par.make_packet_comment(c, None).unwrap()),
            get(&format!("comment{}_auto", i)),
            "comment {} auto",
            i
        );
        assert_eq!(
            hex(&par.make_packet_comment(c, Some(CharOpt::Ascii)).unwrap()),
            get(&format!("comment{}_ascii", i)),
            "comment {} ascii",
            i
        );
        assert_eq!(
            hex(&par.make_packet_comment(c, Some(CharOpt::Unicode)).unwrap()),
            get(&format!("comment{}_uni", i)),
            "comment {} uni",
            i
        );
        assert_eq!(
            hex(&par.make_packet_comment(c, Some(CharOpt::Both)).unwrap()),
            get(&format!("comment{}_both", i)),
            "comment {} both",
            i
        );
        assert_eq!(
            par.packet_comment_size(c, None).unwrap() as u64,
            fx[&format!("comment{}_size_auto", i)].as_u64().unwrap()
        );
    }

    for (i, f) in par.files.iter_mut().enumerate() {
        f.md5 = Some(md5_of(&format!("filemd5-{}", f.display_name)));
        assert_eq!(
            hex(&f.make_packet_description(&par.info, None).unwrap()),
            get(&format!("filedesc{}_auto", i)),
            "filedesc {} auto",
            i
        );
        assert_eq!(
            hex(&f
                .make_packet_description(&par.info, Some(CharOpt::Ascii))
                .unwrap()),
            get(&format!("filedesc{}_ascii", i)),
            "filedesc {} ascii",
            i
        );
        assert_eq!(
            hex(&f
                .make_packet_description(&par.info, Some(CharOpt::Both))
                .unwrap()),
            get(&format!("filedesc{}_both", i)),
            "filedesc {} both",
            i
        );
        assert_eq!(
            f.packet_description_size(None, &par.info.ascii_charset) as u64,
            fx[&format!("filedesc{}_size", i)].as_u64().unwrap()
        );
        assert_eq!(
            hex(&f.make_packet_uni_name(&par.info)),
            get(&format!("uniname{}", i)),
            "uniname {}",
            i
        );
        // fill records region as the hasher would, then emit IFSC
        if let Some(mut recs) = f.take_checksum_records() {
            for (j, b) in recs.iter_mut().enumerate() {
                *b = ((80 + j) * 7 + i as usize) as u8;
            }
            f.put_checksum_records(recs);
        }
        assert_eq!(
            hex(&f.get_packet_checksums(&par.info, true)),
            get(&format!("ifsc{}", i)),
            "ifsc {}",
            i
        );
    }

    for exp in [0u16, 1, 32767, 65535] {
        let md5 = md5_of(&format!("recmd5-{}", exp));
        assert_eq!(
            hex(&par.info.recovery_header(exp, &md5, SLICE_SIZE)),
            get(&format!("recvhdr_{}", exp)),
            "recv header {}",
            exp
        );
    }
}

//! Readers/outfile smoke tests - chunk enumeration, pool semantics,
//! seq-reader ordering, and PAR2OutFile byte layout.

use std::io::Read;
use std::sync::Arc;

use parpar::fileinfo::FileInfo;
use parpar::generate::GenPacket;
use parpar::generate::PacketType;
use parpar::outfile::Par2OutFile;
use parpar::readers::{file_chunk_reader, BufferPool, FileSeqReader};

fn mk_files(dir: &std::path::Path, spec: &[(&str, usize)]) -> Vec<FileInfo> {
    spec.iter()
        .map(|(name, sz)| {
            let path = dir.join(name);
            std::fs::write(&path, vec![7u8; *sz]).unwrap();
            FileInfo {
                name: path.to_string_lossy().into_owned(),
                fs_path: Some(path.clone()),
                display_name: None,
                full_path: None,
                size: *sz as u64,
                md5_16k: [0u8; 16],
                md5: None,
            }
        })
        .collect()
}

#[test]
fn chunk_reader_enumerates_slices() {
    let dir = std::env::temp_dir().join(format!("pp_cr_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let files = mk_files(&dir, &[("a.bin", 10000), ("b.bin", 3), ("empty.bin", 0)]);
    let pool = BufferPool::new(vec![], 4096, 4);
    let mut got = Vec::new();
    let cb_pool = pool.clone();
    file_chunk_reader(&files, 4096, 4096, 0, &pool, 1, None, |f, _, d, n, s| {
        got.push((f.name.clone(), n, s));
        assert!(d[..n].iter().all(|&b| b == 7));
        // the callback owns the pooled buffer - return it like the
        // pipeline's SliceBuf reclaim does or the bounded pool starves
        cb_pool.put(d);
    })
    .unwrap();
    // a: ceil(10000/4096)=3 slices; b: 1; empty skipped
    assert_eq!(got.len(), 4);
    assert!(got[0].0.ends_with("a.bin") && got[0].2 == 0);
    assert_eq!(got[2].1, 1808); // 10000 - 2*4096
    assert!(got[3].0.ends_with("b.bin") && got[3].1 == 3);

    // chunk_offset skips first bytes: file.size <= offset -> skipped
    let mut got2 = Vec::new();
    file_chunk_reader(&files, 4096, 100, 500, &pool, 1, None, |f, _, d, n, s| {
        got2.push((f.name.clone(), n, s));
        cb_pool.put(d);
    })
    .unwrap();
    // b.bin (3 < 500) skipped; a.bin ceil(9500/4096)=3
    assert_eq!(got2.len(), 3);
    assert_eq!(got2[0].1, 100); // chunkSize reads, not sliceSize
}

#[test]
fn buffer_pool_reuse_and_cap() {
    let pool = BufferPool::new(vec![], 100, 2);
    let a = pool.get();
    let b = pool.get();
    assert_eq!(a.len(), 100);
    // a third get would block - verified via pooled count only
    pool.put(a);
    let c = pool.get(); // reuse: same allocation popped from stack
    assert_eq!(pool.pooled(), 0);
    pool.put(b);
    pool.put(c);
    assert_eq!(pool.pooled(), 2);
    pool.end();
}

#[test]
fn seq_reader_streams_in_order() {
    let dir = std::env::temp_dir().join(format!("pp_sr_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let files = mk_files(&dir, &[("a.bin", 10000), ("b.bin", 5000)]);
    let mut r = FileSeqReader::new(files, 4096, 3);
    r.require_chunk(0, 0).unwrap();
    let mut bufs = Vec::new();
    r.run(None, |res| {
        let mut d = res.unwrap();
        assert!(d.buffer.iter().all(|&b| b == 7));
        assert!(d.buffer.len() <= 4096);
        bufs.push((d.file_index, d.pos, d.buffer.len()));
        d.hashed();
        d.release();
    })
    .unwrap();
    // full coverage in file order, sequential positions
    let total: usize = bufs.iter().map(|b| b.2).sum();
    assert_eq!(total, 15000);
    assert_eq!(bufs[0].0, 0);
    // a.bin: 4096,4096,1808 then b.bin: 4096,904
    assert_eq!(
        bufs.iter().map(|b| b.2).collect::<Vec<_>>(),
        vec![4096, 4096, 1808, 4096, 904]
    );
}

#[test]
fn outfile_write_combine_and_layout() {
    let dir = std::env::temp_dir().join(format!("pp_of_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.vol.par2");

    let mut p1 = GenPacket {
        pkt_type: PacketType::Recovery,
        size: 200,
        index: 0,
        data_chunk_offset: 0,
        data: Vec::new(),
    };
    p1.set_data(vec![vec![0xAAu8; 200]], 0);
    let mut p2 = GenPacket {
        pkt_type: PacketType::Recovery,
        size: 100,
        index: 1,
        data_chunk_offset: 0,
        data: Vec::new(),
    };
    p2.set_data(vec![vec![0xBBu8; 100]], 0);

    let mut of = Par2OutFile::new(path.to_string_lossy().into_owned(), 2, 0, vec![p1, p2], 300);
    of.open(false).unwrap();
    of.prealloc().unwrap();
    // write-combining: both packets written in one pwritev call -> 2
    let n = of.write_packets(0, 0).unwrap();
    assert_eq!(n, 2);
    of.close(true).unwrap();

    let mut data = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_end(&mut data)
        .unwrap();
    assert_eq!(data.len(), 300);
    assert!(data[..200].iter().all(|&b| b == 0xAA));
    assert!(data[200..].iter().all(|&b| b == 0xBB));

    // exclusive create refuses overwrite
    let mut of2 = Par2OutFile::new(path.to_string_lossy().into_owned(), 0, 0, vec![], 0);
    assert!(of2.open(false).unwrap_err().contains("EEXIST"));
    assert!(of2.open(true).is_ok()); // overwrite allowed
}

#[test]
fn throttle_none_and_delay() {
    use parpar::throttle::ThrottleQueue;
    let t = ThrottleQueue::none();
    assert_eq!(t.pass(1.0).0, false);
    let d = Arc::new(ThrottleQueue::delay(10.0));
    let t0 = std::time::Instant::now();
    d.pass(1.0);
    assert!(t0.elapsed().as_millis() >= 9);
}

use parpar_backend::HasherInput;
#[test]
fn mdbench() {
    let data = vec![0xabu8; 1 << 30];
    let t = std::time::Instant::now();
    let mut h = parpar_hasher::NativeHasherInput::new(1 << 30, Box::new([]));
    h.update(&data).unwrap();
    let mut out = [0u8; 64];
    let _ = h.end(&mut out);
    let s = t.elapsed().as_secs_f64();
    eprintln!("MD5x2 1GiB: {s:.2}s -> {:.2} GB/s", 1.0 / s);
}

#[test]
#[ignore]
fn lane_prod_bench() {
    let gib = 1usize << 31;
    let chunk = vec![0xabu8; 4 << 20];
    let mut p = parpar_hasher::LaneProducer::new(4 << 20);
    let mut states = Vec::new();
    let t = std::time::Instant::now();
    let mut left = gib;
    while left > 0 {
        p.update(&chunk, &mut states);
        left -= chunk.len();
        states.clear();
    }
    let el = t.elapsed().as_secs_f64();
    eprintln!(
        "LaneProd 2GiB: {:.2}s -> {:.2} GB/s",
        el,
        gib as f64 / el / 1e9
    );
}

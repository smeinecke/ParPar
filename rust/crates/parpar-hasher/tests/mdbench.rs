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

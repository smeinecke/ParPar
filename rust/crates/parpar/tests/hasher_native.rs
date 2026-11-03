//! Differential: parpar-hasher (native Rust) vs parpar-ffi (C++ oracle).
//! Feeds identical byte streams through both HasherInput/HasherOutput
//! implementations and requires identical records + digests.

#![cfg(target_env = "gnu")]
use parpar_backend::{ComputeBackend, HasherInput, HasherOutput};
use parpar_ffi::FfiBackend;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rand_bytes(seed: &mut u64, n: usize) -> Vec<u8> {
    (0..n).map(|_| xorshift(seed) as u8).collect()
}

fn run_input<H: HasherInput>(
    new: impl FnOnce(u64, Box<[u8]>) -> Result<H, parpar_backend::BackendError>,
    data: &[u8],
    feeds: &[usize],
    slice_size: u64,
) -> ([u8; 16], Box<[u8]>) {
    let nrec = data.len() as u64 / slice_size + 1;
    let mut h =
        new(slice_size, vec![0u8; nrec as usize * 20].into_boxed_slice()).expect("hasher create");
    let mut off = 0usize;
    for &f in feeds {
        let end = (off + f).min(data.len());
        h.update(&data[off..end]).unwrap();
        off = end;
    }
    if off < data.len() {
        h.update(&data[off..]).unwrap();
    }
    let mut md5 = [0u8; 16];
    let bo = h.end(&mut md5).unwrap();
    (md5, bo)
}

#[test]
fn input_hasher_matches_ffi() {
    let mut seed = 0x9e3779b97f4a7c15u64;
    for &slice in &[64u64, 128, 200, 4096, 65536] {
        for &len in &[
            0usize,
            1,
            63,
            64,
            65,
            191,
            192,
            1000,
            slice as usize * 3 + 17,
        ] {
            let data = rand_bytes(&mut seed, len);
            // split feeds at awkward points (across tmp/pos_offset boundaries)
            let feeds = [7usize, 64, 1, 33, 128, 5, 4096, 19, 777];
            let (m_ffi, bo_ffi) = run_input(
                |s, bo| FfiBackend::new().and_then(|mut b| b.new_input_hasher(s, bo)),
                &data,
                &feeds,
                slice,
            );
            let (m_nat, bo_nat) = run_input(
                |s, bo| {
                    Ok::<_, parpar_backend::BackendError>(parpar_hasher::NativeHasherInput::new(
                        s, bo,
                    ))
                },
                &data,
                &feeds,
                slice,
            );
            assert_eq!(m_ffi, m_nat, "file md5 mismatch slice={slice} len={len}");
            assert_eq!(
                bo_ffi.as_ref(),
                bo_nat.as_ref(),
                "block_out mismatch slice={slice} len={len}"
            );
        }
    }
}

#[test]
fn output_hasher_matches_ffi() {
    let mut seed = 0xdeadbeefcafe1234u64;
    for &regions in &[1u32, 2, 3, 8, 17] {
        for &len in &[0usize, 1, 63, 64, 65, 300, 4096] {
            let bufs: Vec<Vec<u8>> = (0..regions).map(|_| rand_bytes(&mut seed, len)).collect();
            let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
            let mut out_ffi = vec![0u8; regions as usize * 16];
            let mut out_nat = vec![0u8; regions as usize * 16];
            {
                let mut b = FfiBackend::new().unwrap();
                let mut h = b.new_output_hasher(regions).unwrap();
                // feed in two parts to exercise buffering
                let split = len / 2;
                let p1: Vec<&[u8]> = refs.iter().map(|b| &b[..split]).collect();
                let p2: Vec<&[u8]> = refs.iter().map(|b| &b[split..]).collect();
                h.update(&p1).unwrap();
                h.update(&p2).unwrap();
                h.get(&mut out_ffi).unwrap();
            }
            {
                let mut h = parpar_hasher::NativeHasherOutput::new(regions);
                let split = len / 2;
                let p1: Vec<&[u8]> = refs.iter().map(|b| &b[..split]).collect();
                let p2: Vec<&[u8]> = refs.iter().map(|b| &b[split..]).collect();
                h.update(&p1).unwrap();
                h.update(&p2).unwrap();
                h.get(&mut out_nat).unwrap();
            }
            assert_eq!(out_ffi, out_nat, "regions={regions} len={len}");
        }
    }
}

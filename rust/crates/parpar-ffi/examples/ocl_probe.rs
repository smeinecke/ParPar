fn main() {
    let mut b = parpar_ffi::FfiBackend::new().unwrap();
    let plats = parpar_backend::ComputeBackend::opencl_platforms(&mut b);
    println!("plats: {:?}", plats.map(|v| v.len()));
    let d = parpar_backend::ComputeBackend::opencl_device_info(&mut b, -1, -1);
    println!(
        "dev -1/-1: {:?}",
        d.map(|o| o.map(|(p, d)| (p, d.id, d.name)))
    );
    let d2 = parpar_backend::ComputeBackend::opencl_device_info(&mut b, 0, 0);
    println!("dev 0/0: {:?}", d2.map(|o| o.map(|(p, d)| (p, d.id))));
}

fn main() {
    println!("available: {}", parpar_ocl::available());
    println!("platforms: {}", parpar_ocl::platforms().len());
    for p in parpar_ocl::enumerate() {
        println!("platform {}: {}", p.id, p.name);
        for d in &p.devices {
            println!("  {:?}", d);
        }
    }
}

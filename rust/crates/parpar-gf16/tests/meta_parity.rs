// Kernel metadata must agree with method_info() for every method the
// native backend can instantiate - proc.info() reports kernel values
// while gf_info reports the table, and both must match C++.
#[test]
fn kernel_meta_matches_method_info() {
    for m in 1..32 {
        let Some(k) = parpar_gf16::kernel::pick_explicit(m) else {
            continue;
        };
        let mi = parpar_gf16::kernel::method_info(m);
        assert_eq!(mi.id, m);
        assert_eq!(k.name(), mi.name, "name {}", m);
        // the kernel may be more permissive than the declared method
        // geometry (e.g. scalar serving a SIMD method), never stricter:
        // buffers are allocated/sized to the declared values
        assert!(
            k.alignment() as u64 <= mi.alignment && mi.alignment % k.alignment() as u64 == 0,
            "alignment {}: kernel {} declared {}",
            m,
            k.alignment(),
            mi.alignment
        );
        assert!(
            k.stride() as u64 <= mi.stride && mi.stride % k.stride() as u64 == 0,
            "stride {}: kernel {} declared {}",
            m,
            k.stride(),
            mi.stride
        );
    }
}

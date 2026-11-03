// Unimplemented ISA methods must be rejected by pick_explicit (the C++
// methods require intrinsics stable Rust cannot express) while pick()
// still substitutes the best available kernel.
#[test]
fn unimplemented_isas_explicit_none() {
    for m in [5, 6, 7, 8, 9, 27, 30, 31] {
        assert!(
            parpar_gf16::kernel::pick_explicit(m).is_none(),
            "method {} should be unimplemented",
            m
        );
        let k = parpar_gf16::kernel::pick(m);
        assert!(k.id() != m || m == 0, "pick({}) must substitute", m);
    }
}

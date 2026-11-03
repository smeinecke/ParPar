use parpar::args::{MethodRef, SpecTerm, SpecUnit};
use parpar::fileinfo::FileInfo;
use parpar::generate::{GenOpts, Par2Gen, SysEnv};
use parpar::par2::SharedBackend;
use parpar_ffi::FfiBackend;

fn main() {
    let mut files = vec![FileInfo {
        name: "a.bin".into(),
        size: 1000000,
        ..Default::default()
    }];
    for (u, name) in [
        (SpecUnit::Slices, "slices"),
        (SpecUnit::Bytes, "bytes"),
        (SpecUnit::Count, "count"),
    ] {
        let slice_def = SpecTerm {
            unit: u,
            value: 64.0,
            scale: 1.0,
        };
        let mut o = GenOpts::default();
        o.gf_method = Some(MethodRef::Name("".into()));
        o.recovery_slices = vec![SpecTerm {
            unit: SpecUnit::Count,
            value: 200.0,
            scale: 1.0,
        }];
        let env = SysEnv {
            total_mem: 67338215424,
            free_mem: 20000000000,
            constrained_mem: u64::MAX,
            is_64b: true,
            k_max_length: 9007199254740991,
        };
        let backend = FfiBackend::new().unwrap();
        let mut shared = SharedBackend::new(backend);
        let r = Par2Gen::new(
            &mut files,
            &slice_def,
            o,
            shared.clone(),
            "utf-8",
            &mut shared,
            &env,
        );
        println!(
            "{name}: {:?}",
            r.map(|g| (
                g.opts.slice_size,
                g.input_slices,
                g.opts.recovery_slices_res
            ))
        );
    }
}

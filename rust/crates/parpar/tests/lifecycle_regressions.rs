// Lifecycle regressions: Par2::new slice_size==0, RecoveryGen::finish
// without an initialised proc - both must be catchable errors (JS throws),
// never panics or hangs.
use parpar::par2::*;

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

#[test]
fn par2_zero_slice_size_errors_not_divzero() {
    let files = vec![([0u8; 16], 100u64, "a".to_string(), "a".to_string(), None)];
    let r = Par2::new(
        files,
        0,
        "utf-8".to_string(),
        NullCtx,
        GfOpts::resolve(None, None),
        ProcOpts::default(),
    );
    assert!(r.is_err(), "slice_size=0 must be a catchable error");
}

/// finish() with recovery slices configured but no gf proc (alloc_size=0
/// skips gf_init) must error, not unwrap None.
#[test]
fn finish_without_proc_errors() {
    let mut g: RecoveryGen<NullCtx> = RecoveryGen::new(
        NullCtx,
        0,
        [0u8; 16],
        GfOpts::resolve(None, None),
        ProcOpts::default(),
    );
    g.recovery_slices = vec![1];
    let r = g.finish();
    assert!(r.is_err(), "finish without proc must be a catchable error");
}

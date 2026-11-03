//! `CpuRangeBackend` - adapts `NativeGfProc` to the composite `RangeBackend`
//! contract so a CPU backend can share a slice with OpenCL backends
//! (equivalent to PAR2ProcCPU inside the C++ multi-backend PAR2Proc).
//!
//! The adapter feeds only this backend's byte range of each input and
//! synchronously drains the inner proc's events through a private queue.

use crate::proc::{EventQueues, NativeGfProc};
use parpar_backend::composite::{CanAdd, RangeBackend};
use parpar_backend::{
    AddResult, BackendError, CpuSpec, GfInfo, GfProc, ProcEvent, Result, SliceBuf,
};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub struct CpuRangeBackend {
    inner: NativeGfProc,
    queues: EventQueues,
    token: u64,
    added: bool,
    /// non-InputDone events pulled by drain_progress, replayed to the
    /// end_input/get_output wait loops via drain()
    stashed: VecDeque<ProcEvent>,
}

impl CpuRangeBackend {
    /// `alloc_size` is this backend's fixed byte-range length.
    pub fn new(
        proc_id: usize,
        alloc_size: usize,
        cpu: Option<&CpuSpec>,
        staging_areas: i32,
    ) -> Result<Self> {
        let queues: EventQueues =
            Arc::new(Mutex::new(HashMap::<usize, VecDeque<ProcEvent>>::new()));
        let spec = match cpu {
            Some(c) => {
                let mut s = c.clone();
                s.slice_offset = 0;
                s.slice_size = 0;
                s
            }
            None => CpuSpec {
                method: 0,
                input_batchsize: 1,
                input_minbatchsize: 1,
                chunk_size: 0,
                slice_offset: 0,
                slice_size: 0,
            },
        };
        let inner = NativeGfProc::new(
            proc_id,
            queues.clone(),
            alloc_size as u64,
            Some(&spec),
            staging_areas,
        )?;
        Ok(CpuRangeBackend {
            inner,
            queues,
            token: 0,
            added: false,
            stashed: VecDeque::new(),
        })
    }

    fn next_token(&mut self) -> u64 {
        self.token += 1;
        self.token
    }

    fn drain(&mut self) -> Vec<ProcEvent> {
        let mut out: Vec<ProcEvent> = self.stashed.drain(..).collect();
        let id = self.inner.proc_id();
        out.extend(self.queues.lock().unwrap().entry(id).or_default().drain(..));
        out
    }
}

impl RangeBackend for CpuRangeBackend {
    fn set_slice_size(&mut self, size: usize) {
        let _ = self.inner.set_current_slice_size(size as u64);
    }

    fn set_current_slice_size(&mut self, size: usize) -> Result<()> {
        self.inner.set_current_slice_size(size as u64)
    }

    fn set_recovery_slices(&mut self, exps: Option<&[u16]>, count: usize) -> Result<()> {
        self.inner.set_recovery_slices(exps, count)
    }

    fn can_add(&mut self) -> CanAdd {
        if self.inner.has_staging_room() {
            CanAdd::Ok
        } else {
            CanAdd::Full
        }
    }

    fn add_input(&mut self, data: &[u8], input_num: u16, _flush: bool) -> Result<()> {
        let tok = self.next_token();
        let buf = SliceBuf::new(data.to_vec().into_boxed_slice(), data.len());
        // the inner proc hands work to its background worker; InputDone
        // events accumulate on its private queue (drained lazily below)
        match self.inner.add_input(buf, input_num as i32, tok)? {
            AddResult::Accepted => {
                self.added = true;
                self.drain();
                Ok(())
            }
            AddResult::Full(_) => Err(BackendError::new("CPU staging full")),
        }
    }

    fn end_input(&mut self) -> Result<()> {
        let tok = self.next_token();
        self.inner.end_input(tok)?;
        // wait for the worker to emit Ended so the range backend keeps
        // synchronous end-of-pass semantics for the composite
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            for ev in self.drain() {
                match ev {
                    ProcEvent::Ended { .. } => return Ok(()),
                    ProcEvent::Error { message, .. } => return Err(BackendError::new(message)),
                    _ => {}
                }
            }
            if std::time::Instant::now() > deadline {
                return Err(BackendError::new("CPU range end_input timed out"));
            }
            if let Some(ev) = self
                .inner
                .wait_own_event(std::time::Duration::from_millis(50))
            {
                match ev {
                    ProcEvent::Ended { .. } => return Ok(()),
                    ProcEvent::Error { message, .. } => return Err(BackendError::new(message)),
                    _ => {}
                }
            }
        }
    }

    fn get_output(&mut self, index: usize, out: &mut [u8]) -> Result<bool> {
        let tok = self.next_token();
        let n = out.len();
        let buf = SliceBuf::new(vec![0u8; n].into_boxed_slice(), n);
        self.inner.get_output(index as i32, buf, tok)?;
        // wait for the worker to emit OutputDone (channel-ordered after any
        // prior end_input, so the checksum state is final)
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            for ev in self.drain() {
                match ev {
                    ProcEvent::OutputDone {
                        cksum_valid, buf, ..
                    } => {
                        out[..buf.len.min(n)].copy_from_slice(&buf.as_bytes()[..buf.len.min(n)]);
                        return Ok(cksum_valid);
                    }
                    ProcEvent::Error { message, .. } => return Err(BackendError::new(message)),
                    _ => {}
                }
            }
            if std::time::Instant::now() > deadline {
                return Err(BackendError::new("CPU range get_output timed out"));
            }
            if let Some(ev) = self
                .inner
                .wait_own_event(std::time::Duration::from_millis(50))
            {
                match ev {
                    ProcEvent::OutputDone {
                        cksum_valid, buf, ..
                    } => {
                        out[..buf.len.min(n)].copy_from_slice(&buf.as_bytes()[..buf.len.min(n)]);
                        return Ok(cksum_valid);
                    }
                    ProcEvent::Error { message, .. } => return Err(BackendError::new(message)),
                    _ => {}
                }
            }
        }
    }

    fn has_added(&self) -> bool {
        self.added
    }

    fn discard_output(&mut self) {
        // inner.maybe_discard handles the reset automatically on the next
        // add/end - just drop the added flag.
        self.added = false;
    }

    fn deinit(&mut self) -> Result<()> {
        // full shutdown, not just memory release - otherwise the inner
        // worker thread keeps running until the Box is dropped
        self.inner.close()
    }

    fn free_mem(&mut self) -> Result<()> {
        self.inner.free_mem()
    }

    fn drain_progress(&mut self) -> Result<i64> {
        // the inner proc is async: staging frees arrive as InputDone
        // events on the private queue. Count them for the composite's
        // Progress reporting - without this the composite can wait
        // forever on a Full retry while frees go unreported.
        let mut freed = 0i64;
        for ev in self.drain() {
            match ev {
                ProcEvent::InputDone { .. } => freed += 1,
                ProcEvent::Error { message, .. } => return Err(BackendError::new(message)),
                other => self.stashed.push_back(other),
            }
        }
        Ok(freed)
    }

    fn info(&mut self) -> GfInfo {
        self.inner.info().unwrap_or(GfInfo {
            cpu: None,
            ocl: Vec::new(),
        })
    }

    fn set_num_threads(&mut self, threads: i32) -> Result<i32> {
        self.inner.set_num_threads(threads)
    }
}

//! Multi-backend composite (port of `PAR2Proc` in gf16/controller.cpp).
//! Splits each input slice across per-device range backends and merges the
//! per-backend results into the single `GfProc` event contract.

use crate::{AddResult, BackendError, GfInfo, GfProc, ProcEvent, Result, SliceBuf};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Shared event mailboxes for native processors (one deque per proc_id).
pub type EventQueues = Arc<Mutex<HashMap<usize, VecDeque<ProcEvent>>>>;

/// `PAR2ProcBackendAddResult` - whether a backend can take another input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanAdd {
    Ok,
    OkBusy,
    Full,
}

/// A byte-range compute backend (port of `IPAR2ProcBackend`).
/// Handles only `[0, current_slice_size)` of the data handed to it - the
/// composite slices the caller's buffer to the backend's range.
pub trait RangeBackend: Send {
    fn set_slice_size(&mut self, size: usize);
    /// Reduce the active slice size (<= allocated).
    fn set_current_slice_size(&mut self, size: usize) -> Result<()>;
    fn set_recovery_slices(&mut self, exps: Option<&[u16]>, count: usize) -> Result<()>;

    fn can_add(&mut self) -> CanAdd;
    /// Stage one input. `data` is exactly this backend's range of the input
    /// (already offset by the composite). `input_num` is the global slice
    /// index; `flush` requests dispatch of a partial batch.
    fn add_input(&mut self, data: &[u8], input_num: u16, flush: bool) -> Result<()>;
    fn end_input(&mut self) -> Result<()>;
    /// Writes `current_slice_size` bytes to `out`; false on checksum failure.
    fn get_output(&mut self, index: usize, out: &mut [u8]) -> Result<bool>;

    fn has_added(&self) -> bool;
    fn discard_output(&mut self);
    fn deinit(&mut self) -> Result<()>;
    /// `freeProcessingMem` - CPU backends release processing memory;
    /// PAR2ProcOCL's override is empty, hence the default no-op
    fn free_mem(&mut self) -> Result<()> {
        Ok(())
    }
    /// Free completed staging; returns inputs freed since last call
    /// (Progress event payload). Errors from the backend's event stream
    /// propagate so the composite can surface them instead of hanging.
    fn drain_progress(&mut self) -> Result<i64>;
    /// Method/batching info for the `info()` display.
    fn info(&mut self) -> GfInfo;
    /// CPU-only procs forward thread count; OpenCL backends return -1
    /// (PAR2Proc::setNumThreads only reaches the CPU backend).
    fn set_num_threads(&mut self, _threads: i32) -> Result<i32> {
        Ok(-1)
    }
}

struct Backend {
    be: Box<dyn RangeBackend>,
    offset: usize,
    slice_size: usize,
    alloc_size: usize,
    added: std::collections::HashSet<u32>,
}

/// Composite proc: routes each input/output range to the backend covering it.
pub struct CompositeProc {
    proc_id: usize,
    queues: EventQueues,
    backends: Vec<Backend>,
    current_slice_size: usize,
    has_added: bool,
    needs_discard: bool,
    progress_on: bool,
    closed: bool,
    /// tracks end_input issuance; informational only - deferred adds may
    /// legitimately arrive afterwards (C++ treats add-after-end as an
    /// assert-only check; the cmd queue still orders them correctly)
    #[allow(dead_code)]
    ended: bool,
}

impl CompositeProc {
    /// `backends` are (backend, offset, size); they must tile [0, slice_size)
    /// without overlap or init fails (mirrors checkBackendAllocation).
    pub fn new(
        proc_id: usize,
        queues: EventQueues,
        slice_size: usize,
        backends: Vec<(Box<dyn RangeBackend>, usize, usize)>,
    ) -> Result<CompositeProc> {
        let mut bs = Vec::with_capacity(backends.len());
        for (mut be, offset, size) in backends {
            be.set_slice_size(size);
            bs.push(Backend {
                be,
                offset,
                slice_size: size,
                alloc_size: size,
                added: std::collections::HashSet::new(),
            });
        }
        let p = CompositeProc {
            proc_id,
            queues,
            backends: bs,
            current_slice_size: slice_size,
            has_added: false,
            needs_discard: false,
            progress_on: false,
            closed: false,
            ended: false,
        };
        if !p.check_allocation() {
            return Err(BackendError::new("Invalid backend allocation"));
        }
        Ok(p)
    }

    /// Port of checkBackendAllocation: ranges must chain-cover
    /// [0, current_slice_size) exactly; overlaps unsupported.
    fn check_allocation(&self) -> bool {
        if self.backends.is_empty() {
            return false;
        }
        let mut start = self.backends[0].offset;
        let mut end = start + self.backends[0].slice_size;
        let mut checked = vec![false; self.backends.len()];
        checked[0] = true;
        let mut unchecked = self.backends.len() - 1;
        let mut overlap = false;
        while unchecked > 0 {
            let mut found = false;
            for i in 1..self.backends.len() {
                if checked[i] {
                    continue;
                }
                let b = &self.backends[i];
                let cur_end = b.offset + b.slice_size;
                if b.offset <= start && cur_end >= start {
                    if cur_end > start {
                        overlap = true;
                    }
                    start = b.offset;
                    if cur_end > end {
                        end = cur_end;
                    }
                } else if cur_end >= end && b.offset <= end {
                    if b.offset < end {
                        overlap = true;
                    }
                    end = cur_end;
                    if b.offset < start {
                        start = b.offset;
                    }
                } else if b.offset > start && cur_end < end {
                    overlap = true;
                } else {
                    continue;
                }
                checked[i] = true;
                unchecked -= 1;
                found = true;
                if b.offset & 1 != 0 {
                    return false;
                }
                if b.slice_size & 1 != 0 && b.offset + b.slice_size != self.current_slice_size {
                    return false;
                }
            }
            if !found {
                return false;
            }
        }
        if overlap {
            return false;
        }
        start == 0 && end == self.current_slice_size
    }

    fn emit(&self, ev: ProcEvent) {
        self.queues
            .lock()
            .unwrap()
            .entry(self.proc_id)
            .or_default()
            .push_back(ev);
    }

    fn maybe_discard(&mut self) {
        if self.needs_discard {
            for b in &mut self.backends {
                b.be.discard_output();
            }
            self.has_added = false;
            self.needs_discard = false;
        }
    }

    /// Poll backends for freed staging; emits Progress when enabled,
    /// Error when a backend's event stream reports one (e.g. dead
    /// worker) so waiters don't hang on a Full retry.
    fn poll_progress(&mut self) {
        let mut freed = 0i64;
        let mut errs = Vec::new();
        for b in &mut self.backends {
            match b.be.drain_progress() {
                Ok(n) => freed += n,
                Err(e) => errs.push(e.to_string()),
            }
        }
        for message in errs {
            self.emit(ProcEvent::Error { token: 0, message });
        }
        if freed > 0 && self.progress_on {
            self.emit(ProcEvent::Progress { num_inputs: freed });
        }
    }
}

impl GfProc for CompositeProc {
    fn proc_id(&self) -> usize {
        self.proc_id
    }

    fn set_num_threads(&mut self, threads: i32) -> Result<i32> {
        // PAR2Proc::setNumThreads forwards to the CPU backend only
        let mut r = -1;
        for b in &mut self.backends {
            if let Ok(v) = b.be.set_num_threads(threads) {
                r = v;
            }
        }
        Ok(r)
    }

    fn set_current_slice_size(&mut self, slice_size: u64) -> Result<()> {
        let new_size = slice_size as usize;
        if self.backends.len() == 1 {
            self.current_slice_size = new_size;
            let b = &mut self.backends[0];
            b.slice_size = new_size;
            b.alloc_size = b.alloc_size.max(new_size);
            return b.be.set_current_slice_size(new_size);
        }
        if new_size > self.current_slice_size {
            let total: usize = self.backends.iter().map(|b| b.alloc_size).sum();
            if new_size > total {
                return Err(BackendError::new("slice size exceeds allocation"));
            }
        }
        self.current_slice_size = new_size;
        let mut pos = 0usize;
        for b in &mut self.backends {
            b.slice_size = (new_size - pos).min(b.alloc_size);
            b.offset = pos;
            b.be.set_current_slice_size(b.slice_size)?;
            pos += b.slice_size;
        }
        Ok(())
    }

    fn set_recovery_slices(&mut self, exps: Option<&[u16]>, count: usize) -> Result<()> {
        if self.closed {
            return Err(BackendError::new(
                "Cannot set recovery slices after proc closed",
            ));
        }
        for b in &mut self.backends {
            b.be.set_recovery_slices(exps, count)?;
        }
        // a new recovery set marks a new processing round (multi-pass
        // reuse) - clear the end-input latch
        self.ended = false;
        Ok(())
    }

    fn free_mem(&mut self) -> Result<()> {
        // C++ PAR2Proc::freeProcessingMem propagates to every backend -
        // like close(), keep going after a failure and surface the first
        let mut first: Result<()> = Ok(());
        for b in &mut self.backends {
            if let Err(e) = b.be.free_mem() {
                if first.is_ok() {
                    first = Err(e);
                }
            }
        }
        first
    }

    fn add_input(&mut self, buf: SliceBuf, input_index: i32, token: u64) -> Result<AddResult> {
        if self.closed {
            return Err(BackendError::new("Cannot add input after proc closed"));
        }
        self.poll_progress();
        self.maybe_discard();
        let data = buf.as_bytes();
        let size = data.len();
        for b in &mut self.backends {
            if b.offset >= size {
                continue;
            }
            let amount = (size - b.offset).min(b.slice_size);
            if amount == 0 {
                continue;
            }
            if b.added.contains(&(input_index as u32)) {
                continue;
            }
            if b.be.can_add() == CanAdd::Full {
                return Ok(AddResult::Full(buf));
            }
            // keep `added` marks on error too: a retry with the same buffer
            // resumes mid-way, feeding each backend exactly once
            b.be.add_input(
                &data[b.offset..b.offset + amount],
                input_index as u16,
                false,
            )?;
            b.added.insert(input_index as u32);
        }
        self.has_added = true;
        for b in &mut self.backends {
            b.added.remove(&(input_index as u32));
        }
        self.emit(ProcEvent::InputDone {
            token,
            input_index: input_index as i64,
            buf,
        });
        Ok(AddResult::Accepted)
    }

    fn end_input(&mut self, token: u64) -> Result<()> {
        if self.closed {
            return Err(BackendError::new("Cannot end input after proc closed"));
        }
        self.maybe_discard();
        for b in &mut self.backends {
            if b.slice_size == 0 {
                continue;
            }
            b.be.end_input()?;
        }
        self.needs_discard = true;
        self.ended = true;
        self.emit(ProcEvent::Ended { token });
        Ok(())
    }

    fn get_output(&mut self, output_index: i32, out: SliceBuf, token: u64) -> Result<()> {
        if self.closed {
            return Err(BackendError::new("Cannot get output after proc closed"));
        }
        let mut out = out;
        let buf = out.as_bytes_mut();
        if !self.has_added {
            buf.fill(0);
            self.emit(ProcEvent::OutputDone {
                token,
                cksum_valid: true,
                buf: out,
            });
            return Ok(());
        }
        let need = self
            .backends
            .iter()
            .map(|b| b.offset + b.slice_size)
            .max()
            .unwrap_or(0);
        if buf.len() < need {
            return Err(BackendError::new("Output buffer too small"));
        }
        let mut valid = true;
        for b in &mut self.backends {
            if b.slice_size == 0 {
                continue;
            }
            let out_ptr = &mut buf[b.offset..b.offset + b.slice_size];
            if !b.be.has_added() {
                out_ptr.fill(0);
            } else {
                valid &= b.be.get_output(output_index as usize, out_ptr)?;
            }
        }
        self.emit(ProcEvent::OutputDone {
            token,
            cksum_valid: valid,
            buf: out,
        });
        Ok(())
    }

    fn set_progress(&mut self, on: bool) -> Result<()> {
        self.progress_on = on;
        Ok(())
    }

    fn info(&mut self) -> Result<GfInfo> {
        let mut merged = GfInfo {
            cpu: None,
            ocl: Vec::new(),
        };
        for b in &mut self.backends {
            let i = b.be.info();
            if merged.cpu.is_none() {
                merged.cpu = i.cpu;
            }
            merged.ocl.extend(i.ocl);
        }
        Ok(merged)
    }

    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        // deinit every backend even when one fails - skipping siblings
        // would leak them, and closed=true already marks the proc shut
        let mut first_err: Result<()> = Ok(());
        for b in &mut self.backends {
            if let Err(e) = b.be.deinit() {
                if first_err.is_ok() {
                    first_err = Err(e);
                }
            }
        }
        first_err
    }
}

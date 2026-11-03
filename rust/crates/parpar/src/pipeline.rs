//! Port of the `lib/par2gen.js` generation driver (lines ~1140-1565):
//! `run`/`runPass`/`runChunkPass`/`_readPass`/`finish`/`writeFile(s)`/
//! `_initOutputFiles`/`finishPass`/`_setSlices`/`freeMemory`.
//!
//! JS callback chains become sequential loops; `cbProgress` becomes the
//! `Progress` enum sink. The chunked path (`_chunker`) uses
//! `Par2::start_chunking`, matching JS's PAR2Chunked.

use crate::fileinfo::FileInfo;
use crate::generate::{PacketType, Par2Gen};
use crate::outfile::Par2OutFile;
use crate::par2::{GfCtx, RecoveryGen};
use crate::readers::{file_chunk_reader, BufferPool, FileSeqReader};
use crate::throttle::ThrottleQueue;
use parpar_backend::SliceBuf;

pub type JsResult<T> = Result<T, String>;

const RECOVERY_HEADER_SIZE: u64 = 68;

/// `cbProgress` events - parpar.js consumes these by name.
#[derive(Debug, Clone)]
pub enum Progress {
    /// 'begin_pass' (passNum)
    BeginPass(u32),
    /// 'begin_chunk_pass' (passNum, passChunkNum, numChunksThisChunkPass)
    BeginChunkPass(u32, u32, u64),
    /// 'processing_slice' (file index, sliceNum)
    ProcessingSlice(usize, u64),
    /// 'processing_slice_part' (fraction)
    ProcessingSlicePart(f64),
    /// 'chunk_pass_write' (passNum, passChunkNum)
    ChunkPassWrite(u32, u32),
    /// 'pass_write' (passNum)
    PassWrite(u32),
    /// 'chunk_pass_complete' (passNum, passChunkNum)
    ChunkPassComplete(u32, u32),
    /// 'pass_complete' (passNum)
    PassComplete(u32),
    /// 'writing_file' (file index, name, recoverySlices, totalSize)
    WritingFile(usize, String, i64, u64),
    /// 'writing_file_pos' (file index, pos)
    WritingFilePos(usize, u64),
    /// 'closing_files'
    ClosingFiles,
    /// `gf_info` payload emitted once before the first begin_chunk_pass
    /// (JS prints compute_info inside that event's handler).
    ComputeInfo(parpar_backend::GfInfo),
}

/// Generation driver - JS `this.passNum`/`passChunkNum`/`sliceOffset`/
/// `chunkOffset`/`_slicesPerPass`/`_chunker`/`_buf`.
pub struct GenDriver<'a, C: GfCtx> {
    pub gen: &'a mut Par2Gen<C>,
    pub pass_num: u32,
    pub pass_chunk_num: u32,
    /// recovery slices already generated across passes
    pub slice_offset: u64,
    /// byte offset of the current chunk pass within the slice
    pub chunk_offset: u64,
    pub slices_per_pass: u64,
    /// `_chunker` - active PAR2Chunked while chunk_size < slice_size
    pub chunker: Option<RecoveryGen<C>>,
    /// `_buf` - shared read-buffer pool
    pub buf_pool: Vec<Vec<u8>>,
    /// `_slicesPerPass` divisor (JS `this.passes`)
    pub passes: u32,
    /// `PAR2OutFile` objects built from the solver's `recovery_files`
    pub out_files: Vec<Par2OutFile>,
    /// `this.seqReadThrottle` / `this.chunkReadThrottle`
    pub seq_read_throttle: Option<ThrottleQueue>,
    pub chunk_read_throttle: Option<ThrottleQueue>,
    /// JS `par2.recHashPtr` - cycles through chunked recovery headers
    pub rec_hash_ptr: usize,
}

impl<'a, C: GfCtx + Clone> GenDriver<'a, C> {
    /// JS: the driver is constructed inside `PAR2Gen`; files' hashers are
    /// created eagerly (PAR2File ctor `new binding.HasherInput`).
    pub fn new(gen: &'a mut Par2Gen<C>) -> JsResult<Self> {
        let passes = gen.passes.max(1.0) as u32;
        let out_files = gen
            .recovery_files
            .iter()
            .map(|rf| {
                Par2OutFile::new(
                    rf.name.clone(),
                    rf.recovery_slices,
                    rf.recovery_index,
                    rf.packets.clone(),
                    rf.total_size,
                )
            })
            .collect();
        // PAR2File ctor: `new binding.HasherInput(par2.sliceSize, ...)`
        // for every non-empty file with pktCheck
        let slice_size = gen.par2.info.slice_size;
        let seq_thr = gen.opts.seq_read_throttle.clone();
        let chunk_thr = gen.opts.chunk_read_throttle.clone();
        let ctx = &mut gen.par2.gen.ctx;
        for f in gen.par2.files.iter_mut() {
            f.init_hasher(ctx, slice_size).map_err(|e| e.to_string())?;
        }
        Ok(GenDriver {
            gen,
            pass_num: 0,
            pass_chunk_num: 0,
            slice_offset: 0,
            chunk_offset: 0,
            slices_per_pass: 0,
            chunker: None,
            buf_pool: Vec::new(),
            passes,
            out_files,
            rec_hash_ptr: 0,
            seq_read_throttle: seq_thr.as_ref().and_then(ThrottleQueue::from_spec),
            chunk_read_throttle: chunk_thr.as_ref().and_then(ThrottleQueue::from_spec),
        })
    }

    /// JS `_setSlices(offset)`.
    fn set_slices(&mut self, offset: u64) -> JsResult<()> {
        let remaining = self.gen.opts.recovery_slices_res - offset as f64;
        self.slices_per_pass = (remaining / (self.passes - self.pass_num) as f64).ceil() as u64;
        if self.slices_per_pass == 0 {
            return Ok(()); // reached end
        }
        let end = (offset + self.slices_per_pass) as usize;
        // C++ setRecoverySlices rejects indices > 65534; JS Uint32
        // conversion turns negatives into huge values which hit the same
        // check - validate rather than wrap with `as u16`
        let slices: Vec<u16> = self.gen.slice_nums
            [offset as usize..end.min(self.gen.slice_nums.len())]
            .iter()
            .map(|s| {
                let v = s.unwrap_or(0);
                if !(0..=65534).contains(&v) {
                    return Err("Invalid recovery index supplied".to_string());
                }
                Ok(v as u16)
            })
            .collect::<Result<Vec<u16>, String>>()?;
        if (self.gen.chunk_size as u64) < self.gen.opts.slice_size as u64 {
            self.gen
                .par2
                .gen
                .set_recovery_slices(Vec::new())
                .map_err(|e| e.to_string())?;
            match &mut self.chunker {
                Some(ch) => ch.set_recovery_slices(slices).map_err(|e| e.to_string())?,
                None => {
                    self.chunker = Some(self.gen.par2.start_chunking(slices)?);
                }
            }
            let cs = self.gen.chunk_size as u64;
            self.chunker
                .as_mut()
                .unwrap()
                .set_chunk_size(cs)
                .map_err(|e| e.to_string())?;
        } else {
            if let Some(mut ch) = self.chunker.take() {
                ch.set_recovery_slices(Vec::new())
                    .map_err(|e| e.to_string())?;
                ch.close().map_err(|e| e.to_string())?;
            }
            self.gen
                .par2
                .gen
                .set_recovery_slices(slices)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// JS `_initOutputFiles`.
    fn init_output_files(&mut self) -> JsResult<()> {
        let mut slice_offset = 0i64;
        let overwrite = self.gen.opts.output_overwrite;
        for rf in &mut self.out_files {
            slice_offset += rf.recovery_slices;
            rf.open(overwrite)?;
            if rf.recovery_slices != 0
                && (self.chunker.is_some() || slice_offset > self.slices_per_pass as i64)
            {
                rf.prealloc()?;
            }
        }
        Ok(())
    }

    /// JS `_getCriticalPackets` + the finish() attachment of non-recovery
    /// packet data (first pass only).
    fn attach_critical_packets(&mut self) -> JsResult<()> {
        let unicode = self.gen.unicode_opt();
        let creator = self.gen.opts.creator.clone();
        let comments = self.gen.opts.comments.clone();

        // build the critical packets via the par2 object
        let mut crit: Vec<(PacketType, i64, Vec<u8>)> = Vec::new();
        {
            let info = &self.gen.par2.info;
            let files = &mut self.gen.par2.files;
            for (i, f) in files.iter_mut().enumerate() {
                let d = f.make_packet_description(info, unicode)?;
                crit.push((PacketType::FileDesc, i as i64, d));
                if f.size > 0 {
                    let c = f.get_packet_checksums(info, false);
                    crit.push((PacketType::FileChk, i as i64, c));
                }
            }
        }
        for (i, c) in comments.iter().enumerate() {
            let p = self.gen.par2.make_packet_comment(c, unicode)?;
            crit.push((PacketType::Comment, i as i64, p));
        }
        let main = self.gen.par2.get_packet_main(false);
        crit.push((PacketType::Main, -1, main));
        let cre = self.gen.par2.make_packet_creator(&creator);
        crit.push((PacketType::Creator, -1, cre));

        for rf in &mut self.out_files {
            for pkt in &mut rf.packets {
                if pkt.pkt_type != PacketType::Recovery {
                    if let Some((_, _, d)) = crit
                        .iter()
                        .find(|(t, i, _)| *t == pkt.pkt_type && (pkt.index < 0 || *i == pkt.index))
                    {
                        pkt.set_data(vec![d.clone()], 0);
                    }
                }
            }
        }
        Ok(())
    }

    /// JS `finish(cb)` - resets file positions + gf.end (both via
    /// `finish_files`); first pass attaches critical packets.
    fn finish(&mut self, first: bool) -> JsResult<()> {
        match &mut self.chunker {
            Some(ch) => ch
                .finish_files(&mut self.gen.par2.files)
                .map_err(|e| e.to_string())?,
            None => self
                .gen
                .par2
                .gen
                .finish_files(&mut self.gen.par2.files)
                .map_err(|e| e.to_string())?,
        }
        if first {
            self.attach_critical_packets()?;
        }
        Ok(())
    }

    /// `(idx, data, header)` pull from the active gen. Chunked mode: data
    /// only (JS `pkt.setData(data.data, chunkOffset + HEADER)`); whole-slice
    /// mode: `[header, data]` at offset 0.
    fn next_recovery_data(&mut self) -> JsResult<(usize, Vec<u8>, Vec<u8>)> {
        match &mut self.chunker {
            Some(ch) => {
                let idx = ch
                    .next_recovery()
                    .map_err(|e| e.to_string())?
                    .ok_or("Recovery data index mismatch")?;
                let d = ch.recovery_data(idx).map_err(|e| e.to_string())?.to_vec();
                Ok((idx, d, Vec::new()))
            }
            None => {
                let g = &mut self.gen.par2.gen;
                let idx = g
                    .next_recovery()
                    .map_err(|e| e.to_string())?
                    .ok_or("Recovery data index mismatch")?;
                let d = g.recovery_data(idx).map_err(|e| e.to_string())?.to_vec();
                let md5 = g.recovery_md5(idx).map_err(|e| e.to_string())?;
                let exp = g.recovery_slices[idx];
                let hdr = self
                    .gen
                    .par2
                    .info
                    .recovery_header(exp, &md5, self.gen.par2.info.slice_size)
                    .to_vec();
                Ok((idx, d, hdr))
            }
        }
    }

    /// mark a pulled recovery index consumed (JS `data.release()`).
    fn release_recovery(&mut self, idx: usize) -> JsResult<()> {
        match &mut self.chunker {
            Some(ch) => ch.mark_consumed(idx).map_err(|e| e.to_string()),
            None => self
                .gen
                .par2
                .gen
                .mark_consumed(idx)
                .map_err(|e| e.to_string()),
        }
    }

    /// JS `writeFile(rf, cbProgress, cb)`.
    fn write_file(&mut self, rfi: usize, cb_progress: &mut dyn FnMut(Progress)) -> JsResult<()> {
        let mut cpos = 0u64;
        let rf_index = self.out_files[rfi].recovery_index;
        let mut rel_rec_idx = rf_index - self.slice_offset as i64 - 1;
        let end = (self.slice_offset + self.slices_per_pass) as usize;
        let slices: Vec<i64> = self.gen.slice_nums
            [self.slice_offset as usize..end.min(self.gen.slice_nums.len())]
            .iter()
            .map(|s| s.unwrap_or(0))
            .collect();
        let chunk_offset = self.chunk_offset;
        let overwrite = self.gen.opts.output_overwrite;

        for pkt_i in 0..self.out_files[rfi].packets.len() {
            let (ptype, pindex, has_data) = {
                let p = &self.out_files[rfi].packets[pkt_i];
                (p.pkt_type, p.index, !p.data.is_empty())
            };
            if ptype == PacketType::Recovery {
                rel_rec_idx += 1;
            }
            let mut rec_idx: Option<usize> = None;
            if !has_data {
                if ptype != PacketType::Recovery
                    || rel_rec_idx < 0
                    || rel_rec_idx >= self.slices_per_pass as i64
                {
                    // JS `cb(true)` - skip packet entirely
                    cpos += self.out_files[rfi].packets[pkt_i].size;
                    continue;
                }
                let (idx2, data, header) = self.next_recovery_data()?;
                let idx2 = idx2;
                if pindex != slices[idx2] {
                    return Err(format!(
                        "Recovery data index mismatch ({}<>{})",
                        pindex, slices[idx2]
                    ));
                }
                let pkt = &mut self.out_files[rfi].packets[pkt_i];
                if self.chunker.is_some() {
                    pkt.set_data(vec![data], chunk_offset + RECOVERY_HEADER_SIZE);
                } else {
                    pkt.set_data(vec![header, data], 0);
                }
                rec_idx = Some(idx2);
            }
            self.out_files[rfi].open(overwrite)?;
            let pos = cpos + self.out_files[rfi].packets[pkt_i].data_chunk_offset;
            cb_progress(Progress::WritingFilePos(rfi, pos));
            self.out_files[rfi].write_packets(pkt_i, cpos)?;
            if let Some(i) = rec_idx {
                self.release_recovery(i)?;
            }
            cpos += self.out_files[rfi].packets[pkt_i].size;
        }
        Ok(())
    }

    /// JS `writeFiles`.
    fn write_files(&mut self, cb_progress: &mut dyn FnMut(Progress)) -> JsResult<()> {
        for rfi in 0..self.out_files.len() {
            let (name, rs, ts) = {
                let rf = &self.out_files[rfi];
                (rf.name.clone(), rf.recovery_slices, rf.total_size)
            };
            cb_progress(Progress::WritingFile(rfi, name, rs, ts));
            self.write_file(rfi, cb_progress)?;
        }
        Ok(())
    }

    /// JS `g.gf_info()` - info for the active proc (chunker if present).
    pub fn gf_info(&mut self) -> JsResult<parpar_backend::GfInfo> {
        match &mut self.chunker {
            Some(ch) => ch.gf_info().map_err(|e| e.to_string()),
            None => self.gen.par2.gen.gf_info().map_err(|e| e.to_string()),
        }
    }

    /// JS `closeFiles`.
    fn close_files(&mut self) -> JsResult<()> {
        let sync = self.gen.opts.output_sync;
        for rf in &mut self.out_files {
            rf.close(sync)?;
        }
        Ok(())
    }

    /// JS `discardData`.
    #[allow(dead_code)]
    fn discard_data(&mut self) {
        for rf in &mut self.out_files {
            rf.discard_data();
        }
    }

    /// JS `freeMemory`.
    fn free_memory(&mut self) -> JsResult<()> {
        if let Some(mut ch) = self.chunker.take() {
            ch.set_recovery_slices(Vec::new())
                .map_err(|e| e.to_string())?;
            ch.close().map_err(|e| e.to_string())?;
        }
        self.gen
            .par2
            .gen
            .set_recovery_slices(Vec::new())
            .map_err(|e| e.to_string())?;
        self.gen.par2.gen.close().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// JS `numChunksThisChunkPass`.
    pub fn num_chunks_this_pass(&self) -> u64 {
        let ss = self.gen.opts.slice_size as u64;
        self.gen
            .par2
            .files
            .iter()
            .filter(|f| !(f.size == 0 || f.size <= self.chunk_offset))
            .map(|f| (f.size - self.chunk_offset + ss - 1) / ss)
            .sum()
    }

    /// JS `totalNumInputChunks`.
    pub fn total_num_input_chunks(&self) -> u64 {
        let ss = self.gen.opts.slice_size as u64;
        let cs = self.gen.chunk_size as u64;
        let chunks = self.gen.chunks;
        self.gen
            .par2
            .files
            .iter()
            .filter(|f| f.size > 0)
            .map(|f| {
                let non_last = f.num_slices - 1;
                let last = f.size - non_last * ss;
                (non_last as f64 * chunks) as u64 + (last + cs - 1) / cs
            })
            .sum()
    }

    /// JS `runChunkPass`.
    fn run_chunk_pass(&mut self, cb_progress: &mut dyn FnMut(Progress)) -> JsResult<()> {
        let first = self.pass_num == 0 && self.pass_chunk_num == 0;
        let mut chunk_size = self.gen.opts.slice_size as u64;
        if let Some(ch) = &mut self.chunker {
            chunk_size = (self.gen.chunk_size as u64)
                .min(self.gen.opts.slice_size as u64 - self.chunk_offset);
            if chunk_size != ch.chunk_size {
                ch.set_chunk_size(chunk_size).map_err(|e| e.to_string())?;
            }
        }
        if first {
            // JS: compute info fetched inside the first begin_chunk_pass
            // event handler; emit before the event itself.
            if let Ok(info) = self.gf_info() {
                cb_progress(Progress::ComputeInfo(info));
            }
        }
        cb_progress(Progress::BeginChunkPass(
            self.pass_num,
            self.pass_chunk_num,
            self.num_chunks_this_pass(),
        ));

        self.read_pass(chunk_size, first, cb_progress)?;
        if first {
            self.init_output_files()?;
        }
        cb_progress(Progress::ChunkPassWrite(self.pass_num, self.pass_chunk_num));
        self.finish(first)?;
        self.write_files(cb_progress)?;
        cb_progress(Progress::ChunkPassComplete(
            self.pass_num,
            self.pass_chunk_num,
        ));

        self.pass_chunk_num += 1;
        self.chunk_offset += chunk_size;
        if let Some(ch) = &mut self.chunker {
            ch.wait_recovery_complete().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// JS `runPass`.
    fn run_pass(&mut self, cb_progress: &mut dyn FnMut(Progress)) -> JsResult<()> {
        cb_progress(Progress::BeginPass(self.pass_num));
        self.set_slices(self.slice_offset)?;
        while self.chunk_offset < self.gen.opts.slice_size as u64 {
            self.run_chunk_pass(cb_progress)?;
        }
        self.finish_pass(cb_progress)
    }

    /// JS `finishPass` - chunked: pull + write packet headers, writeFiles.
    fn finish_pass(&mut self, cb_progress: &mut dyn FnMut(Progress)) -> JsResult<()> {
        if self.chunker.is_some() {
            cb_progress(Progress::PassWrite(self.pass_num));
            let end = (self.slice_offset + self.slices_per_pass) as usize;
            let slices: Vec<i64> = self.gen.slice_nums
                [self.slice_offset as usize..end.min(self.gen.slice_nums.len())]
                .iter()
                .map(|s| s.unwrap_or(0))
                .collect();
            let range =
                self.traverse_recovery_packet_range(self.slice_offset, self.slices_per_pass);
            for (rfi, pkt_i, _rel) in range {
                // JS `getNextRecoveryPacketHeader` - recHashPtr cycles on
                // the main par2; MD5 + header come from the chunker. No
                // data pull / release: slots drained in the chunk passes.
                let (idx, hdr) = {
                    let ch = self.chunker.as_mut().unwrap();
                    let idx = self.rec_hash_ptr;
                    self.rec_hash_ptr += 1;
                    let hdr = ch
                        .recovery_packet_header(idx, self.gen.par2.info.slice_size)
                        .map_err(|e| e.to_string())?;
                    if self.rec_hash_ptr >= ch.recovery_slices.len() {
                        self.rec_hash_ptr = 0;
                    }
                    (idx, hdr.to_vec())
                };
                let pkt = &mut self.out_files[rfi].packets[pkt_i];
                if pkt.index != slices[idx] {
                    return Err(format!(
                        "Packet index mismatch ({}<>{})",
                        pkt.index, slices[idx]
                    ));
                }
                pkt.set_data(vec![hdr], 0);
            }
            self.write_files(&mut |_| {})?;
        }
        cb_progress(Progress::PassComplete(self.pass_num));
        self.slice_offset += self.slices_per_pass;
        self.pass_num += 1;
        self.chunk_offset = 0;
        self.pass_chunk_num = 0;
        Ok(())
    }

    /// JS `_traverseRecoveryPacketRange` -> (rfi, pktI, relIdx) of recovery
    /// packets with global idx in [sliceOffset, +numSlices).
    fn traverse_recovery_packet_range(
        &self,
        slice_offset: u64,
        num_slices: u64,
    ) -> Vec<(usize, usize, u64)> {
        let end = slice_offset + num_slices;
        let mut out = Vec::new();
        let mut rf_base = 0u64;
        for (rfi, rf) in self.out_files.iter().enumerate() {
            for (pi, pkt) in rf.packets.iter().enumerate() {
                if pkt.pkt_type != PacketType::Recovery {
                    continue;
                }
                let i = rf_base;
                rf_base += 1;
                if i >= slice_offset && i < end {
                    out.push((rfi, pi, i - slice_offset));
                }
            }
        }
        out
    }

    /// JS `_readPass`.
    fn read_pass(
        &mut self,
        chunk_size: u64,
        first: bool,
        cb_progress: &mut dyn FnMut(Progress),
    ) -> JsResult<()> {
        let slice_size = self.gen.opts.slice_size as u64;
        let seeking = chunk_size != slice_size && !first;
        if seeking {
            if self.chunker.is_none() {
                return Err("Trying to perform chunked reads without a chunker".into());
            }
            let pool = BufferPool::new(
                std::mem::take(&mut self.buf_pool),
                chunk_size as usize,
                self.gen.opts.read_buffers as usize,
            );
            let files: Vec<FileInfo> = self.gen.par2.files.iter().map(|f| reader_file(f)).collect();
            let chunk_offset = self.chunk_offset;
            let mut err: JsResult<()> = Ok(());
            {
                let r = file_chunk_reader(
                    &files,
                    slice_size,
                    chunk_size,
                    chunk_offset,
                    &pool,
                    self.gen.opts.chunk_read_threads as usize,
                    self.chunk_read_throttle.as_ref(),
                    |_file, fidx, data, slice_num| {
                        // files[] mirrors gen.par2.files order, so the index
                        // identifies the Par2File even for duplicate names
                        cb_progress(Progress::ProcessingSlice(fidx, slice_num));
                        let off = self.gen.par2.files[fidx].slice_offset + slice_num;
                        let ch = self.chunker.as_mut().unwrap();
                        let r = ch.process_data(
                            off as i64,
                            SliceBuf::new(data.to_vec().into_boxed_slice(), data.len()),
                        );
                        if let Err(e) = r {
                            err = Err(e.to_string());
                        }
                    },
                );
                // return pooled buffers to the shared pool for the next
                // pass (JS keeps one this._buf throughout)
                self.buf_pool = pool.drain();
                r?;
            }
            err?;
            return Ok(());
        }

        // sequential read path
        let files: Vec<FileInfo> = self.gen.par2.files.iter().map(|f| reader_file(f)).collect();
        let mut reader = FileSeqReader::new(
            files,
            self.gen.read_size as u64,
            self.gen.opts.read_buffers as usize,
        );
        reader.set_buffers(std::mem::take(&mut self.buf_pool));
        reader.set_max_queue_per_file(self.gen.opts.read_hash_queue as u32);

        let mut slices_per_read = 0u64;
        if self.chunker.is_some() {
            reader.require_chunk(slice_size, chunk_size)?;
        } else if self.gen.opts.recovery_slices_res > 0.0 {
            let spr = self.gen.read_size / self.gen.opts.slice_size;
            if spr != spr.floor() {
                return Err(format!(
                    "Expected read size ({}) to be a multiple of slice size ({})",
                    self.gen.read_size, self.gen.opts.slice_size
                ));
            }
            slices_per_read = spr as u64;
        }

        let mut res: JsResult<()> = Ok(());
        let run_res = reader.run(self.seq_read_throttle.as_ref(), |r| {
            match r {
                Ok(mut data) => {
                    let fidx = data.file_index;
                    if first {
                        let buf = data.buffer.clone();
                        let f = &mut self.gen.par2.files[fidx];
                        if let Err(e) = f.process_hash(&buf) {
                            res = Err(e.to_string());
                            return;
                        }
                    }
                    data.hashed();

                    if self.gen.opts.recovery_slices_res < 1.0 {
                        let to =
                            (data.pos + data.buffer.len() as u64) as f64
                                / self.gen.opts.slice_size;
                        let to_base = to.floor();
                        let mut sp =
                            (data.pos as f64 / self.gen.opts.slice_size).floor();
                        while sp < to_base {
                            cb_progress(Progress::ProcessingSlice(fidx, sp as u64));
                            sp += 1.0;
                        }
                        cb_progress(Progress::ProcessingSlicePart(to - to_base));
                        data.release();
                        return;
                    }

                    if self.chunker.is_some() {
                        let mut slice_pos: Option<u64> = None;
                        let chunks = data.chunks.clone().unwrap_or_default();
                        let fsize = self.gen.par2.files[fidx].size;
                        let fslice_off =
                            self.gen.par2.files[fidx].slice_offset;
                        for ch in &chunks {
                            let sp = match slice_pos {
                                None => {
                                    if (data.pos + ch) % slice_size != 0 {
                                        res = Err(format!(
                                            "Unexpected position not a multiple of slice size (got data at position {}, expected multiple of {})",
                                            data.pos + ch, slice_size
                                        ));
                                        return;
                                    }
                                    (data.pos + ch) / slice_size
                                }
                                Some(v) => v + 1,
                            };
                            slice_pos = Some(sp);
                            cb_progress(Progress::ProcessingSlice(fidx, sp));
                            let end = (*ch + chunk_size).min(fsize) as usize;
                            let end = end.min(data.buffer.len());
                            let buf =
                                data.buffer[*ch as usize..end].to_vec();
                            let c = self.chunker.as_mut().unwrap();
                            let r = c.process_data(
                                (fslice_off + sp) as i64,
                                SliceBuf::new(buf.clone().into_boxed_slice(), buf.len()),
                            );
                            if let Err(e) = r {
                                res = Err(e.to_string());
                                return;
                            }
                        }
                        data.release();
                    } else {
                        if data.pos % slice_size != 0 {
                            res = Err(format!(
                                "Unexpected position not a multiple of slice size (got data at position {}, expected multiple of {})",
                                data.pos, slice_size
                            ));
                            return;
                        }
                        let slice_pos = data.pos / slice_size;
                        let num_slices =
                            (data.buffer.len() as u64 + slice_size - 1)
                                / slice_size;
                        let file_ns = self.gen.par2.files[fidx].num_slices;
                        // JS Math.min allows negative; saturate so a
                        // pos-past-EOF still lands on the error path
                        // rather than wrapping or panicking
                        let expected = file_ns.saturating_sub(slice_pos).min(slices_per_read);
                        if num_slices != expected {
                            res = Err(format!(
                                "Data read failure: read {} bytes ({} slices) but expected {} slices",
                                data.buffer.len(), num_slices, expected
                            ));
                            return;
                        }
                        for i in 0..num_slices {
                            cb_progress(
                                Progress::ProcessingSlice(fidx, slice_pos + i),
                            );
                            let bp = (i * slice_size) as usize;
                            let end =
                                (bp + slice_size as usize).min(data.buffer.len());
                            let slice = data.buffer[bp..end].to_vec();
                            let len = slice.len();
                            let buf =
                                SliceBuf::new(slice.into_boxed_slice(), len);
                            let f = &mut self.gen.par2.files[fidx];
                            let gen = &mut self.gen.par2.gen;
                            if let Err(e) = f.process_data(gen, buf) {
                                res = Err(e.to_string());
                                return;
                            }
                        }
                        data.release();
                    }
                }
                Err(e) => res = Err(e),
            }
        });
        // return pooled read buffers to the shared pool for the next pass
        // (JS shares this._buf); run() only finishes once all lent buffers
        // are back
        self.buf_pool = reader.take_pooled();
        run_res?;
        res
    }

    /// JS `run(cbProgress, cb)`.
    pub fn run(&mut self, cb_progress: &mut dyn FnMut(Progress)) -> JsResult<()> {
        let mut result: JsResult<()> = Ok(());
        // `whilst(sliceOffset < recoverySlices || first pass)` - at least
        // one pass always runs
        while (self.slice_offset as f64) < self.gen.opts.recovery_slices_res
            || (self.pass_num == 0 && self.pass_chunk_num == 0)
        {
            match self.run_pass(cb_progress) {
                Err(e) => {
                    result = Err(e);
                    break;
                }
                Ok(()) => {}
            }
        }
        if result.is_ok() {
            if let Err(e) = self.free_memory() {
                result = Err(e);
            }
        }
        cb_progress(Progress::ClosingFiles);
        match self.close_files() {
            Ok(()) => result,
            Err(e2) => {
                if result.is_err() {
                    result
                } else {
                    Err(e2)
                }
            }
        }
    }
}

/// Reader-side file view of a Par2File (name + size only).
fn reader_file(f: &crate::par2::Par2File) -> FileInfo {
    FileInfo {
        name: f.name.clone(),
        fs_path: f.fs_path.clone(),
        display_name: None,
        full_path: None,
        size: f.size,
        md5_16k: [0u8; 16],
        md5: None,
    }
}

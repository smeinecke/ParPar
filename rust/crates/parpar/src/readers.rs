//! Ports of `lib/bufferpool.js`, `lib/procqueue.js`,
//! `lib/filechunkreader.js` and `lib/fileseqreader.js`.
//!
//! The JS modules are callback-based over libuv I/O. The Rust pipeline is
//! synchronous, so reads run inline and `cb_chunk`/`cb` are plain
//! callbacks. `concurrency` in FileChunkReader only affects I/O overlap
//! (JS delivery order is nondeterministic), so reads here are sequential
//! while preserving per-file slice enumeration and buffer-pool semantics.
//! FileSeqReader keeps the shortest-hash-queue file preference and
//! `requireChunk` read sizing.

use std::collections::VecDeque;
use std::fs::File;
use std::io::ErrorKind;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::fileinfo::FileInfo;
use crate::throttle::ThrottleQueue;

pub type JsResult<T> = Result<T, String>;

/// `lib/bufferpool.js` - stack of buffers with a cap; `get` blocks (JS
/// queued the callback) when `pool_size == max_bufs` until a `put`.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    state: Mutex<PoolState>,
    cv: Condvar,
}

struct PoolState {
    pool: Vec<Vec<u8>>,
    /// JS `poolSize` - allocated bufs, including those lent out.
    pool_size: usize,
    length: usize,
    max_bufs: usize,
}

impl BufferPool {
    pub fn new(bufs: Vec<Vec<u8>>, length: usize, max_bufs: usize) -> Self {
        let pool_size = bufs.len();
        BufferPool {
            inner: Arc::new(PoolInner {
                state: Mutex::new(PoolState {
                    pool: bufs,
                    pool_size,
                    length,
                    max_bufs,
                }),
                cv: Condvar::new(),
            }),
        }
    }

    /// JS `get(cb)` - pops a reusable buffer >= length, or allocates one.
    /// Blocks while the pool is exhausted at `max_bufs`.
    pub fn get(&self) -> Vec<u8> {
        let mut st = self.inner.state.lock().unwrap();
        loop {
            while let Some(buf) = st.pool.pop() {
                if buf.len() >= st.length {
                    return buf;
                }
                st.pool_size -= 1; // too small - discard
            }
            if st.pool_size < st.max_bufs {
                st.pool_size += 1;
                return vec![0u8; st.length];
            }
            st = self.inner.cv.wait(st).unwrap();
        }
    }

    /// JS `put(buffer)`. notify_all: a single notify could wake an `end`
    /// waiter that re-checks and sleeps while a `get` waiter misses the
    /// wakeup and starves with a buffer available
    pub fn put(&self, buffer: Vec<u8>) {
        let mut st = self.inner.state.lock().unwrap();
        st.pool.push(buffer);
        self.inner.cv.notify_all();
    }

    /// JS `end(cb)` - returns when every allocated buffer is back (caller
    /// waits on outstanding lends).
    pub fn end(&self) {
        let mut st = self.inner.state.lock().unwrap();
        while st.pool.len() != st.pool_size {
            st = self.inner.cv.wait(st).unwrap();
        }
    }

    /// buffers currently in the pool (JS `pool.length`)
    pub fn pooled(&self) -> usize {
        self.inner.state.lock().unwrap().pool.len()
    }

    /// Drain all currently-pooled buffers (for recycling into the next
    /// pass's pool - JS keeps one shared `this._buf` across passes).
    pub fn drain(&self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.inner.state.lock().unwrap().pool)
    }
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], pos: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    loop {
        match f.read_at(buf, pos) {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            r => return r,
        }
    }
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], pos: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    loop {
        match f.seek_read(buf, pos) {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            r => return r,
        }
    }
}

/// `lib/filechunkreader.js` - per-file slice reads: `cb_chunk(file, data,
/// slice_num)` per chunk. Files iterated in order; zero-size / offset-past
/// -size files skipped. Reads `chunkSize` at `sliceNum*sliceSize +
/// chunkOffset`; the final slice may be short.
pub fn file_chunk_reader(
    files: &[FileInfo],
    slice_size: u64,
    chunk_size: u64,
    chunk_offset: u64,
    pool: &BufferPool,
    _concurrency: usize,
    throttle: Option<&ThrottleQueue>,
    mut cb_chunk: impl FnMut(&FileInfo, usize, &[u8], u64),
) -> JsResult<()> {
    let no_throttle;
    let throttle = match throttle {
        Some(t) => t,
        None => {
            no_throttle = ThrottleQueue::none();
            &no_throttle
        }
    };
    let mut read_err: Option<String> = None;

    for (fidx, file) in files.iter().enumerate() {
        if file.size == 0 || file.size <= chunk_offset {
            continue;
        }
        let fd = match File::open(file.fs_path()) {
            Ok(f) => f,
            Err(e) => return Err(crate::compat::fs_io_error(&e, "open", &file.name)),
        };
        let num_slices = (file.size - chunk_offset + slice_size - 1) / slice_size;
        for s in 0..num_slices {
            if read_err.is_some() {
                break;
            }
            let pos = s * slice_size + chunk_offset;
            let (cancelled, _done) = throttle.pass(chunk_size as f64);
            if cancelled {
                continue; // JS `if(cancelled) return cb()` - skips the slice
            }
            let mut buf = pool.get();
            match read_at(&fd, &mut buf[..chunk_size as usize], pos) {
                Ok(n) => {
                    let data = &buf[..n];
                    // JS passes bufferSlice(0, bytesRead) - a view; we pass a
                    // slice and return the buf to the pool after the callback
                    cb_chunk(file, fidx, data, s);
                    pool.put(buf);
                }
                Err(e) => {
                    pool.put(buf);
                    read_err = Some(crate::compat::fs_io_error(&e, "read", &file.name));
                }
            }
        }
    }
    if let Some(e) = read_err {
        return Err(e);
    }
    Ok(())
}

/// `lib/fileseqreader.js` FileReaderData - a delivered read chunk.
/// `hashed()` + `release()` drop `_refs`; at 0 the buffer returns to the
/// reader's pool (JS `processed`).
pub struct FileReaderData {
    /// index into the reader's `files` (JS `file.info`)
    pub file_index: usize,
    /// the chunk data (`buffer.slice(0, len)` in JS; owned here - the pool
    /// gets an equivalent-length fresh buffer on release)
    pub buffer: Vec<u8>,
    /// position of this chunk in the file
    pub pos: u64,
    /// sub-chunk boundaries (reqChunk mode) - byte offsets within buffer
    pub chunks: Option<Vec<u64>>,
    /// identity of the OpenFile this chunk came from (stable across
    /// open_files reorder/removal - JS kept a reference to the object)
    reader_slot: u64,
    reader: Arc<SeqInner>,
    refs: i32,
    /// the hash-queue slot is released exactly once even if hashed() ends
    /// up running twice (explicit call + Drop)
    hash_pending: bool,
}

impl FileReaderData {
    /// JS `hashed()` - one hash queue slot drained.
    pub fn hashed(&mut self) {
        self.refs -= 1;
        let mut st = self.reader.state.lock().unwrap();
        if self.hash_pending {
            self.hash_pending = false;
            if let Some(f) = st
                .open_files
                .iter_mut()
                .find(|f| f.slot == self.reader_slot)
            {
                f.hash_queue = f.hash_queue.saturating_sub(1);
            }
        }
        if self.refs == 0 {
            let buf = std::mem::take(&mut self.buffer);
            st.buf.push(buf);
            self.reader.cv.notify_one();
        }
    }
    /// JS `release()` - buffer recycled once hashing also done.
    pub fn release(&mut self) {
        self.refs -= 1;
        if self.refs == 0 {
            let mut st = self.reader.state.lock().unwrap();
            let buf = std::mem::take(&mut self.buffer);
            st.buf.push(buf);
            self.reader.cv.notify_one();
        }
    }
}

impl Drop for FileReaderData {
    fn drop(&mut self) {
        if self.refs > 0 {
            self.hashed();
            self.release();
        }
    }
}

struct OpenFile {
    fd: File,
    info: usize,
    pos: u64,
    hash_queue: u32,
    /// stable identity for FileReaderData chunks - open_files is reordered
    /// and shrunk at EOF, so positional indices cannot be used
    slot: u64,
}

struct SeqState {
    buf: Vec<Vec<u8>>,
    buf_count: usize,
    open_files: Vec<OpenFile>,
    file_queue: VecDeque<usize>,
    next_slot: u64,
}

struct SeqInner {
    state: Mutex<SeqState>,
    cv: Condvar,
    read_size: u64,
    max_bufs: usize,
    max_queue_per_file: AtomicU32,
}

/// `lib/fileseqreader.js` - sequential reader preferring in-order file
/// progress with a bounded buffer pool and per-file hash queues.
pub struct FileSeqReader {
    files: Vec<FileInfo>,
    inner: Arc<SeqInner>,
    req_slice_len: u64,
    req_chunk_len: u64,
}

impl FileSeqReader {
    /// JS `new FileSeqReader(files, readSize, readBuffers, throttleQ)`;
    /// zero-size files are filtered out.
    pub fn new(files: Vec<FileInfo>, read_size: u64, read_buffers: usize) -> Self {
        let file_queue: VecDeque<usize> = (0..files.len()).filter(|&i| files[i].size > 0).collect();
        FileSeqReader {
            files,
            inner: Arc::new(SeqInner {
                state: Mutex::new(SeqState {
                    buf: Vec::new(),
                    buf_count: 0,
                    open_files: Vec::new(),
                    file_queue,
                    next_slot: 1,
                }),
                cv: Condvar::new(),
                read_size,
                max_bufs: read_buffers,
                max_queue_per_file: AtomicU32::new(5), // JS `maxQueuePerFile`
            }),
            req_slice_len: 0,
            req_chunk_len: 0,
        }
    }

    /// JS `reader.maxQueuePerFile = opts.readHashQueue`.
    pub fn set_max_queue_per_file(&mut self, v: u32) {
        self.inner.max_queue_per_file.store(v, Ordering::Relaxed);
    }

    /// JS `requireChunk(sliceLen, chunkLen)` - each returned buffer must
    /// contain whole `chunkLen` heads of each `sliceLen` slice.
    pub fn require_chunk(&mut self, slice_len: u64, chunk_len: u64) -> JsResult<()> {
        if chunk_len > self.inner.read_size {
            return Err("Required chunk length cannot exceed maximum read length".into());
        }
        self.req_slice_len = slice_len;
        self.req_chunk_len = chunk_len;
        Ok(())
    }

    /// JS `setBuffers(bufs)` - external buffers replace the pool.
    pub fn set_buffers(&mut self, bufs: Vec<Vec<u8>>) {
        let mut st = self.inner.state.lock().unwrap();
        st.buf_count = bufs.len();
        st.buf = bufs;
    }

    /// Take back the pooled read buffers after a pass (JS shares
    /// `this._buf` between passes instead of reallocating).
    pub fn take_pooled(&mut self) -> Vec<Vec<u8>> {
        let mut st = self.inner.state.lock().unwrap();
        let bufs = std::mem::take(&mut st.buf);
        st.buf_count -= st.buf_count.min(bufs.len());
        bufs
    }

    /// JS `_getBuf`.
    fn get_buf(&self, st: &mut SeqState) -> Option<Vec<u8>> {
        while let Some(buf) = st.buf.pop() {
            if buf.len() as u64 >= self.inner.read_size {
                return Some(buf);
            }
            st.buf_count -= 1;
        }
        if st.buf_count < self.inner.max_bufs {
            st.buf_count += 1;
            return Some(vec![0u8; self.inner.read_size as usize]);
        }
        None
    }

    /// JS `_readSize(pos, size)` - returns (read_size, chunk_offsets?).
    fn read_size(&self, pos: u64, size: u64) -> (u64, Option<Vec<u64>>) {
        let inner = &*self.inner;
        if self.req_slice_len == 0 {
            return (inner.read_size, None);
        }
        let next_slice_pos = pos.div_ceil(self.req_slice_len) * self.req_slice_len;
        if next_slice_pos >= size {
            return (inner.read_size, None);
        }
        let mut chunks = Vec::new();
        let mut read_size = next_slice_pos - pos;
        let max_size = size - pos;
        while read_size < inner.read_size {
            if max_size.min(read_size + self.req_chunk_len) > inner.read_size {
                break;
            }
            chunks.push(read_size);
            read_size += self.req_slice_len;
            if read_size >= max_size {
                break;
            }
        }
        (inner.read_size.min(read_size), Some(chunks))
    }

    /// JS `run(readCb, finishCb)` - synchronous driver. `cb(Ok(data))`
    /// fires per chunk; `cb(Err(msg))` once on failure. Ok return = JS
    /// `finishCb` (all files done, all buffers returned).
    pub fn run(
        &mut self,
        throttle: Option<&ThrottleQueue>,
        mut cb: impl FnMut(JsResult<FileReaderData>),
    ) -> JsResult<()> {
        let no_throttle;
        let throttle = match throttle {
            Some(t) => t,
            None => {
                no_throttle = ThrottleQueue::none();
                &no_throttle
            }
        };
        loop {
            let mut st = self.inner.state.lock().unwrap();
            let buffer = match self.get_buf(&mut st) {
                Some(b) => b,
                None => {
                    st = self.inner.cv.wait(st).unwrap();
                    continue;
                }
            };

            // JS readNext file selection: open_files[0] if its hash queue
            // isn't full, else the file with the shortest queue gets moved
            // to front, else open a new file, else wait/finish
            let mut picked: Option<usize> = None;
            if !st.open_files.is_empty()
                && st.open_files[0].hash_queue
                    < self.inner.max_queue_per_file.load(Ordering::Relaxed)
            {
                picked = Some(0);
            } else {
                let mut shortest = self.inner.max_queue_per_file.load(Ordering::Relaxed);
                let mut idx = 0usize;
                for i in 1..st.open_files.len() {
                    if st.open_files[i].hash_queue < shortest {
                        shortest = st.open_files[i].hash_queue;
                        idx = i;
                    }
                }
                if idx > 0 && (shortest == 0 || st.file_queue.is_empty()) {
                    let f = st.open_files.remove(idx);
                    st.open_files.insert(0, f);
                    picked = Some(0);
                }
            }

            let file_idx = match picked {
                Some(i) => i,
                None => {
                    if let Some(next) = st.file_queue.pop_front() {
                        let name = self.files[next].name.clone();
                        let fs_path = self.files[next].fs_path().to_path_buf();
                        drop(st);
                        match File::open(&fs_path) {
                            Ok(fd) => {
                                let mut st = self.inner.state.lock().unwrap();
                                let slot = st.next_slot;
                                st.next_slot += 1;
                                st.open_files.push(OpenFile {
                                    fd,
                                    info: next,
                                    pos: 0,
                                    hash_queue: 0,
                                    slot,
                                });
                                st.buf.push(buffer);
                            }
                            Err(e) => {
                                let msg = crate::compat::fs_io_error(&e, "open", &name);
                                let mut st = self.inner.state.lock().unwrap();
                                st.buf.push(buffer);
                                drop(st);
                                cb(Err(msg.clone()));
                                return Err(msg);
                            }
                        }
                        continue;
                    } else {
                        // nothing readable - park the buffer; JS finishCb
                        // also waits for every lent buffer to come back via
                        // hashed()+release()
                        st.buf.push(buffer);
                        if st.open_files.is_empty() && st.buf.len() >= st.buf_count {
                            return Ok(());
                        }
                        st = self.inner.cv.wait(st).unwrap();
                        continue;
                    }
                }
            };

            // _doRead
            let (rs, chunks, pos, info) = {
                let ofile = &st.open_files[file_idx];
                let (rs, chunks) = self.read_size(ofile.pos, self.files[ofile.info].size);
                (rs, chunks, ofile.pos, ofile.info)
            };
            let fd = st.open_files[file_idx]
                .fd
                .try_clone()
                .map_err(|e| e.to_string())?;
            drop(st);

            let (cancelled, _done) = throttle.pass(rs as f64);
            if cancelled {
                // a cancelled throttle cancels every pass - continuing here
                // would spin forever on the same buffer
                let mut st = self.inner.state.lock().unwrap();
                st.buf.push(buffer);
                return Err("read cancelled".to_string());
            }

            let mut buffer = buffer;
            let n = match read_at(&fd, &mut buffer[..rs as usize], pos) {
                Ok(n) => n,
                Err(e) => {
                    // return the buffer to the pool like the open-error
                    // path does - keeps buf.len() consistent for the
                    // finish check even though run() returns Err
                    {
                        let mut st = self.inner.state.lock().unwrap();
                        st.buf.push(buffer);
                    }
                    let msg = crate::compat::fs_io_error(&e, "read", &self.files[info].name);
                    cb(Err(msg.clone()));
                    return Err(msg);
                }
            };

            let new_pos = pos + n as u64;
            let fsize = self.files[info].size;
            if new_pos > fsize {
                let msg = format!(
                    "Read past expected end of file - latest position ({}) exceeds size ({})",
                    new_pos, fsize
                );
                cb(Err(msg.clone()));
                return Err(msg);
            }
            let eof = new_pos == fsize;
            if !eof && n as u64 != rs {
                let msg = format!(
                    "Read failure - expected {} bytes, got {} bytes instead.",
                    rs, n
                );
                cb(Err(msg.clone()));
                return Err(msg);
            }

            let mut st = self.inner.state.lock().unwrap();
            st.open_files[file_idx].hash_queue += 1;
            st.open_files[file_idx].pos += n as u64;
            let mut ret = FileReaderData {
                file_index: info,
                buffer: Vec::new(),
                pos,
                chunks,
                reader_slot: st.open_files[file_idx].slot,
                reader: self.inner.clone(),
                refs: 2,
                hash_pending: true,
            };
            // JS hands out a view of the buffer and returns it via
            // hashed+release; deliver the buffer itself
            buffer.truncate(n);
            ret.buffer = buffer;
            drop(st);
            cb(Ok(ret));

            if eof {
                let mut st = self.inner.state.lock().unwrap();
                st.open_files.remove(file_idx);
            }
        }
    }
}

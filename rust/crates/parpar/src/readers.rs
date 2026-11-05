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
/// Reads the `chunk_offset`-aligned `chunk_size` strip of every slice in
/// file order. `concurrency` read workers (JS `chunkReadThreads`) pull
/// work items and hand pooled buffers to the callback in-order; the
/// callback owns each buffer and returns it to `pool` via the SliceBuf
/// reclaim hook - zero copy like the JS `bufPool.put` binding.
#[allow(clippy::too_many_arguments)]
pub fn file_chunk_reader(
    files: &[FileInfo],
    slice_size: u64,
    chunk_size: u64,
    chunk_offset: u64,
    pool: &BufferPool,
    concurrency: usize,
    throttle: Option<&ThrottleQueue>,
    mut cb_chunk: impl FnMut(&FileInfo, usize, Vec<u8>, usize, u64),
) -> JsResult<()> {
    let no_throttle;
    let throttle = match throttle {
        Some(t) => t,
        None => {
            no_throttle = ThrottleQueue::none();
            &no_throttle
        }
    };
    // flat (file, slice) work list in delivery order
    let mut work: Vec<(usize, u64, u64)> = Vec::new();
    let mut fds: Vec<Option<File>> = Vec::with_capacity(files.len());
    for (fidx, file) in files.iter().enumerate() {
        if file.size == 0 || file.size <= chunk_offset {
            fds.push(None);
            continue;
        }
        let fd = match File::open(file.fs_path()) {
            Ok(f) => f,
            Err(e) => return Err(crate::compat::fs_io_error(&e, "open", &file.name)),
        };
        let num_slices = (file.size - chunk_offset + slice_size - 1) / slice_size;
        for s in 0..num_slices {
            work.push((fidx, s * slice_size + chunk_offset, s));
        }
        fds.push(Some(fd));
    }
    let nwork = work.len();
    if nwork == 0 {
        return Ok(());
    }
    let workers = concurrency.clamp(1, 32).min(nwork);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel::<(
        usize,
        Result<(Vec<u8>, usize), String>,
        bool,
    )>();
    let work = &work;
    let fds = &fds;
    std::thread::scope(|sc| {
        for _ in 0..workers {
            let next = &next;
            let tx = tx.clone();
            let throttle = &*throttle;
            sc.spawn(move || loop {
                let seq = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if seq >= nwork {
                    break;
                }
                let (fidx, pos, _s) = work[seq];
                let fd = fds[fidx].as_ref().unwrap();
                // JS throttles inside the read worker before fs.read
                let (cancelled, _done) = throttle.pass(chunk_size as f64);
                if cancelled {
                    let _ = tx.send((seq, Err(String::new()), true));
                    continue;
                }
                let mut buf = pool.get();
                // the pooled buffer may exceed chunk_size; n is the real
                // byte count and travels as SliceBuf.len (buffered_process
                // clamps to it) so stale pool bytes past n stay invisible
                let r = match read_at(fd, &mut buf[..chunk_size as usize], pos) {
                    Ok(n) => Ok((buf, n)),
                    Err(e) => {
                        pool.put(buf);
                        Err(crate::compat::fs_io_error(&e, "read", &files[fidx].name))
                    }
                };
                let _ = tx.send((seq, r, false));
            });
        }
        drop(tx);
        // deliver in work order; out-of-order arrivals park in the map
        let mut pend: std::collections::BTreeMap<
            usize,
            (Result<(Vec<u8>, usize), String>, bool),
        > = std::collections::BTreeMap::new();
        let mut want = 0usize;
        let mut read_err: Option<String> = None;
        for (seq, r, cancelled) in rx {
            pend.insert(seq, (r, cancelled));
            while let Some((r, cancelled)) = pend.remove(&want) {
                match r {
                    Ok((buf, n)) => {
                        let (fidx, _pos, s) = work[want];
                        cb_chunk(&files[fidx], fidx, buf, n, s);
                    }
                    Err(e) => {
                        // cancelled slots carry an empty error and only skip
                        if !cancelled && read_err.is_none() {
                            read_err = Some(e);
                        }
                    }
                }
                want += 1;
            }
        }
        if let Some(e) = read_err {
            return Err(e);
        }
        Ok(())
    })
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
    /// this chunk ends its file - the lane-split hasher thread finalises
    /// the file's hash state here
    pub eof: bool,
    /// whole-file MD5 computed on the hasher thread (lane-split mode);
    /// set on the file's final chunk and deposited on the HasherInput
    pub file_md5: Option<[u8; 16]>,
    /// per-slice lane0 chain states emitted at the slice boundaries this
    /// chunk spans (lane-split states mode), in stream order
    pub block_states: Vec<[u32; 4]>,
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
            st.lent_bufs = st.lent_bufs.saturating_sub(1);
            let buf = std::mem::take(&mut self.buffer);
            if buf.len() as u64 >= self.reader.read_size {
                st.buf.push(buf);
            }
            self.reader.cv.notify_one();
        }
    }
    /// JS `release()` - buffer recycled once hashing also done.
    pub fn release(&mut self) {
        self.refs -= 1;
        if self.refs == 0 {
            let mut st = self.reader.state.lock().unwrap();
            st.lent_bufs = st.lent_bufs.saturating_sub(1);
            let buf = std::mem::take(&mut self.buffer);
            // undersized/empty buffers never re-enter the pool - the
            // zero-copy path empties `buffer` when it moves into a
            // SliceBuf; that storage returns via its reclaim callback
            if buf.len() as u64 >= self.reader.read_size {
                st.buf.push(buf);
            }
            self.reader.cv.notify_one();
        }
    }

    /// Reclaim closure for the zero-copy slice handoff: the pooled buffer
    /// travels inside a SliceBuf and is pushed back here when the backend
    /// drops it (the C++ backend returns read buffers to the pool in the
    /// same way).
    pub fn buf_reclaimer(&self) -> impl FnOnce(Vec<u8>) + Send {
        let inner = self.reader.clone();
        move |v: Vec<u8>| {
            if v.len() as u64 >= inner.read_size {
                let mut st = inner.state.lock().unwrap();
                st.buf.push(v);
                drop(st);
                inner.cv.notify_one();
            }
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
    /// read slots currently lent out (incremented at get_buf, decremented
    /// when the slot is released via hashed()+release() refs hitting 0 or
    /// when the buffer is parked back) - the run-finish check needs this:
    /// real storage may still sit inside a SliceBuf in the proc while its
    /// slot is already released
    lent_bufs: usize,
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

/// Lane-split producer mode.
#[derive(Clone, Copy)]
pub enum LaneMode {
    /// no producer-side hashing
    Off,
    /// producer computes only the whole-file MD5
    FileOnly,
    /// producer runs both MD5 lanes (x2) and ships per-slice lane0
    /// chain states + the file digest; needs slice_size % 64 == 0
    States(u64),
}

/// Per-open-file producer hashing state (keyed by reader_slot).
enum LaneProd {
    File(parpar_hasher::FileMd5Ctx),
    States(parpar_hasher::LaneProducer),
}

/// `lib/fileseqreader.js` - sequential reader preferring in-order file
/// progress with a bounded buffer pool and per-file hash queues.
pub struct FileSeqReader {
    files: Vec<FileInfo>,
    inner: Arc<SeqInner>,
    req_slice_len: u64,
    req_chunk_len: u64,
    /// producer thread also streams each file's bytes through the lane
    /// hasher(s); results ride out on the file's chunks
    lane_mode: LaneMode,
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
                    lent_bufs: 0,
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
            lane_mode: LaneMode::Off,
        }
    }

    /// Enable lane-split hashing: the producer thread computes each file's
    /// external hash lane(s) and attaches them to the file's chunks.
    pub fn set_lane_mode(&mut self, mode: LaneMode) {
        self.lane_mode = mode;
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
                st.lent_bufs += 1;
                return Some(buf);
            }
            st.buf_count -= 1;
        }
        if st.buf_count < self.inner.max_bufs {
            st.buf_count += 1;
            st.lent_bufs += 1;
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
        // JS performs the reads on background workers while the main loop
        // hashes; replicate that overlap with a scoped producer thread so
        // the pread of chunk N+1 hides under the hashing of chunk N. In
        // lane-split mode a dedicated hasher thread sits between reader
        // and consumer: reads stay off the MD5 dependency chain and the
        // consumer keeps only CRC + finals + packing
        let chan_bound = self.inner.state.lock().unwrap().buf_count.max(1);
        let (tx, rx) = std::sync::mpsc::sync_channel::<JsResult<FileReaderData>>(chan_bound);
        let (tx2, rx2) = std::sync::mpsc::sync_channel::<JsResult<FileReaderData>>(chan_bound);
        let mut first_err: Option<String> = None;
        let mode = self.lane_mode;
        std::thread::scope(|s| {
            let me: &Self = self;
            let producer = s.spawn(move || me.produce(throttle, tx));
            let hasher = s.spawn(move || me.hash_lane(mode, rx, tx2));
            for item in rx2 {
                let is_err = item.is_err();
                if let Err(e) = &item {
                    if first_err.is_none() {
                        first_err = Some(e.clone());
                    }
                }
                cb(item);
                if is_err {
                    break;
                }
            }
            let _ = producer.join();
            let _ = hasher.join();
        });
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn produce(
        &self,
        throttle: Option<&ThrottleQueue>,
        tx: std::sync::mpsc::SyncSender<JsResult<FileReaderData>>,
    ) {
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
                                st.lent_bufs = st.lent_bufs.saturating_sub(1);
                                st.buf.push(buffer);
                            }
                            Err(e) => {
                                let msg = crate::compat::fs_io_error(&e, "open", &name);
                                let mut st = self.inner.state.lock().unwrap();
                                st.lent_bufs = st.lent_bufs.saturating_sub(1);
                                st.buf.push(buffer);
                                drop(st);
                                let _ = tx.send(Err(msg));
                                return;
                            }
                        }
                        continue;
                    } else {
                        // nothing readable - park the buffer; JS finishCb
                        // also waits for every lent buffer to come back via
                        // hashed()+release()
                        st.lent_bufs = st.lent_bufs.saturating_sub(1);
                        st.buf.push(buffer);
                        if st.open_files.is_empty() && st.lent_bufs == 0 {
                            return;
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
            let fd = match st.open_files[file_idx].fd.try_clone() {
                Ok(fd) => fd,
                Err(e) => {
                    let msg = e.to_string();
                    let mut st = self.inner.state.lock().unwrap();
                    st.lent_bufs = st.lent_bufs.saturating_sub(1);
                    st.buf.push(buffer);
                    drop(st);
                    let _ = tx.send(Err(msg));
                    return;
                }
            };
            drop(st);

            let (cancelled, _done) = throttle.pass(rs as f64);
            if cancelled {
                // a cancelled throttle cancels every pass - continuing here
                // would spin forever on the same buffer
                let mut st = self.inner.state.lock().unwrap();
                st.lent_bufs = st.lent_bufs.saturating_sub(1);
                st.buf.push(buffer);
                let _ = tx.send(Err("read cancelled".to_string()));
                return;
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
                        st.lent_bufs = st.lent_bufs.saturating_sub(1);
                        st.buf.push(buffer);
                    }
                    let msg = crate::compat::fs_io_error(&e, "read", &self.files[info].name);
                    let _ = tx.send(Err(msg));
                    return;
                }
            };

            let new_pos = pos + n as u64;
            let fsize = self.files[info].size;
            if new_pos > fsize {
                let msg = format!(
                    "Read past expected end of file - latest position ({}) exceeds size ({})",
                    new_pos, fsize
                );
                let _ = tx.send(Err(msg));
                return;
            }
            let eof = new_pos == fsize;
            if !eof && n as u64 != rs {
                let msg = format!(
                    "Read failure - expected {} bytes, got {} bytes instead.",
                    rs, n
                );
                let _ = tx.send(Err(msg));
                return;
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
                eof,
                file_md5: None,
                block_states: Vec::new(),
            };
            // JS hands out a view of the buffer and returns it via
            // hashed+release; deliver the buffer itself
            buffer.truncate(n);
            ret.buffer = buffer;
            drop(st);
            if tx.send(Ok(ret)).is_err() {
                return;
            }

            if eof {
                let mut st = self.inner.state.lock().unwrap();
                st.open_files.remove(file_idx);
            }
        }
    }

    /// Lane-split hasher thread: sits between the reader and the
    /// consumer so the serial MD5 work runs off both of them. Keeps one
    /// external-hash context per open file keyed by its stable slot id;
    /// attaches per-slice lane0 chain states to every chunk and the
    /// whole-file digest to the file's last chunk. `Off` just forwards.
    fn hash_lane(
        &self,
        mode: LaneMode,
        rx: std::sync::mpsc::Receiver<JsResult<FileReaderData>>,
        tx: std::sync::mpsc::SyncSender<JsResult<FileReaderData>>,
    ) {
        use std::collections::HashMap;
        let mut lanes: HashMap<u64, LaneProd> = HashMap::new();
        let mut _upd = std::time::Duration::ZERO;
        let mut _wait = std::time::Duration::ZERO;
        for item in rx {
            let item = match item {
                Ok(mut data) => {
                    if !matches!(mode, LaneMode::Off) {
                        let lp = lanes.entry(data.reader_slot).or_insert_with(|| match mode {
                            LaneMode::FileOnly => LaneProd::File(parpar_hasher::FileMd5Ctx::new()),
                            LaneMode::States(ss) => {
                                LaneProd::States(parpar_hasher::LaneProducer::new(ss))
                            }
                            LaneMode::Off => unreachable!(),
                        });
                        let _t = std::time::Instant::now();
                        match lp {
                            LaneProd::File(c) => c.update(&data.buffer),
                            LaneProd::States(p) => p.update(&data.buffer, &mut data.block_states),
                        }
                        _upd += _t.elapsed();
                        if data.eof {
                            if let Some(lp) = lanes.remove(&data.reader_slot) {
                                data.file_md5 = Some(match lp {
                                    LaneProd::File(mut c) => c.finish(),
                                    LaneProd::States(mut p) => p.finish(&mut data.block_states),
                                });
                            }
                        }
                    }
                    Ok(data)
                }
                Err(e) => Err(e),
            };
            let _t = std::time::Instant::now();
            if tx.send(item).is_err() {
                return;
            }
            _wait += _t.elapsed();
        }
        if std::env::var_os("PARPAR_PERF").is_some() {
            eprintln!(
                "[perf] lane: upd={:.2}s send-wait={:.2}s",
                _upd.as_secs_f64(),
                _wait.as_secs_f64()
            );
        }
    }
}

//! Port of `lib/par2outfile.js` - PAR2OutFile: open/prealloc/writePackets/
//! close for a recovery volume. Write-combining uses `pwritev` when
//! available, with the JS MAX_WRITE_SIZE (0x7ffff000) cap and
//! 'wx'/'w' create modes; prealloc is ftruncate + a junk byte at the end.

use std::fs::{File, OpenOptions};

use crate::generate::GenPacket;

#[cfg(unix)]
fn write_at(f: &File, buf: &[u8], pos: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.write_at(buf, pos)
}

#[cfg(windows)]
fn write_at(f: &File, buf: &[u8], pos: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_write(buf, pos)
}

/// write may legally return short (EINTR mid-write, RLIMIT_FSIZE, quota) -
/// loop until the buffer is out the door or a real error surfaces
fn write_all_at(f: &File, mut buf: &[u8], mut pos: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        match write_at(f, buf, pos) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ))
            }
            Ok(n) => {
                pos += n as u64;
                buf = &buf[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

pub const MAX_WRITE_SIZE: u64 = 0x7ffff000; // writev 2GB-4KB page limit

pub type JsResult<T> = Result<T, String>;

/// `lib/par2outfile.js` PAR2OutFile.
pub struct Par2OutFile {
    pub name: String,
    pub recovery_slices: i64,
    /// relative index used for processing
    pub recovery_index: i64,
    pub packets: Vec<GenPacket>,
    pub total_size: u64,
    fd: Option<File>,
}

impl Par2OutFile {
    pub fn new(
        name: String,
        recovery_slices: i64,
        recovery_index: i64,
        packets: Vec<GenPacket>,
        total_size: u64,
    ) -> Self {
        Par2OutFile {
            name,
            recovery_slices,
            recovery_index,
            packets,
            total_size,
            fd: None,
        }
    }

    /// JS `open(overwrite, cb)` - 'wx' exclusive or 'w' truncate.
    pub fn open(&mut self, overwrite: bool) -> JsResult<()> {
        if self.fd.is_some() {
            return Ok(());
        }
        let mut o = OpenOptions::new();
        o.write(true);
        if overwrite {
            o.create(true).truncate(true);
        } else {
            o.create_new(true);
        }
        match o.open(&self.name) {
            Ok(f) => {
                self.fd = Some(f);
                Ok(())
            }
            Err(e) => Err(crate::compat::fs_io_error(&e, "open", &self.name)),
        }
    }

    /// JS `prealloc(cb)` - ftruncate(totalSize) then a junk byte at
    /// totalSize-1 (emulates fallocate).
    pub fn prealloc(&mut self) -> JsResult<()> {
        if self.total_size == 0 {
            return Ok(());
        }
        let fd = match &self.fd {
            Some(f) => f,
            None => return Ok(()),
        };
        if let Err(e) = fd.set_len(self.total_size) {
            return Err(crate::compat::fs_io_error(&e, "ftruncate", &self.name));
        }
        write_all_at(&fd, &[255u8], self.total_size - 1)
            .map_err(|e| crate::compat::fs_io_error(&e, "write", &self.name))?;
        Ok(())
    }

    /// JS `writePackets(pktI, curPos, cb)` packet-selection half: compute
    /// the write-combined job starting at `pkt_i` and take the data out of
    /// the packets, without performing any I/O. Returns the job and the
    /// number of packets covered. Split from `write_packets` so the job
    /// can be handed to a WriteQueue worker (JS dispatches writes through
    /// libuv's thread pool the same way).
    pub fn take_write_job(&mut self, pkt_i: usize, cur_pos: u64) -> JsResult<(WriteJob, usize)> {
        let fd = match &self.fd {
            Some(f) => f.try_clone().map_err(|e| e.to_string())?,
            None => return Err("file not open".into()),
        };
        let name = self.name.clone();
        let pkt = &self.packets[pkt_i];
        let mut write_to = pkt_i + 1;
        let mut write_len = pkt.data_len() as u64;
        if pkt.data_chunk_offset as u64 + write_len == pkt.size {
            while write_to < self.packets.len() {
                let np = &self.packets[write_to];
                if np.data.is_empty() || np.data_chunk_offset != 0 {
                    break;
                }
                let nlen = np.data_len() as u64;
                if write_len + nlen > MAX_WRITE_SIZE {
                    break;
                }
                write_to += 1;
                write_len += nlen;
                if nlen != np.size {
                    break; // partial write needs a seek
                }
            }
        }

        let pos = cur_pos + pkt.data_chunk_offset as u64;
        if write_len <= MAX_WRITE_SIZE {
            let mut wbufs: Vec<Vec<u8>> = Vec::new();
            for p in &mut self.packets[pkt_i..write_to] {
                wbufs.append(&mut p.take_data());
            }
            Ok((WriteJob::Writev { fd, pos, bufs: wbufs, name }, write_to - pkt_i))
        } else {
            let bufs = self.packets[pkt_i].take_data();
            Ok((WriteJob::Writes { fd, pos, bufs, name }, 1))
        }
    }

    /// JS `writePackets(pktI, curPos, cb)` - write as many packets as
    /// possible starting at `pkt_i`; returns the number of packets written.
    /// Write-combines consecutive packets (writev) when the first packet's
    /// data ends flush with its size; else sequential pwrite per data chunk
    /// split at MAX_WRITE_SIZE.
    pub fn write_packets(&mut self, pkt_i: usize, cur_pos: u64) -> JsResult<usize> {
        let (job, n) = self.take_write_job(pkt_i, cur_pos)?;
        job.run()?;
        Ok(n)
    }

    /// JS `close(sync, cb)` - optional fsync then close.
    pub fn close(&mut self, sync: bool) -> JsResult<()> {
        if let Some(fd) = self.fd.take() {
            if sync {
                if let Err(e) = fd.sync_data() {
                    let _ = fd; // still close like JS
                    return Err(crate::compat::fs_io_error(&e, "fsync", &self.name));
                }
            }
        }
        Ok(())
    }

    /// JS `discardData()`.
    pub fn discard_data(&mut self) {
        for p in &mut self.packets {
            p.data.clear();
        }
    }
}

/// An owned write operation: the fd clone, position and data all move to
/// whichever thread performs the syscall - matching how JS hands writes
/// to libuv's pool while the main thread keeps pulling recovery data.
pub enum WriteJob {
    /// write-combined packets via pwritev
    Writev {
        fd: File,
        pos: u64,
        bufs: Vec<Vec<u8>>,
        name: String,
    },
    /// sequential pwrite per chunk split at MAX_WRITE_SIZE
    Writes {
        fd: File,
        pos: u64,
        bufs: Vec<Vec<u8>>,
        name: String,
    },
}

impl WriteJob {
    pub fn run(self) -> JsResult<()> {
        match self {
            WriteJob::Writev {
                fd,
                pos,
                bufs,
                name,
            } => {
                let ios: Vec<std::io::IoSlice<'_>> =
                    bufs.iter().map(|d| std::io::IoSlice::new(d)).collect();
                writev_at(&fd, &ios, pos)
                    .map_err(|e| crate::compat::fs_io_error(&e, "write", &name))
            }
            WriteJob::Writes {
                fd,
                pos,
                bufs,
                name,
            } => {
                let mut pos = pos;
                for data in &bufs {
                    let mut off = 0usize;
                    while off < data.len() {
                        let wlen = (MAX_WRITE_SIZE as usize).min(data.len() - off);
                        write_all_at(&fd, &data[off..off + wlen], pos)
                            .map_err(|e| crate::compat::fs_io_error(&e, "write", &name))?;
                        pos += wlen as u64;
                        off += wlen;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Background writer: a single worker drains write jobs off a bounded
/// channel so `write_file`'s pull/assemble loop overlaps the syscalls,
/// the same overlap JS gets from libuv. Writes carry explicit offsets so
/// the single worker's ordering is sufficient.
pub struct WriteQueue {
    tx: Option<std::sync::mpsc::SyncSender<WriteJob>>,
    worker: Option<std::thread::JoinHandle<()>>,
    shared: std::sync::Arc<WriteShared>,
}

struct WriteShared {
    pending: std::sync::Mutex<usize>,
    empty: std::sync::Condvar,
    err: std::sync::Mutex<Option<String>>,
}

impl Default for WriteQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteQueue {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<WriteJob>(4);
        let shared = std::sync::Arc::new(WriteShared {
            pending: std::sync::Mutex::new(0),
            empty: std::sync::Condvar::new(),
            err: std::sync::Mutex::new(None),
        });
        let wshared = shared.clone();
        let worker = std::thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                if let Err(e) = job.run() {
                    let mut err = wshared.err.lock().unwrap();
                    if err.is_none() {
                        *err = Some(e);
                    }
                }
                let mut p = wshared.pending.lock().unwrap();
                *p -= 1;
                if *p == 0 {
                    wshared.empty.notify_all();
                }
            }
        });
        WriteQueue {
            tx: Some(tx),
            worker: Some(worker),
            shared,
        }
    }

    /// Queue a job; blocks when the bound is full so memory stays bounded
    /// (jobs carry the packet buffers).
    pub fn submit(&self, job: WriteJob) -> JsResult<()> {
        {
            let mut p = self.shared.pending.lock().unwrap();
            *p += 1;
        }
        match self.tx.as_ref() {
            Some(tx) if tx.send(job).is_ok() => Ok(()),
            _ => {
                let mut p = self.shared.pending.lock().unwrap();
                *p -= 1;
                Err("write worker gone".into())
            }
        }
    }

    /// Surface a worker error early (mirrors JS's deferred write-callback
    /// errors) without waiting for a full drain.
    pub fn check(&self) -> JsResult<()> {
        match &*self.shared.err.lock().unwrap() {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Wait until every queued write finished, then return the first
    /// worker error if any. Must be called before fsync/close so the
    /// sync covers everything enqueued.
    pub fn drain(&self) -> JsResult<()> {
        let mut p = self.shared.pending.lock().unwrap();
        while *p > 0 {
            p = self.shared.empty.wait(p).unwrap();
        }
        drop(p);
        match self.shared.err.lock().unwrap().take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl Drop for WriteQueue {
    fn drop(&mut self) {
        let _ = self.drain();
        self.tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// `pwritev` wrapper; loops on short writes so the caller can assume the
/// whole iov set landed or an error was raised.
#[cfg(unix)]
fn writev_at(f: &File, bufs: &[std::io::IoSlice<'_>], mut pos: u64) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let mut i = 0usize;
    let mut off = 0usize; // bytes already written within bufs[i]
    while i < bufs.len() {
        // advance past fully-written buffers
        while i < bufs.len() && off >= bufs[i].len() {
            off -= bufs[i].len();
            i += 1;
        }
        if i >= bufs.len() {
            break;
        }
        // pwritev fails EINVAL past IOV_MAX (1024); Node/libuv chunks
        // internally - cap each syscall the same way
        const IOV_MAX: usize = 1024;
        let iovecs: Vec<libc::iovec> = bufs[i..]
            .iter()
            .take(IOV_MAX)
            .enumerate()
            .map(|(k, b)| {
                let skip = if k == 0 { off } else { 0 };
                libc::iovec {
                    iov_base: unsafe { b.as_ptr().add(skip) } as *mut libc::c_void,
                    iov_len: b.len() - skip,
                }
            })
            .collect();
        let n = unsafe {
            libc::pwritev(
                f.as_raw_fd(),
                iovecs.as_ptr(),
                iovecs.len() as libc::c_int,
                pos as libc::off_t,
            )
        };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "failed to write whole buffer",
            ));
        }
        pos += n as u64;
        off += n as usize;
    }
    Ok(())
}

/// Windows has no positional writev; write each buffer sequentially at the
/// advancing offset (same observable result, one syscall per buffer).
#[cfg(windows)]
fn writev_at(f: &File, bufs: &[std::io::IoSlice<'_>], mut pos: u64) -> std::io::Result<()> {
    for b in bufs {
        write_all_at(f, &b[..], pos)?;
        pos += b.len() as u64;
    }
    Ok(())
}

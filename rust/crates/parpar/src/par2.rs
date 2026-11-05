//! Port of `lib/par2.js` - the PAR2 packet model (main/creator/comment/
//! filedesc/unifilen/ifsc/recovery packets, file IDs, setID) plus the
//! GFWrapper recovery-generation lifecycle, adapted to a synchronous pull
//! model over the backend event queue.
//!
//! Byte-parity notes carried over from JS:
//! - file sort uses `BufferCompareReverse` - reversed byte order
//! - `strToAscii` uses the configured `asciiCharset` (default utf-8)
//! - `comment.length`/`displayName.length` are UTF-16 code-unit counts
//! - unicode packet selection: AUTO->BOTH iff any char >= 0x80
//! - the UniFileN/CommUni link field is the ascii packet's MD5 (16B)

use md5::{Digest, Md5};
use parpar_backend::*;
use std::collections::VecDeque;

/// Result for JS-thrown (string) errors, vs parpar_backend::Result.
pub type JsResult<T> = std::result::Result<T, String>;

pub const RECOVERY_HEADER_SIZE: usize = 68;
const MAGIC: &[u8; 8] = b"PAR2\0PKT";

/// `CHAR_CONST` in par2.js.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharOpt {
    Both,
    Ascii,
    Unicode,
}

fn md5_of(parts: &[&[u8]]) -> [u8; 16] {
    let mut h = Md5::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

fn put_u64le(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// `/[\u0080-\uffff]/` on the JS string - any non-ASCII character.
fn has_unicode(s: &str) -> bool {
    s.chars().any(|c| c as u32 >= 0x80)
}

/// JS `str.length` is a UTF-16 code-unit count.
fn utf16_units(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `Buffer.from(str, asciiCharset)` for the encodings parpar accepts.
/// `Buffer.isEncoding` admits more labels but only these are meaningful
/// for a charset option (hex/base64 encodings are decode-direction only
/// in JS Buffer.from(string) semantics).
pub fn str_to_ascii(s: &str, charset: &str) -> Vec<u8> {
    let enc = charset.to_lowercase();
    match enc.as_str() {
        "utf-8" | "utf8" => s.as_bytes().to_vec(),
        "utf16le" | "utf-16le" | "ucs2" | "ucs-2" => {
            let mut out = Vec::with_capacity(s.len() * 2);
            for u in s.encode_utf16() {
                out.extend_from_slice(&u.to_le_bytes());
            }
            out
        }
        // Node encodes latin1/ascii per UTF-16 code unit - non-BMP chars
        // are surrogate pairs producing two bytes, not one
        // Node treats 'ascii' identically to 'latin1' when *encoding*
        // (the 7-bit mask only applies on decode) - mask to 0xff
        "latin1" | "binary" | "ascii" => s.encode_utf16().map(|u| (u & 0xff) as u8).collect(),
        // Buffer.from(str, 'hex'/'base64') decodes rather than passing
        // bytes through
        "hex" => str_from_hex(s),
        "base64" | "base64url" => str_from_base64(s),
        // unreachable through validated charsets; treat unknown as utf-8
        _ => s.as_bytes().to_vec(),
    }
}

/// `Buffer.from(str, 'hex')` - decodes hex pairs, stopping at the first
/// invalid or incomplete pair (Node's lenient semantics).
fn str_from_hex(s: &str) -> Vec<u8> {
    let nib = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len() / 2);
    for pair in b.chunks_exact(2) {
        match (nib(pair[0]), nib(pair[1])) {
            (Some(hi), Some(lo)) => out.push(hi << 4 | lo),
            _ => break,
        }
    }
    out
}

/// `Buffer.from(str, 'base64'|'base64url')` - lenient decode: characters
/// outside the alphabet are skipped, '=' terminates, truncated tails
/// drop, and both +/ and -_ pair spellings are accepted interchangeably
/// (Node accepts either alphabet for both labels).
fn str_from_base64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' | b'-' => Some(62),
            b'/' | b'_' => Some(63),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        match val(c) {
            Some(v) => {
                acc = (acc << 6) | v as u32;
                nbits += 6;
                if nbits >= 8 {
                    nbits -= 8;
                    out.push((acc >> nbits) as u8);
                }
            }
            None => continue, // skip non-alphabet chars (whitespace etc.)
        }
    }
    out
}

/// `Buffer.compare(a.reverse(), b.reverse())` - compares last byte first.
/// (JS sorts PAR2File ids this way.)
fn cmp_reversed(a: &[u8; 16], b: &[u8; 16]) -> std::cmp::Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// One input file's PAR2 model (`PAR2File` in par2.js).
pub struct Par2File {
    /// File ID: md5(md5_16k || size u64le || displayName in ascii charset).
    pub id: [u8; 16],
    /// `file.name` - the filesystem path (copied onto PAR2File in JS).
    /// May be lossy for non-UTF-8 names; fs_path carries the raw path.
    pub name: String,
    /// Raw filesystem path when `name` was lossy-derived from a non-UTF-8
    /// readdir entry - readers open this, falling back to `name`.
    pub fs_path: Option<std::path::PathBuf>,
    pub display_name: String,
    pub size: u64,
    /// PAR2 slice size this file was chunked against (`par2.sliceSize`).
    pub slice_size: u64,
    pub num_slices: u64,
    pub slice_offset: u64,
    /// Whole-file MD5 (None until hashing completes; `d41d8c...` constant
    /// for empty files).
    pub md5: Option<[u8; 16]>,
    pub md5_16k: [u8; 16],
    /// IFSC packet being built: 64-byte header area + file.id + 20B/slice
    /// records. The hasher's block_out fills records at offset 80+.
    pkt_check: Option<Vec<u8>>,
    /// `binding.HasherInput` equivalent (None for empty files).
    hasher: Option<Box<dyn HasherInput>>,
    /// position counters (JS hashPos / sliceDataPos)
    pub hash_pos: u64,
    pub slice_data_pos: u64,
}

const EMPTY_MD5: [u8; 16] = [
    0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8, 0x42, 0x7e,
];

impl Par2File {
    /// `file` = {md5_16k, size, displayName} - displayName must already be
    /// resolved (JS defaults basename(name) in the constructor).
    pub fn new(par2: &Par2Info, md5_16k: [u8; 16], size: u64, display_name: String) -> Par2File {
        let mut id_src = Vec::with_capacity(16 + 8 + display_name.len());
        id_src.extend_from_slice(&md5_16k);
        id_src.extend_from_slice(&size.to_le_bytes());
        id_src.extend_from_slice(&str_to_ascii(&display_name, &par2.ascii_charset));
        let id = md5_of(&[&id_src]);

        let num_slices = if size == 0 {
            0
        } else {
            size.div_ceil(par2.slice_size)
        };
        let pkt_check = if size > 0 {
            Some(vec![0u8; 64 + 16 + 20 * num_slices as usize])
        } else {
            None
        };
        Par2File {
            id,
            name: String::new(),
            fs_path: None,
            display_name,
            size,
            slice_size: par2.slice_size,
            num_slices,
            slice_offset: 0,
            md5: if size == 0 { Some(EMPTY_MD5) } else { None },
            md5_16k,
            pkt_check,
            hasher: None,
            hash_pos: 0,
            slice_data_pos: 0,
        }
    }

    /// The 20B/slice record region the input hasher writes into
    /// (`pktCheck` tail at offset 80).
    pub fn take_checksum_records(&mut self) -> Option<Box<[u8]>> {
        let pkt = self.pkt_check.as_mut()?;
        let tail: Vec<u8> = pkt.drain(80..).collect();
        Some(tail.into_boxed_slice())
    }
    /// `HasherInput::end` hands block_out back; splice it at offset 80.
    pub fn put_checksum_records(&mut self, records: Box<[u8]>) {
        if let Some(pkt) = self.pkt_check.as_mut() {
            debug_assert_eq!(pkt.len(), 80);
            pkt.extend_from_slice(&records);
        }
    }

    pub fn packet_checksums_size(&self) -> usize {
        if self.num_slices == 0 {
            0
        } else {
            64 + 16 + 20 * self.num_slices as usize
        }
    }

    /// IFSC packet (empty Vec for zero-slice files). `keep` mirrors JS:
    /// false consumes the buffer so a second call returns an empty packet.
    pub fn get_packet_checksums(&mut self, par2: &Par2Info, keep: bool) -> Vec<u8> {
        if self.num_slices == 0 {
            return Vec::new();
        }
        // a second call after !keep (or before any hashing) yields an
        // empty packet, matching JS's consumed/never-built buffer
        let Some(pkt) = self.pkt_check.take() else {
            return Vec::new();
        };
        let mut pkt = pkt;
        // records were drained into the hasher's buffer; if hashing was
        // abandoned mid-file the tail is missing - JS always holds the
        // full buffer (hasher writes via a view), so restore the packet's
        // declared length with zeroed records
        let expected = 64 + 16 + 20 * self.num_slices as usize;
        if pkt.len() < expected {
            pkt.resize(expected, 0);
        }
        pkt[64..80].copy_from_slice(&self.id);
        let plen = pkt.len() - 64;
        par2.write_pkt_header(&mut pkt, 0, plen, b"PAR 2.0\0IFSC\0\0\0\0", false);
        if keep {
            self.pkt_check = Some(pkt.clone());
        }
        pkt
    }

    pub fn packet_description_size(&self, unicode: Option<CharOpt>, charset: &str) -> usize {
        let unicode = match unicode {
            None => {
                if has_unicode(&self.display_name) {
                    CharOpt::Both
                } else {
                    CharOpt::Ascii
                }
            }
            Some(u) => {
                debug_assert!(u != CharOpt::Unicode, "Unknown unicode option");
                u
            }
        };
        let len = ceil4(str_to_ascii(&self.display_name, charset).len());
        let len2 = ceil4(utf16_units(&self.display_name) * 2);
        let mut n = 64 + 56 + len;
        if unicode == CharOpt::Both {
            n += 64 + 16 + len2;
        }
        n
    }

    /// FileDesc packet (+ appended UniFileN when unicode == Both).
    /// Requires `md5` set (JS throws when data hasn't been processed).
    pub fn make_packet_description(
        &mut self,
        par2: &Par2Info,
        unicode: Option<CharOpt>,
    ) -> JsResult<Vec<u8>> {
        let unicode = match unicode {
            None => {
                if has_unicode(&self.display_name) {
                    CharOpt::Both
                } else {
                    CharOpt::Ascii
                }
            }
            Some(u) => {
                if u == CharOpt::Unicode {
                    return Err("Unknown unicode option".into());
                }
                u
            }
        };
        let charset = par2.ascii_charset.clone();
        let file_name = str_to_ascii(&self.display_name, &charset);
        let len = ceil4(file_name.len());
        let len2 = ceil4(utf16_units(&self.display_name) * 2);
        let pkt1_len = 64 + 56 + len;
        let total = pkt1_len
            + if unicode == CharOpt::Both {
                64 + 16 + len2
            } else {
                0
            };
        let mut pkt = vec![0u8; total];
        pkt[64..80].copy_from_slice(&self.id);
        let Some(md5) = self.md5 else {
            return Err(
                "MD5 of file not available. Ensure that all data has been read and processed."
                    .into(),
            );
        };
        pkt[80..96].copy_from_slice(&md5);
        pkt[96..112].copy_from_slice(&self.md5_16k);
        put_u64le(&mut pkt, 112, self.size);
        pkt[120..120 + file_name.len()].copy_from_slice(&file_name);
        par2.write_pkt_header(&mut pkt, 0, 56 + len, b"PAR 2.0\0FileDesc", false);
        if unicode == CharOpt::Both {
            self.write_packet_uni_name(par2, &mut pkt, len2, pkt1_len);
        }
        Ok(pkt)
    }

    /// UniFileN packet body = file.id || utf16le(name); last 2 bytes zeroed.
    fn write_packet_uni_name(&self, par2: &Par2Info, pkt: &mut [u8], len: usize, offset: usize) {
        pkt[offset + 64..offset + 80].copy_from_slice(&self.id);
        let end = offset + 64 + 16 + len;
        // JS zeroes the tail padding *before* writing the text
        pkt[end - 2] = 0;
        pkt[end - 1] = 0;
        let mut pos = offset + 80;
        for u in self.display_name.encode_utf16() {
            pkt[pos..pos + 2].copy_from_slice(&u.to_le_bytes());
            pos += 2;
        }
        par2.write_pkt_header(
            &mut pkt[..end],
            offset,
            end - offset - 64,
            b"PAR 2.0\0UniFileN",
            false,
        );
    }

    pub fn packet_uni_name_size(&self) -> usize {
        64 + 16 + ceil4(utf16_units(&self.display_name) * 2)
    }

    pub fn make_packet_uni_name(&self, par2: &Par2Info) -> Vec<u8> {
        let len = self.packet_uni_name_size();
        let mut pkt = vec![0u8; len];
        self.write_packet_uni_name(par2, &mut pkt, len - 64 - 16, 0);
        pkt
    }
}

fn ceil4(n: usize) -> usize {
    (n + 3) & !3
}

/// Shared read-only PAR2 context (what `this.par2` means to PAR2File):
/// slice size, setID, charset - separated so files can outlive mutation.
#[derive(Debug, Clone)]
pub struct Par2Info {
    pub slice_size: u64,
    pub set_id: [u8; 16],
    pub ascii_charset: String,
}

impl Par2Info {
    /// `_writePktHeader(buf, name, offset, len, skipMD5)`.
    /// `name` must be exactly 16 bytes.
    pub fn write_pkt_header(
        &self,
        buf: &mut [u8],
        offset: usize,
        len: usize,
        name: &[u8; 16],
        skip_md5: bool,
    ) {
        assert_eq!(len % 4, 0, "Packet length must be a multiple of 4");
        let pkt_len = 64 + len;
        buf[offset..offset + 8].copy_from_slice(MAGIC);
        put_u64le(buf, offset + 8, pkt_len as u64);
        buf[offset + 32..offset + 48].copy_from_slice(&self.set_id);
        buf[offset + 48..offset + 64].copy_from_slice(name);
        if !skip_md5 {
            let digest = md5_of(&[&buf[offset + 32..offset + pkt_len]]);
            buf[offset + 16..offset + 32].copy_from_slice(&digest);
        }
    }

    /// 68-byte recovery packet header (`getRecoveryHeader`).
    pub fn recovery_header(&self, exponent: u16, md5: &[u8; 16], slice_size: u64) -> [u8; 68] {
        let mut pkt = [0u8; 68];
        pkt[..8].copy_from_slice(MAGIC);
        put_u64le(&mut pkt, 8, slice_size + 68);
        pkt[16..32].copy_from_slice(md5);
        pkt[32..48].copy_from_slice(&self.set_id);
        pkt[48..64].copy_from_slice(b"PAR 2.0\0RecvSlic");
        pkt[64..66].copy_from_slice(&exponent.to_le_bytes());
        pkt
    }
}

/// `PAR2` in par2.js - owns the file list, main packet and setID, plus the
/// mixed-in GFWrapper (here: `gen`).
pub struct Par2<C: GfCtx> {
    pub info: Par2Info,
    pub files: Vec<Par2File>,
    pub total_slices: u64,
    pkt_main: Vec<u8>,
    /// The GFWrapper mixin (recoverySlices, gf, recData machinery).
    pub gen: RecoveryGen<C>,
}

impl<C: GfCtx> Par2<C> {
    /// files: `(md5_16k, size, display_name)` per input file.
    /// `slice_size` must be a multiple of 4 (JS throws otherwise).
    /// `gf_opts`/`proc_opts` are the resolved `_gfOpts`.
    pub fn new(
        files: Vec<(
            /* md5_16k */ [u8; 16],
            /* size */ u64,
            /* fs name */ String,
            /* display name */ String,
            /* raw fs path (when name is lossy) */ Option<std::path::PathBuf>,
        )>,
        slice_size: u64,
        ascii_charset: String,
        ctx: C,
        gf_opts: GfOpts,
        proc_opts: ProcOpts,
    ) -> JsResult<Par2<C>> {
        // JS lets sliceSize==0 through the %4 check but then computes
        // numSlices = Infinity and dies allocating - reject it up front
        if slice_size == 0 {
            return Err("Invalid slice size".into());
        }
        if slice_size % 4 != 0 {
            return Err("Slice size must be a multiple of 4".into());
        }
        let mut info = Par2Info {
            slice_size,
            set_id: [0u8; 16],
            ascii_charset,
        };
        let mut files: Vec<Par2File> = files
            .into_iter()
            .map(|(md5_16k, size, name, display, fs_path)| {
                let mut f = Par2File::new(&info, md5_16k, size, display);
                f.name = name;
                f.fs_path = fs_path;
                f
            })
            .collect();
        files.sort_by(|a, b| cmp_reversed(&a.id, &b.id));
        let mut slice_offset = 0u64;
        for f in files.iter_mut() {
            f.slice_offset = slice_offset;
            slice_offset += f.num_slices;
        }
        let total_slices = slice_offset;

        // main packet: 64B header + 12B + 16B/file
        let pkt_len = 64 + 12 + files.len() * 16;
        let mut pkt = vec![0u8; pkt_len];
        put_u64le(&mut pkt, 64, slice_size);
        pkt[72..76].copy_from_slice(&(files.len() as u32).to_le_bytes());
        for (i, f) in files.iter().enumerate() {
            pkt[76 + 16 * i..76 + 16 * i + 16].copy_from_slice(&f.id);
        }
        info.set_id = md5_of(&[&pkt[64..]]);
        info.write_pkt_header(&mut pkt, 0, pkt_len - 64, b"PAR 2.0\0Main\0\0\0\0", false);
        let gen = RecoveryGen::new(ctx, slice_size, info.set_id, gf_opts, proc_opts);
        Ok(Par2 {
            info,
            files,
            total_slices,
            pkt_main: pkt,
            gen,
        })
    }

    /// `startChunking(recoverySlices)` - a new PAR2Chunked sharing this
    /// ctx; the caller sets its chunk size before use.
    pub fn start_chunking(&self, slices: Vec<u16>) -> JsResult<RecoveryGen<C>>
    where
        C: Clone,
    {
        if slices.is_empty() {
            return Err("Must supply recovery slices for chunked operation".into());
        }
        let mut c = RecoveryGen::new(
            self.gen.ctx.clone(),
            0,
            self.info.set_id,
            self.gen.gf_opts.clone(),
            self.gen.proc_opts.clone(),
        );
        c.set_recovery_slices(slices).map_err(|e| e.message)?;
        Ok(c)
    }

    /// `packetRecoverySize()` - per recovery packet.
    pub fn packet_recovery_size(&self) -> u64 {
        self.info.slice_size + RECOVERY_HEADER_SIZE as u64
    }

    /// `getFiles(true)` - borrow the file list (JS returns the array).
    pub fn files(&self) -> &[Par2File] {
        &self.files
    }

    /// `getFiles(false)` - consume the file list.
    pub fn take_files(&mut self) -> Vec<Par2File> {
        std::mem::take(&mut self.files)
    }

    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    pub fn packet_main_size(&self) -> usize {
        64 + 12 + self.files.len() * 16
    }

    /// `getPacketMain(keep)` - false consumes.
    pub fn get_packet_main(&mut self, keep: bool) -> Vec<u8> {
        if keep {
            self.pkt_main.clone()
        } else {
            std::mem::take(&mut self.pkt_main)
        }
    }

    pub fn packet_creator_size(&self, creator: &str) -> usize {
        64 + ceil4(str_to_ascii(creator, &self.info.ascii_charset).len())
    }

    pub fn make_packet_creator(&self, creator: &str) -> Vec<u8> {
        let c = str_to_ascii(creator, &self.info.ascii_charset);
        let len = ceil4(c.len());
        let mut pkt = vec![0u8; 64 + len];
        pkt[64..64 + c.len()].copy_from_slice(&c);
        self.info
            .write_pkt_header(&mut pkt, 0, len, b"PAR 2.0\0Creator\0", false);
        pkt
    }

    fn resolve_char_opt(&self, comment: &str, unicode: Option<CharOpt>) -> JsResult<CharOpt> {
        match unicode {
            None => Ok(if has_unicode(comment) {
                CharOpt::Both
            } else {
                CharOpt::Ascii
            }),
            Some(u) => Ok(u),
        }
    }

    pub fn packet_comment_size(&self, comment: &str, unicode: Option<CharOpt>) -> JsResult<usize> {
        let unicode = self.resolve_char_opt(comment, unicode)?;
        let len = ceil4(str_to_ascii(comment, &self.info.ascii_charset).len());
        let len2 = ceil4(utf16_units(comment) * 2);
        let mut n = 0;
        if unicode == CharOpt::Both || unicode == CharOpt::Ascii {
            n += 64 + len;
        }
        if unicode == CharOpt::Both || unicode == CharOpt::Unicode {
            n += 64 + 16 + len2;
        }
        Ok(n)
    }

    /// CommASCI (+ CommUni when unicode requires). The Uni packet's link
    /// field is the ascii packet's MD5 (bytes 16..32 of the packet).
    pub fn make_packet_comment(
        &self,
        comment: &str,
        unicode: Option<CharOpt>,
    ) -> JsResult<Vec<u8>> {
        let unicode = self.resolve_char_opt(comment, unicode)?;
        let do_ascii = unicode == CharOpt::Both || unicode == CharOpt::Ascii;
        let do_uni = unicode == CharOpt::Both || unicode == CharOpt::Unicode;
        let c = str_to_ascii(comment, &self.info.ascii_charset);
        let len = ceil4(c.len());
        let len2 = ceil4(utf16_units(comment) * 2);
        let total = (if do_ascii { 64 + len } else { 0 }) + if do_uni { 64 + 16 + len2 } else { 0 };
        let mut pkt = vec![0u8; total];
        let mut offset = 0usize;
        if do_ascii {
            pkt[64..64 + c.len()].copy_from_slice(&c);
            self.info
                .write_pkt_header(&mut pkt, 0, len, b"PAR 2.0\0CommASCI", false);
            offset = 64 + len;
        }
        if do_uni {
            let end = offset + 64 + 16 + len2;
            // JS zeroes the tail padding *before* writing the text
            pkt[end - 2] = 0;
            pkt[end - 1] = 0;
            let mut pos = offset + 64 + 16;
            for u in comment.encode_utf16() {
                pkt[pos..pos + 2].copy_from_slice(&u.to_le_bytes());
                pos += 2;
            }
            if do_ascii {
                // link: ascii packet's MD5 at [16..32)
                let link: [u8; 16] = pkt[16..32].try_into().unwrap();
                pkt[offset + 64..offset + 80].copy_from_slice(&link);
            }
            self.info.write_pkt_header(
                &mut pkt[..end],
                offset,
                16 + len2,
                b"PAR 2.0\0CommUni\0",
                false,
            );
        }
        Ok(pkt)
    }
}

// ---------------------------------------------------------------------------
// recovery generation (GFWrapper in par2.js)
// ---------------------------------------------------------------------------

/// The event/proc surface RecoveryGen needs, abstracted so that multiple
/// gens (PAR2 + PAR2Chunked) can share one backend event queue - like the
/// JS binding dispatching callbacks per GfProc instance.
pub trait GfCtx {
    type Proc: GfProc;
    fn new_proc(
        &mut self,
        slice_size: u64,
        cpu: Option<&CpuSpec>,
        ocl: &[OclSpec],
        staging: i32,
    ) -> Result<Self::Proc>;
    /// Wait for the next event belonging to `proc_id`.
    fn wait_for(&mut self, proc_id: usize) -> Result<ProcEvent>;
    /// Single non-blocking poll of `proc_id`'s mailbox. `wait_event(0)`
    /// polls once and returns immediately - used to drain InputDone
    /// buffers during the add loop so emitted-but-unconsumed events do
    /// not retain the whole input in memory.
    fn try_wait_for(&mut self, _proc_id: usize) -> Result<Option<ProcEvent>> {
        Ok(None)
    }
    fn new_output_hasher(&mut self, regions: u32) -> Result<Box<dyn HasherOutput>>;
    fn new_input_hasher(
        &mut self,
        slice_size: u64,
        block_out: Box<[u8]>,
    ) -> Result<Box<dyn HasherInput>>;
    /// Whether new input hashers run lane-split (block MD5+CRC only) and
    /// expect the read thread to deposit the whole-file MD5. Used to gate
    /// the reader's producer-side file-MD5 lane.
    fn lane_split_inhash(&self) -> bool {
        false
    }
}

/// A compute backend shared between multiple gens. Events are routed to
/// per-proc mailboxes; a gen waiting for proc A never consumes proc B's
/// events (the JS callback dispatch equivalent).
pub struct SharedBackend<B: ComputeBackend> {
    inner: std::rc::Rc<std::cell::RefCell<SharedInner<B>>>,
}

struct SharedInner<B: ComputeBackend> {
    backend: B,
    stash: std::collections::HashMap<usize, VecDeque<ProcEvent>>,
    native_hashers: bool,
    native_proc: bool,
    native_queues: parpar_gf16::EventQueues,
    /// which proc ids were created on the native path - wait_for routes
    /// per proc, not on the current flag, so mixed FFI+native procs and
    /// mid-session flag toggles can't misroute waits
    native_pids: std::collections::HashSet<usize>,
    next_native_pid: usize,
}

impl<B: ComputeBackend> SharedBackend<B> {
    pub fn new(backend: B) -> Self {
        SharedBackend {
            inner: std::rc::Rc::new(std::cell::RefCell::new(SharedInner {
                backend,
                stash: std::collections::HashMap::new(),
                native_hashers: false,
                native_proc: false,
                native_queues: Default::default(),
                native_pids: std::collections::HashSet::new(),
                next_native_pid: 1_000_000,
            })),
        }
    }

    /// Swap the FFI hashers for the native parpar-hasher implementations.
    pub fn set_native_hashers(&mut self, native: bool) {
        self.inner.borrow_mut().native_hashers = native;
    }
    pub fn native_hashers(&self) -> bool {
        self.inner.borrow().native_hashers
    }

    /// Route new_proc() to the native parpar-gf16 CPU processor instead of
    /// the FFI backend. FFI remains the solver/OpenCL host.
    pub fn set_native_proc(&mut self, native: bool) {
        self.inner.borrow_mut().native_proc = native;
    }
    pub fn native_proc(&self) -> bool {
        self.inner.borrow().native_proc
    }

    /// Access the wrapped backend (for solver/host queries like `gf_info`
    /// and `opencl_device_info`).
    pub fn with_backend<R>(&mut self, f: impl FnOnce(&mut B) -> R) -> R {
        f(&mut self.inner.borrow_mut().backend)
    }
}

impl<B: ComputeBackend> Clone for SharedBackend<B> {
    fn clone(&self) -> Self {
        SharedBackend {
            inner: self.inner.clone(),
        }
    }
}

impl<B: ComputeBackend> GfCtx for SharedBackend<B>
where
    B::InputHasher: 'static,
    B::OutputHasher: 'static,
    B::Proc: 'static,
{
    type Proc = Box<dyn GfProc>;

    fn new_proc(
        &mut self,
        slice_size: u64,
        cpu: Option<&CpuSpec>,
        ocl: &[OclSpec],
        staging: i32,
    ) -> Result<Self::Proc> {
        let mut inner = self.inner.borrow_mut();
        if inner.native_proc {
            let pid = inner.next_native_pid;
            inner.next_native_pid += 1;
            inner.native_pids.insert(pid);
            inner
                .native_queues
                .lock()
                .unwrap()
                .insert(pid, VecDeque::new());
            return crate::native_backend::native_new_proc(
                pid,
                inner.native_queues.clone(),
                slice_size,
                cpu,
                ocl,
                staging,
            );
        }
        inner
            .backend
            .new_proc(slice_size, cpu, ocl, staging)
            .map(|p| Box::new(p) as Box<dyn GfProc>)
    }

    fn wait_for(&mut self, proc_id: usize) -> Result<ProcEvent> {
        let mut inner = self.inner.borrow_mut();
        loop {
            if inner.native_pids.contains(&proc_id) {
                // native procs emit from a worker thread - poll until an
                // event lands rather than erroring on a momentarily
                // empty queue
                {
                    let mut nq = inner.native_queues.lock().unwrap();
                    if let Some(q) = nq.get_mut(&proc_id) {
                        if let Some(ev) = q.pop_front() {
                            return Ok(ev);
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
                continue;
            }
            if let Some(q) = inner.stash.get_mut(&proc_id) {
                if let Some(ev) = q.pop_front() {
                    return Ok(ev);
                }
            }
            let (pid, ev) = inner
                .backend
                .wait_event(-1)
                .ok_or_else(|| BackendError::new("backend event queue closed"))?;
            inner.stash.entry(pid).or_default().push_back(ev);
        }
    }

    fn try_wait_for(&mut self, proc_id: usize) -> Result<Option<ProcEvent>> {
        let mut inner = self.inner.borrow_mut();
        if inner.native_pids.contains(&proc_id) {
            let mut nq = inner.native_queues.lock().unwrap();
            return Ok(nq.get_mut(&proc_id).and_then(|q| q.pop_front()));
        }
        if let Some(q) = inner.stash.get_mut(&proc_id) {
            if let Some(ev) = q.pop_front() {
                return Ok(Some(ev));
            }
        }
        // single poll of the backend queue; an event for another proc is
        // stashed for its owner rather than consumed
        match inner.backend.wait_event(0) {
            Some((pid, ev)) => {
                inner.stash.entry(pid).or_default().push_back(ev);
                Ok(inner.stash.get_mut(&proc_id).and_then(|q| q.pop_front()))
            }
            None => Ok(None),
        }
    }

    fn new_output_hasher(&mut self, regions: u32) -> Result<Box<dyn HasherOutput>> {
        if self.inner.borrow().native_hashers {
            return Ok(Box::new(parpar_hasher::NativeHasherOutput::new(regions)));
        }
        Ok(Box::new(
            self.inner.borrow_mut().backend.new_output_hasher(regions)?,
        ))
    }

    fn new_input_hasher(
        &mut self,
        slice_size: u64,
        block_out: Box<[u8]>,
    ) -> Result<Box<dyn HasherInput>> {
        if self.inner.borrow().native_hashers {
            // lane-split: the sequential read thread computes the external
            // lane(s) and deposits them via set_file_md5/set_block_states;
            // with 64B-aligned slices the producer ships per-slice lane0
            // states too, leaving only CRC + finalisation here
            return Ok(Box::new(parpar_hasher::NativeHasherInput::new_ext(
                slice_size,
                block_out,
                slice_size % 64 == 0,
            )));
        }
        Ok(Box::new(
            self.inner
                .borrow_mut()
                .backend
                .new_input_hasher(slice_size, block_out)?,
        ))
    }

    fn lane_split_inhash(&self) -> bool {
        self.inner.borrow().native_hashers
    }
}

/// `_gfOpts` - hash batching and recovery-buffer pool sizing.
/// JS: hashBatchSize default 8, recDataSize default ceil(hb*1.5), then
/// hashBatchSize = min(hashBatchSize, recDataSize).
#[derive(Debug, Clone)]
pub struct GfOpts {
    pub hash_batch_size: usize,
    pub rec_data_size: usize,
}

impl GfOpts {
    pub fn resolve(hash_batch_size: Option<usize>, rec_data_size: Option<usize>) -> GfOpts {
        // JS falsy check: 0 (like undefined) falls back to the default
        let hb = hash_batch_size.filter(|&v| v > 0).unwrap_or(8);
        let rd = rec_data_size
            .filter(|&v| v > 0)
            .unwrap_or_else(|| (hb * 3 + 1) / 2);
        GfOpts {
            hash_batch_size: hb.min(rd),
            rec_data_size: rd,
        }
    }

    /// Invariant guard for hand-constructed opts (fields are public):
    /// hash_batch_size == 0 would div-by-zero downstream.
    fn normalized(&self) -> GfOpts {
        GfOpts {
            hash_batch_size: self.hash_batch_size.max(1),
            rec_data_size: self.rec_data_size.max(self.hash_batch_size.max(1)),
        }
    }
}

/// The `_gfOpts.proc_*` + threads spec used for `gf_init`.
#[derive(Debug, Clone, Default)]
pub struct ProcOpts {
    pub cpu: Option<CpuSpec>,
    pub ocl: Vec<OclSpec>,
    pub staging: i32,
    pub threads: Option<i32>,
}

struct AddItem {
    slice_num: i64,
    buf: SliceBuf,
}

/// Synchronous analogue of the GFWrapper fetch/hash interlock. The JS
/// version is callback-driven (queue drains on progress events, refetches
/// deferred via setImmediate, batched HasherOutput callbacks); here
/// `wait_for` supplies the same sequencing with identical pool/refcount
/// bookkeeping.
pub struct RecoveryGen<C: GfCtx> {
    pub ctx: C,
    proc_opts: ProcOpts,
    gf: Option<C::Proc>,
    proc_id: usize,
    alloc_size: u64,
    /// JS `chunkSize` - the meaningful length of recovery data (`sliceSize`
    /// for the non-chunked wrapper).
    pub chunk_size: u64,
    pub set_id: [u8; 16],
    gf_opts: GfOpts,
    pub recovery_slices: Vec<u16>,

    add_queue: VecDeque<AddItem>,
    progress_set: bool,
    next_token: u64,
    ended: bool,

    // --- recData machinery ---
    rec_data: Vec<Option<SliceBuf>>,
    refcount: Vec<u32>,
    fetch_done: Vec<bool>,
    hash_done: Vec<bool>,
    group_finalized: Vec<bool>,
    hashers: Vec<Option<Box<dyn HasherOutput>>>,
    hash_bufs_needed: Vec<usize>,
    rec_md5: Vec<u8>,
    rec_ptr: usize,
    pending_outputs: std::collections::HashMap<u64, usize>,
}

impl<C: GfCtx> RecoveryGen<C> {
    /// `alloc_size` = JS `_allocSize` (PAR2: sliceSize; chunked: set later
    /// via set_chunk_size).
    pub fn new(
        ctx: C,
        alloc_size: u64,
        set_id: [u8; 16],
        gf_opts: GfOpts,
        proc_opts: ProcOpts,
    ) -> RecoveryGen<C> {
        RecoveryGen {
            ctx,
            proc_opts,
            gf: None,
            proc_id: 0,
            alloc_size,
            chunk_size: alloc_size,
            set_id,
            gf_opts: gf_opts.normalized(),
            recovery_slices: Vec::new(),
            add_queue: VecDeque::new(),
            progress_set: false,
            next_token: 1,
            ended: false,
            rec_data: Vec::new(),
            refcount: Vec::new(),
            fetch_done: Vec::new(),
            hash_done: Vec::new(),
            group_finalized: Vec::new(),
            hashers: Vec::new(),
            hash_bufs_needed: Vec::new(),
            rec_md5: Vec::new(),
            rec_ptr: 0,
            pending_outputs: std::collections::HashMap::new(),
        }
    }

    fn token(&mut self) -> u64 {
        let t = self.next_token;
        self.next_token += 1;
        t
    }

    /// `gf_init(size)` - JS closes any existing proc, creates a new one
    /// and applies thread count when a CPU backend is configured.
    fn gf_init(&mut self, size: u64) -> Result<()> {
        if let Some(mut g) = self.gf.take() {
            let _ = g.close();
        }
        self.add_queue.clear();
        let mut proc = self.ctx.new_proc(
            size,
            self.proc_opts.cpu.as_ref(),
            &self.proc_opts.ocl,
            self.proc_opts.staging,
        )?;
        if self.proc_opts.cpu.is_some() {
            if let Some(t) = self.proc_opts.threads {
                proc.set_num_threads(t)?;
            }
        }
        self.proc_id = proc.proc_id();
        self.alloc_size = size;
        self.gf = Some(proc);
        Ok(())
    }

    /// `setRecoverySlices(slices)` - slices is the JS array form
    /// (caller converts a count to range(0,n)).
    pub fn set_recovery_slices(&mut self, slices: Vec<u16>) -> Result<()> {
        self.recovery_slices = slices;
        if !self.recovery_slices.is_empty() {
            if self.gf.is_none() && self.alloc_size > 0 {
                self.gf_init(self.alloc_size)?;
            }
            if let Some(g) = self.gf.as_mut() {
                let n = self.recovery_slices.len();
                g.set_recovery_slices(Some(&self.recovery_slices), n)?;
            }
        } else {
            if let Some(g) = self.gf.as_mut() {
                g.free_mem()?;
            }
            self.alloc_size = 0;
        }
        Ok(())
    }

    /// `PAR2Chunked.setChunkSize`.
    pub fn set_chunk_size(&mut self, size: u64) -> Result<()> {
        if size % 2 != 0 {
            return Err(BackendError::new("Chunk size must be a multiple of 2"));
        }
        if self.recovery_slices.is_empty() {
            return Err(BackendError::new(
                "Generating no recovery - cannot set chunk size",
            ));
        }
        if size > 0 {
            self.chunk_size = size;
            if self.alloc_size == 0 || size > self.alloc_size {
                // requested size is larger than allocated - reallocate
                self.gf_init(size)?;
            }
            let gf = self.gf.as_mut().unwrap();
            gf.set_current_slice_size(size)?;
            let n = self.recovery_slices.len();
            gf.set_recovery_slices(Some(&self.recovery_slices), n)?;
        } else {
            // setting size to 0 indicates a clear
            self.alloc_size = 0;
            self.chunk_size = 0;
            if let Some(g) = self.gf.as_mut() {
                g.free_mem()?;
            }
        }
        Ok(())
    }

    /// `gf_info` for the live proc (JS `g.gf_info()`).
    pub fn gf_info(&mut self) -> Result<parpar_backend::GfInfo> {
        match self.gf.as_mut() {
            Some(g) => g.info(),
            None => Err(BackendError::new("GF context not initialised")),
        }
    }

    /// `processSlice(data, sliceNum)` - noop when no recovery generated.
    pub fn process_slice(&mut self, buf: SliceBuf, slice_num: i64) -> Result<()> {
        if !self.recovery_slices.is_empty() {
            self.buffered_process(slice_num, buf, self.chunk_size as usize)
        } else {
            Ok(())
        }
    }

    /// `processData(fileOrNum, data)` on the wrapper (file variant lives
    /// on Par2File::process_data which computes the slice number).
    pub fn process_data(&mut self, slice_num: i64, buf: SliceBuf) -> Result<()> {
        self.buffered_process(slice_num, buf, self.chunk_size as usize)
    }

    /// `bufferedProcess(dataSlice, sliceNum, len, cb)` - synchronous
    /// equivalent; blocks (backpressure) only when staging is full.
    fn buffered_process(&mut self, slice_num: i64, mut buf: SliceBuf, len: usize) -> Result<()> {
        if len == 0 || buf.data.is_empty() {
            return Ok(());
        }
        // bufferSlice.call(dataSlice, 0, len) clamps to the view length -
        // buf.len is the caller's declared byte count; the storage may be
        // a larger pooled buffer with stale bytes past len
        buf.len = len.min(buf.len).min(buf.data.len());
        if !self.progress_set {
            if let Some(g) = self.gf.as_mut() {
                g.set_progress(true)?;
            }
            self.progress_set = true;
        }
        if !self.add_queue.is_empty() {
            // queue non-empty: preserve submission order
            self.add_queue.push_back(AddItem { slice_num, buf });
            return self.pump_adds();
        }
        let t = self.token();
        match self
            .gf
            .as_mut()
            .ok_or_else(|| BackendError::new("GF context not initialised"))?
            .add_input(buf, slice_num as i32, t)?
        {
            AddResult::Accepted => self.drain_ready(),
            AddResult::Full(b) => {
                self.add_queue.push_back(AddItem { slice_num, buf: b });
                self.pump_adds()
            }
        }
    }

    /// Drain already-queued events without blocking. Emitted InputDone
    /// events carry their input buffer - if they pile up unconsumed the
    /// mailbox retains the whole input (JS consumed events on the event
    /// loop continuously). Called after each successful add to keep the
    /// queue shallow.
    fn drain_ready(&mut self) -> Result<()> {
        while let Some(ev) = self.ctx.try_wait_for(self.proc_id)? {
            self.handle_event(ev)?;
        }
        Ok(())
    }

    /// Retry queue head on each staging-freeing event until drained -
    /// the sync equivalent of the JS setProgressCb drain loop.
    fn pump_adds(&mut self) -> Result<()> {
        loop {
            while let Some(item) = self.add_queue.pop_front() {
                let t = self.token();
                match self
                    .gf
                    .as_mut()
                    .unwrap()
                    .add_input(item.buf, item.slice_num as i32, t)?
                {
                    AddResult::Accepted => {}
                    AddResult::Full(b) => {
                        self.add_queue.push_front(AddItem {
                            slice_num: item.slice_num,
                            buf: b,
                        });
                        break;
                    }
                }
            }
            if self.add_queue.is_empty() {
                return Ok(());
            }
            self.pump_one()?;
        }
    }

    /// Apply one event's side effects (routing mirrors the JS callback
    /// dispatch). Returns when an event was consumed.
    fn pump_one(&mut self) -> Result<()> {
        let ev = self.ctx.wait_for(self.proc_id)?;
        self.handle_event(ev)
    }

    /// Event routing shared by the blocking pump and the non-blocking
    /// drain.
    fn handle_event(&mut self, ev: ProcEvent) -> Result<()> {
        match ev {
            ProcEvent::InputDone { buf, .. } => {
                // JS invokes the submitter's callback to release the buffer;
                // our SliceBuf is an owned copy so dropping it is the
                // equivalent release.
                drop(buf);
            }
            ProcEvent::Progress { .. } => {}
            ProcEvent::Ended { .. } => {
                self.ended = true;
            }
            ProcEvent::OutputDone {
                token,
                cksum_valid,
                buf,
            } => {
                self.on_output(token, cksum_valid, buf)?;
            }
            ProcEvent::Error { message, .. } => return Err(BackendError::new(message)),
        }
        Ok(())
    }

    /// `finish(files, cb)` - JS queues a finish marker after pending adds;
    /// here pump_adds drains first, then end_input + wait for Ended, then
    /// start the recData machinery.
    pub fn finish(&mut self) -> Result<()> {
        self.pump_adds()?;
        if self.recovery_slices.is_empty() {
            return Ok(());
        }
        let t = self.token();
        self.ended = false; // await a fresh Ended for this pass
        self.gf
            .as_mut()
            .ok_or_else(|| BackendError::new("GF proc not initialised"))?
            .end_input(t)?;
        while !self.ended {
            self.pump_one()?;
        }
        self.pull_rec_data()
    }

    /// `finish(files, cb)` - JS resets per-file position counters before
    /// ending input.
    pub fn finish_files(&mut self, files: &mut [Par2File]) -> Result<()> {
        for f in files.iter_mut() {
            f.reset_positions();
        }
        self.finish()
    }

    /// `_pullRecData` - allocate the recData pool + per-group hashers
    /// primed with the packet prefix (setID || RecvSlic || exponent u32),
    /// then kick off a fetch per pool slot.
    fn pull_rec_data(&mut self) -> Result<()> {
        let n = self.recovery_slices.len();
        let hb = self.gf_opts.hash_batch_size;
        let pool_n = n.min(self.gf_opts.rec_data_size);
        self.rec_data = (0..pool_n).map(|_| None).collect();
        self.refcount = vec![0; pool_n];
        self.fetch_done = vec![false; n];
        self.hash_done = vec![false; n];
        self.rec_md5 = vec![0u8; 16 * n.min(hb)];
        self.rec_ptr = 0;

        let groups = n.div_ceil(hb);
        // hashers persist until all groups finalized (recDataHashers)
        if self.hashers.iter().all(Option::is_none) {
            self.hashers = Vec::with_capacity(groups);
            for g in 0..groups {
                let base = g * hb;
                let num_bufs = hb.min(n - base);
                let mut h = self.ctx.new_output_hasher(num_bufs as u32)?;
                let bufs: Vec<Vec<u8>> = (0..num_bufs)
                    .map(|i| {
                        let mut b = Vec::with_capacity(36);
                        b.extend_from_slice(&self.set_id);
                        b.extend_from_slice(b"PAR 2.0\0RecvSlic");
                        b.extend_from_slice(&(self.recovery_slices[base + i] as u32).to_le_bytes());
                        b
                    })
                    .collect();
                let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
                h.update(&refs)?;
                self.hashers.push(Some(h));
            }
        }
        self.group_finalized = vec![false; groups];
        self.hash_bufs_needed = (0..groups).map(|g| hb.min(n - g * hb)).collect();
        self.pending_outputs.clear();
        for i in 0..pool_n {
            self.start_fetch(i)?;
        }
        Ok(())
    }

    /// `_fetchRecData(idx)` - refcount the slot, (re)use its buffer, get.
    fn start_fetch(&mut self, idx: usize) -> Result<()> {
        if idx >= self.recovery_slices.len() {
            return Ok(());
        }
        let slot = idx % self.rec_data.len();
        self.refcount[slot] = 2; // ref'd by hasher + user disposal
        let buf = match self.rec_data[slot].take() {
            Some(mut b) => {
                b.len = self.chunk_size as usize;
                b
            }
            None => SliceBuf::new(
                vec![0u8; self.alloc_size as usize].into_boxed_slice(),
                self.chunk_size as usize,
            ),
        };
        let t = self.token();
        self.pending_outputs.insert(t, idx);
        if let Err(e) = self.gf.as_mut().unwrap().get_output(idx as i32, buf, t) {
            self.pending_outputs.remove(&t);
            return Err(e);
        }
        Ok(())
    }

    /// The `gf.get` completion callback: stash the buffer, run the hash
    /// group countdown and (sync) hasher update on group completion.
    fn on_output(&mut self, token: u64, cksum_valid: bool, buf: SliceBuf) -> Result<()> {
        let idx = match self.pending_outputs.remove(&token) {
            Some(i) => i,
            // a stray/duplicate OutputDone must not kill the process
            None => return Err(BackendError::new("output event with no pending fetch")),
        };
        if !cksum_valid {
            eprintln!(
                "Memory checksum error detected in recovery slice {} -  recovery data is likely corrupt. This is likely due to hardware memory corruption or a bug in ParPar.",
                self.recovery_slices[idx]
            );
        }
        let slot = idx % self.rec_data.len();
        self.rec_data[slot] = Some(buf);
        self.fetch_done[idx] = true;

        let hb = self.gf_opts.hash_batch_size;
        let group = idx / hb;
        self.hash_bufs_needed[group] -= 1;
        if self.hash_bufs_needed[group] == 0 {
            let hash_base = group * hb;
            let num_bufs = hb.min(self.recovery_slices.len() - hash_base);
            // collect group buffers: recData[(hashBaseIdx+i) % pool]
            let pool = self.rec_data.len();
            let refs: Vec<&[u8]> = (0..num_bufs)
                .map(|i| {
                    let slot = (hash_base + i) % pool;
                    self.rec_data[slot].as_ref().unwrap().as_bytes()
                })
                .collect();
            match self.hashers[group].as_mut() {
                Some(h) => h.update(&refs)?,
                None => {
                    return Err(BackendError::new(
                        "recovery hasher for group already finalized",
                    ))
                }
            }
            for i in 0..num_bufs {
                let ridx = hash_base + i;
                self.hash_done[ridx] = true;
                self.mark_consumed(ridx)?;
            }
        }
        Ok(())
    }

    /// `_markRecDataConsumed(idx)` - release one ref; at zero the slot is
    /// reused for idx+pool or freed.
    pub fn mark_consumed(&mut self, idx: usize) -> Result<()> {
        if self.rec_data.is_empty() {
            return Err(BackendError::new("Recovery data not initialized"));
        }
        let slot = idx % self.rec_data.len();
        if self.refcount[slot] == 0 {
            return Err(BackendError::new("Recovery data already consumed"));
        }
        self.refcount[slot] -= 1;
        if self.refcount[slot] == 0 {
            let next = idx + self.rec_data.len();
            if next < self.recovery_slices.len() {
                // JS defers via setImmediate to avoid recursion; a direct
                // call is fine since start_fetch doesn't reenter here
                self.start_fetch(next)?;
            } else {
                self.rec_data[slot] = None;
            }
        }
        Ok(())
    }

    /// Block until recovery idx's data is fetched.
    fn ensure_fetched(&mut self, idx: usize) -> Result<()> {
        while !self.fetch_done[idx] {
            self.pump_one()?;
        }
        Ok(())
    }

    /// `getNextRecoveryData` - returns the recovery index in sequence, or
    /// None when all recovery data fetched (JS throws; the caller-facing
    /// iterator shape is more natural here).
    pub fn next_recovery(&mut self) -> Result<Option<usize>> {
        if self.rec_ptr >= self.recovery_slices.len() {
            return Ok(None);
        }
        if self.rec_data.is_empty() || self.fetch_done.is_empty() {
            return Err(BackendError::new(
                "Recovery data requested before processing finished",
            ));
        }
        let idx = self.rec_ptr;
        self.ensure_fetched(idx)?;
        self.rec_ptr += 1;
        Ok(Some(idx))
    }

    /// The fetched recovery data for `idx` (a chunk_size slice of the
    /// slot buffer). Valid until the next `&mut self` call that refetches
    /// or releases the slot.
    pub fn recovery_data(&self, idx: usize) -> Result<&[u8]> {
        if self.rec_data.is_empty() {
            return Err(BackendError::new("Recovery data not initialized"));
        }
        let slot = idx % self.rec_data.len();
        self.rec_data[slot]
            .as_ref()
            .map(|b| b.as_bytes())
            .ok_or_else(|| BackendError::new("Recovery slice not fetched"))
    }

    /// `_getRecoveryPacketMD5(idx, cb)` - waits for the group hash, then
    /// finalizes the group's hasher on first request.
    pub fn recovery_md5(&mut self, idx: usize) -> Result<[u8; 16]> {
        if idx >= self.recovery_slices.len() {
            return Err(BackendError::new("Invalid packet index"));
        }
        if idx >= self.rec_ptr {
            return Err(BackendError::new(
                "Can't get MD5 before corresponding data fetched",
            ));
        }
        let hb = self.gf_opts.hash_batch_size;
        let group = idx / hb;
        while !self.hash_done[idx] {
            self.pump_one()?;
        }
        if !self.group_finalized[group] {
            if let Some(mut h) = self.hashers[group].take() {
                h.get(&mut self.rec_md5)?;
                self.group_finalized[group] = true;
            }
        }
        let off = 16 * (idx % hb);
        Ok(self.rec_md5[off..off + 16].try_into().unwrap())
    }

    /// `getRecoveryPacketHeader` equivalent: (exponent, 68B header).
    /// `slice_size` is the PAR2 slice size (packet length field), not the
    /// chunk size - matching _writeRecoveryHeader using this.sliceSize.
    pub fn recovery_packet_header(
        &mut self,
        idx: usize,
        slice_size: u64,
    ) -> Result<[u8; RECOVERY_HEADER_SIZE]> {
        let md5 = self.recovery_md5(idx)?;
        let mut pkt = [0u8; RECOVERY_HEADER_SIZE];
        pkt[..8].copy_from_slice(MAGIC);
        put_u64le(&mut pkt, 8, slice_size + 68);
        pkt[16..32].copy_from_slice(&md5);
        pkt[32..48].copy_from_slice(&self.set_id);
        pkt[48..64].copy_from_slice(b"PAR 2.0\0RecvSlic");
        pkt[64..68].copy_from_slice(&(self.recovery_slices[idx] as u32).to_le_bytes());
        Ok(pkt)
    }

    /// `_isRecoveryProcessed` - all slots at refcount 0.
    pub fn recovery_processed(&self) -> bool {
        self.refcount.iter().all(|&r| r == 0)
    }

    /// `waitForRecoveryComplete` - pump until fully consumed.
    pub fn wait_recovery_complete(&mut self) -> Result<()> {
        // JS dereferences recDataRefcount unconditionally - a never-
        // started gen throws there; empty refcount is that state here
        if self.refcount.is_empty() {
            return Err(BackendError::new("Recovery processing not started"));
        }
        while !self.recovery_processed() {
            self.pump_one()?;
        }
        Ok(())
    }

    /// `gf.close` equivalent.
    pub fn close(&mut self) -> Result<()> {
        if let Some(mut g) = self.gf.take() {
            g.close()?;
        }
        self.add_queue.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Par2File data/hash processing (PAR2File.prototype.process*)
// ---------------------------------------------------------------------------

impl Par2File {
    /// Create the file's HasherInput with block_out = pktCheck records
    /// region (`new binding.HasherInput(sliceSize, pktCheck.slice(80))`).
    pub fn init_hasher<C: GfCtx>(&mut self, ctx: &mut C, slice_size: u64) -> Result<()> {
        if self.size == 0 || self.hasher.is_some() {
            return Ok(());
        }
        let block = self.take_checksum_records().unwrap_or_default();
        self.hasher = Some(ctx.new_input_hasher(slice_size, block)?);
        Ok(())
    }

    /// Deposit a whole-file MD5 produced by the read thread (lane-split
    /// mode) - forwards to the live input hasher before end() consumes it.
    pub fn set_file_md5(&mut self, md5: &[u8; 16]) {
        if let Some(h) = self.hasher.as_mut() {
            h.set_file_md5(md5);
        }
    }

    /// Deposit per-slice lane0 chain states produced by the read thread
    /// (lane-split states mode) - appended to the hasher's state queue in
    /// stream order, before the chunks that complete those slices hash.
    pub fn set_block_states(&mut self, states: &[[u32; 4]]) {
        if let Some(h) = self.hasher.as_mut() {
            h.set_block_states(states);
        }
    }

    /// `processHash(data, cb)` - feed the file hasher; at end-of-file the
    /// hasher finalizes the whole-file MD5 and returns the block records.
    pub fn process_hash(&mut self, data: &[u8]) -> Result<()> {
        if self.hash_pos + data.len() as u64 > self.size {
            return Err(BackendError::new("Too much data given to hash"));
        }
        self.hash_pos += data.len() as u64;
        // !pktCheck => no hasher (JS cb() early return)
        if self.pkt_check.is_none() || self.size == 0 {
            return Ok(());
        }
        let at_end = self.hash_pos == self.size;
        let h = self
            .hasher
            .as_mut()
            .expect("hasher missing for non-empty file");
        h.update(data)?;
        if at_end {
            let mut md5 = [0u8; 16];
            let block = self.hasher.take().unwrap().end(&mut md5)?;
            self.md5 = Some(md5);
            self.put_checksum_records(block);
        }
        Ok(())
    }

    /// `processData(data, cb)` - validate slice length, compute the global
    /// slice number and submit to the wrapper.
    pub fn process_data<C: GfCtx>(
        &mut self,
        gen: &mut RecoveryGen<C>,
        data: SliceBuf,
    ) -> Result<()> {
        if self.slice_data_pos >= self.num_slices {
            return Err(BackendError::new("Too many slices given"));
        }
        let buf_len = data.len as u64;
        if self.slice_data_pos == self.num_slices - 1 {
            let mut last_len = self.size % self.slice_size;
            if last_len == 0 {
                last_len = self.slice_size;
            }
            if buf_len != last_len {
                return Err(BackendError::new("Last data slice has mismatched length"));
            }
        } else if buf_len != self.slice_size {
            return Err(BackendError::new(
                "Given data does not match declared slice size",
            ));
        }
        let slice_num = self.slice_offset + self.slice_data_pos;
        self.slice_data_pos += 1;
        gen.process_slice(data, slice_num as i64)
    }

    /// `finish(files)` resets per-pass position counters.
    pub fn reset_positions(&mut self) {
        self.hash_pos = 0;
        self.slice_data_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// method tables (must match the native module's ordering)
// ---------------------------------------------------------------------------

pub const GF_METHODS: &[&str] = &[
    "",
    "lookup",
    "lookup-sse",
    "3p_lookup",
    "shuffle-neon",
    "shuffle128-sve",
    "shuffle128-sve2",
    "shuffle2x128-sve2",
    "shuffle512-sve2",
    "shuffle128-rvv",
    "shuffle-sse",
    "shuffle-avx",
    "shuffle-avx2",
    "shuffle-avx512",
    "shuffle-vbmi",
    "shuffle2x-avx2",
    "shuffle2x-avx512",
    "xor-sse",
    "xorjit-sse",
    "xorjit-avx2",
    "xorjit-avx512",
    "affine-sse",
    "affine-avx2",
    "affine-avx512",
    "affine2x-sse",
    "affine2x-avx2",
    "affine2x-avx512",
    "affine-bmm",
    "clmul-neon",
    "clmul-sha3",
    "clmul-sve2",
    "clmul-rvv",
];

pub const GFOCL_METHODS: &[&str] = &[
    "",
    "lookup",
    "lookup_half",
    "lookup_nc",
    "lookup_half_nc",
    "lookup_grp2",
    "lookup_grp2_nc",
    "shuffle",
    "log",
    "log_small",
    "log_small2",
    "log_tiny",
    "log_small_lm",
    "log_tiny_lm",
    "by2",
];

pub const INHASH_METHODS: &[&str] = &["scalar", "simd", "crc", "simd-crc", "bmi", "avx512"];

pub const OUTHASH_METHODS: &[&str] = &[
    "scalar", "sse", "avx2", "xop", "avx512f", "avx512vl", "neon", "sve2",
];

/// `getMethodNum` - name to index; JS also accepts a validated number.
pub fn method_num(table: &[&str], name: &str) -> JsResult<i32> {
    table
        .iter()
        .position(|&m| m == name)
        .map(|i| i as i32)
        .ok_or_else(|| format!("Unknown method \"{}\"", name))
}

/// `getMethodNum` - numbers check `index in table` (integer in bounds);
/// names use indexOf.
pub fn method_num_ref(table: &[&str], m: &crate::args::MethodRef) -> JsResult<i32> {
    match m {
        crate::args::MethodRef::Index(n) => {
            if *n >= 0.0 && n.fract() == 0.0 && (*n as usize) < table.len() {
                Ok(*n as i32)
            } else {
                Err(format!(
                    "Unknown method index {}",
                    crate::compat::fmt_num(*n)
                ))
            }
        }
        crate::args::MethodRef::Name(s) => method_num(table, s),
    }
}

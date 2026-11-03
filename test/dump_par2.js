// Differential oracle: dumps packet bytes produced by lib/par2.js so the
// Rust par2.rs model can be compared byte-for-byte.
// Usage: node test/dump_par2.js > rust/fixtures/par2_packets.json
"use strict";
const ParPar = require('../lib/par2.js');
const crypto = require('crypto');

const sliceSize = 65536;

function h(seed) { // deterministic 16B patterns
	return crypto.createHash('md5').update('seed-' + seed).digest();
}
function hex(b) { return Buffer.from(b).toString('hex'); }

const files = [
	{md5_16k: h('a'), size: 0, displayName: 'empty.txt'},
	{md5_16k: h('b'), size: sliceSize * 2 + 123, displayName: 'data file.bin'},
	{md5_16k: h('c'), size: sliceSize, displayName: 'ünïcode fíle.dat'},
	{md5_16k: h('d'), size: 100, displayName: 'nul\u0000name'},
];

var par = new ParPar.PAR2(files, sliceSize);
var out = {};

out.setID = hex(par.setID);
out.sliceSize = par.sliceSize;
out.totalSlices = par.totalSlices;
out.fileOrder = par.files.map(f => f.displayName);
out.fileIds = par.files.map(f => hex(f.id));
out.sliceOffsets = par.files.map(f => f.sliceOffset);
out.main = hex(par.getPacketMain(true));
out.creator = hex(par.makePacketCreator('ParPar v0.4.6 test äöü'));
out.creatorAsciiOnly = hex(par.makePacketCreator('plain'));

['test comment', 'ünïcode çømment ✓', 'emoji \ud83d\ude80 surrogate', ''].forEach(function(c, i) {
	out['comment' + i + '_auto'] = hex(par.makePacketComment(c));
	out['comment' + i + '_ascii'] = hex(par.makePacketComment(c, ParPar.CHAR.ASCII));
	out['comment' + i + '_uni'] = hex(par.makePacketComment(c, ParPar.CHAR.UNICODE));
	out['comment' + i + '_both'] = hex(par.makePacketComment(c, ParPar.CHAR.BOTH));
	out['comment' + i + '_size_auto'] = par.packetCommentSize(c);
});

// filedesc requires md5 - inject deterministic values like a finished hasher would
par.files.forEach(function(f, i) {
	f.md5 = h('filemd5-' + f.displayName);
	out['filedesc' + i + '_auto'] = hex(f.makePacketDescription());
	out['filedesc' + i + '_ascii'] = hex(f.makePacketDescription(ParPar.CHAR.ASCII));
	out['filedesc' + i + '_both'] = hex(f.makePacketDescription(ParPar.CHAR.BOTH));
	out['filedesc' + i + '_size'] = f.packetDescriptionSize();
	out['uniname' + i] = hex(f.makePacketUniName());
	// fill pktCheck records as the hasher would, then emit IFSC
	if(f.pktCheck) {
		for(var j = 80; j < f.pktCheck.length; j++)
			f.pktCheck[j] = (j * 7 + i) & 0xff;
		out['ifsc' + i] = hex(f.getPacketChecksums(true));
	} else {
		out['ifsc' + i] = hex(f.getPacketChecksums(true));
	}
});

// recovery headers for a few exponents
[0, 1, 32767, 65535].forEach(function(exp) {
	out['recvhdr_' + exp] = hex(par.getRecoveryHeader(exp, h('recmd5-' + exp)));
});

console.log(JSON.stringify(out, null, 1));

// Differential oracle for the GFWrapper recovery lifecycle: feeds
// deterministic slices, dumps recovery data + packet MD5s.
// Usage: node test/dump_recdata.js > rust/fixtures/recdata.json
"use strict";
const ParPar = require('../lib/par2.js');

const sliceSize = 32768;
const nIn = 9, nRec = 5;

function sliceData(num) { // deterministic content, sliceSize bytes
	var b = Buffer.alloc(sliceSize);
	for(var i = 0; i < sliceSize; i++)
		b[i] = (i * 31 + num * 17) & 0xff;
	return b;
}

var par = new ParPar.PAR2([], sliceSize, {
	proc_cpu: {method: 0, chunk_size: 0, input_batchsize: 4}
});
par.setRecoverySlices(nRec);

var pending = nIn;
for(var i = 0; i < nIn; i++)
	(function(num) {
		par.processSlice(sliceData(num), num, function() {
			if(--pending == 0) finish();
		});
	})(i);

function finish() {
	par.finish(function() {
		var out = {md5s: [], datas: []};
		var idx = 0;
		(function next() {
			if(idx >= nRec) {
				console.log(JSON.stringify(out));
				process.exit(0);
			}
			var cur = idx++;
			par.getNextRecoveryData(function(i2, rec) {
				out.datas.push(rec.data.toString('hex'));
				rec.getMD5(function(md5) {
					out.md5s.push(md5.toString('hex'));
					rec.release();
					next();
				});
			});
		})();
	});
}

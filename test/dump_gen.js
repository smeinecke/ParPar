// Differential oracle: JSON case on stdin -> PAR2Gen layout dump on stdout.
// Case: {files:[{name,size,md5_16k?}], slice_def:<spec>, opts:{...}, sys:{...}}
var fs = require('fs');
var os = require('os');
var path = require('path');
var Par2Gen = require('../lib/par2gen.js');
require('../lib/par2.js'); // ensure binding loaded for PAR2's gf_info/opencl_device_info

var input = JSON.parse(fs.readFileSync(0, 'utf8'));

// mock machine facts
if (input.sys) {
	if ('total' in input.sys) os.totalmem = function() { return input.sys.total; };
	if ('free' in input.sys) os.freemem = function() { return input.sys.free; };
	if ('constrained' in input.sys) process.constrainedMemory = function() { return input.sys.constrained; };
}

var files = (input.files || []).map(function(f) {
	return {
		name: f.name,
		displayName: f.displayName !== undefined ? f.displayName : undefined,
		fullPath: f.fullPath,
		size: f.size,
		md5_16k: Buffer.from(f.md5_16k || '00000000000000000000000000000000', 'hex')
	};
});
// drop explicit-undefined displayName keys so `'displayName' in file` is false
files.forEach(function(f) { if (f.displayName === undefined) delete f.displayName; if (f.fullPath === undefined) delete f.fullPath; });

var out;
try {
	var g = new Par2Gen.PAR2Gen(files, input.slice_def, input.opts || {});
	var gfopts = g.par2._gfOpts;
	out = {
		sliceSize: g.opts.sliceSize,
		inputSlices: g.inputSlices,
		recoverySlices: g.opts.recoverySlices,
		chunkSize: g._chunkSize,
		chunks: g.chunks,
		passes: g.passes,
		slicesPerPass: g.slicesPerPass,
		readSize: g.readSize,
		procInStagingBufferCount: g.procInStagingBufferCount,
		stagingCount: gfopts.stagingCount,
		cpuRatio: g.cpuRatio,
		minMemoryLimit: g.minMemoryLimit,
		processBatchSize: g.opts.processBatchSize,
		recDataSize: g.opts.recDataSize,
		outputFileMaxSlices: g.opts.outputFileMaxSlices,
		outputFirstFileSlices: g.opts.outputFirstFileSlices === undefined ? null : g.opts.outputFirstFileSlices,
		outputFileCount: g.opts.outputFileCount,
		memoryLimit: g.opts.memoryLimit,
		sliceNums: g._sliceNums,
		recoveryFiles: g.recoveryFiles.map(function(f) {
			return {
				name: f.opts ? f.opts.name : f.name,
				recoverySlices: f.recoverySlices,
				recoveryIndex: f.recoveryIndex,
				totalSize: f.totalSize,
				packets: f.packets.map(function(p) {
					return {type: p.type, size: p.size, index: p.index === null ? -1 : p.index, dataChunkOffset: p.dataChunkOffset};
				})
			};
		}),
		openclDevices: g.opts.openclDevices,
		procCpu: gfopts.proc_cpu,
		procOcl: gfopts.proc_ocl,
		displayNames: g.files.map(function(f) { return f.displayName; })
	};
} catch (x) {
	out = {error: x.message};
}
fs.writeSync(1, JSON.stringify({
	sys: {total: os.totalmem(), free: os.freemem(), constrained: process.constrainedMemory(), kmax: require('buffer').kMaxLength, is64: ['arm64','ppc64','x64'].indexOf(process.arch) > -1},
	out: out
}) + '\n');
process.exit(0); // binding keeps the uv loop alive

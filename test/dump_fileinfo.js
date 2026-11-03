// Differential oracle for fileInfo scanning.
// Usage: node test/dump_fileinfo.js <recurse> <skipSymlinks> <files...>
"use strict";
const util = require('util');
const ParParGen = require('../lib/par2gen.js');

const recurse = JSON.parse(process.argv[2]);
const skip = JSON.parse(process.argv[3]);
const files = process.argv.slice(4);

ParParGen.fileInfo(files, recurse, skip, 2, function(err, info) {
	var out;
	if(err) {
		// console.error(err) renders fs errors in util.inspect style and
		// plain Errors as 'Error: msg' + stack (first line compared)
		if(err && err.code)
			out = {error: util.inspect(err)};
		else
			out = {error: 'Error: ' + (err.message || err)};
	} else {
		out = {results: info.map(function(f) {
			return {
				name: f.name,
				size: f.size,
				md5_16k: f.md5_16k.toString('hex'),
				md5: f.md5 ? f.md5.toString('hex') : null
			};
		}).sort(function(a,b){ return a.name < b.name ? -1 : 1; })};
	}
	console.log(JSON.stringify(out));
	setTimeout(function(){ process.exit(0); }, 200);
});

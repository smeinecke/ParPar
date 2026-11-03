"use strict";
// Differential test harness: evaluates bin/parpar.js in-process, dumping the
// parsed argv/ppo/inputFiles state right before generation would begin.
// Usage: node test/dump_args.js -- <parpar args...>
// Output: single-line JSON on stdout {argv,ppo,inputFiles} on success, or
// {error,exit} when an error path or early exit (help/version/opencl-list)
// is taken. stderr output (error messages, help text) passes through.

var fs = require('fs');
var path = require('path');

var dashdash = process.argv.indexOf('--');
var passthrough = dashdash >= 0 ? process.argv.slice(dashdash + 1) : [];

var src = fs.readFileSync(path.join(__dirname, '..', 'bin', 'parpar.js'), 'utf8');

var marker = '\tvar startTime = Date.now();';
if (src.indexOf(marker) < 0) {
	process.stderr.write('dump_args.js: injection marker not found\n');
	process.exit(2);
}
var injected =
	'\tglobal.__dump_args_result({argv: argv, ppo: ppo, inputFiles: inputFiles});\n';
src = src.replace(marker, injected + marker);

// stub exit so early-exit paths can be observed instead of terminating
var realExit = process.exit.bind(process);
global.__dump_args_result = function (o) {
	process.stdout.write(JSON.stringify(o) + '\n');
	realExit(0);
};
process.exit = function (code) {
	throw { __exit: code === undefined ? 0 : code };
};

// record console.error lines alongside letting them pass through to stderr
var errLines = [];
var realErr = console.error.bind(console);
console.error = function (m) {
	errLines.push('' + m);
	realErr(m);
};

// error()/exit paths inside the async input-list callback arrive here
process.on('uncaughtException', function (e) {
	if (e && typeof e.__exit === 'number') realExit(e.__exit);
	if (e && e.__dumped) realExit(0);
	process.stdout.write(
		JSON.stringify({ error: String((e && e.message) || e), stderr: errLines }) + '\n'
	);
	realExit(1);
});

process.argv = ['node', 'parpar'].concat(passthrough);

eval(src);

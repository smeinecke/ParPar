#!/usr/bin/env python3
"""Differential arg-parser harness.

Runs each argv vector through `node test/dump_args.js --` (JS reference) and
the Rust parpar binary with PARPAR_DUMP_ARGS=1, then deep-compares the
emitted {argv, ppo, inputFiles} JSON and stderr/exit code.

Usage: test/diff_args.py [case-file] [-v]
Case file: `name | <argv...>` per line (# comments / blank lines skipped).
"""
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUST_BIN = ROOT / "rust" / "target" / "debug" / "parpar"
CASE_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ROOT / "test" / "diff_args_cases.txt"
VERBOSE = "-v" in sys.argv


def norm(v):
    if isinstance(v, dict):
        return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing in JS (rust={b[k]!r})")
            elif k not in b:
                out.append(f"{path}.{k}: missing in rust (js={a[k]!r})")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}: {a!r} vs {b!r}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def run(args, env_extra, cmd):
    env = {**__import__("os").environ, **env_extra}
    # empty stdin for both sides -> `-`/`fd://0` cases get deterministic EOF
    return subprocess.run(
        cmd + args, capture_output=True, text=True, env=env, cwd=ROOT, input=""
    )


def load(out):
    out = out.strip()
    if not out:
        return None
    try:
        return json.loads(out.splitlines()[-1])
    except Exception:
        return {"_raw": out}


clean = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)

USAGE_HINT = "Enter `parpar --help` or `parpar --help-full` for usage information"


def first_error(stderr):
    """The real binary exits at the first error() call. The dump harness
    stubs process.exit, so errors thrown inside `try{arg_parser}catch` get
    a second spurious `error(undefined)` (x.message on the exit-sentinel).
    Truncate JS stderr after the first usage-hint line."""
    s = clean(stderr)
    i = s.find(USAGE_HINT)
    if i >= 0:
        end = s.find("\n", i)
        return s[: end + 1 if end >= 0 else len(s)]
    return s


def compare(name, args):
    js = run(args, {}, ["node", "test/dump_args.js", "--"])
    rs = run(args, {"PARPAR_DUMP_ARGS": "1"}, [str(RUST_BIN)])
    j, r = load(js.stdout), load(rs.stdout)
    problems = []

    if j is None and r is None:
        if first_error(js.stderr) != clean(rs.stderr):
            problems.append(
                f"stderr differs:\nJS : {first_error(js.stderr)}\nRS : {clean(rs.stderr)}"
            )
        if js.returncode != rs.returncode:
            problems.append(f"exit codes {js.returncode} vs {rs.returncode}")
    elif j is None or r is None:
        problems.append(
            f"one side failed (js={js.returncode} rs={rs.returncode}):\n"
            f"JSerr: {js.stderr}\nRSerr: {rs.stderr}"
        )
    else:
        if isinstance(j, dict) and "error" in j:
            problems.append("JS late-error path not mirrored: " + str(j))
        else:
            problems += diff(norm(j), norm(r))
        if js.returncode != rs.returncode:
            problems.append(f"exit codes {js.returncode} vs {rs.returncode}")
    return problems


def main():
    cases = []
    for line in CASE_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, argv = line.partition("|")
        cases.append((name.strip(), shlex.split(argv)))

    npass = nfail = 0
    for name, args in cases:
        problems = compare(name, args)
        if problems:
            nfail += 1
            print(f"FAIL {name}: {args}")
            for p in problems:
                print("     " + p.replace("\n", "\n     "))
        else:
            npass += 1
            if VERBOSE:
                print(f"PASS {name}")
    print(f"{npass} passed, {nfail} failed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()

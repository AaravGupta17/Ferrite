#!/usr/bin/env python3
"""
tools/check_perf.py — performance regression gate (Section 1.3).

Compares `bench_model --json` output against a committed baseline
(temps/bench_baseline.json) and fails when a tracked metric regresses past
`--threshold` (default 0.20 = 20%; the deep-dive's starting point to dodge
noise).

Usage:
  check_perf.py --current current.json --baseline baseline.json [--threshold 0.20]

  --current   path to `bench_model --json` output (the JSON object may be
              embedded in a larger stdout stream; the part from the first
              line "{": to EOF is parsed).
  --baseline  committed baseline JSON (same schema as --current).
  --seed SAVE.json   write a baseline from --current and exit 0 (used once on
              the CI runner, then committed).

Exits non-zero if any metric is both tracked and regressed past the
threshold. Untracked keys in --current are ignored; tracked-but-missing keys
in --current are an error (a bench that silently stops reporting a metric
should fail the gate, not pass it).

This is a best-effort gate on noisy hardware. Keep the threshold conservative,
report the build mode (Debug/Release) in CI logs, and re-seed the baseline
when the hardware or the model changes — never tune the threshold to hide a
real regression.
"""
import argparse
import json
import sys


TRACKED = ("matmul_naive_ms", "matmul_avx2_ms", "matmul_speedup",
           "elem_naive_ms", "elem_avx2_ms", "elem_speedup",
           "gemm_base_ms", "gemm_transb_ms", "gemm_speedup")

# Larger-is-better metrics (speedups). All others are lower-is-better.
LOWER_IS_BETTER = set(TRACKED) - {"matmul_speedup", "elem_speedup",
                                  "gemm_speedup"}


def extract_json(text):
    start = text.find("\n{")
    if start == -1:
        start = text.find("{")
        if start == -1:
            return None
    return json.loads(text[start:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", help="bench_model --json output (stream)")
    ap.add_argument("--baseline", help="committed baseline JSON")
    ap.add_argument("--seed", help="write baseline JSON from --current")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    with open(args.current, "r", encoding="utf-8") as f:
        cur = extract_json(f.read())
    if cur is None:
        print("ERROR: no JSON object found in --current output", file=sys.stderr)
        return 1

    if args.seed:
        with open(args.seed, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"seeded baseline {args.seed} ({args.label or 'unlabeled'})")
        return 0

    with open(args.baseline, "r", encoding="utf-8") as f:
        base = json.load(f)

    failed = False
    print(f"perf gate {args.label or ''} vs {args.baseline} (threshold {args.threshold:.0%})")
    print(f"  {'metric':<14} {'baseline':>12} {'current':>12} {'delta':>9}")
    for key in TRACKED:
        if key in base and key not in cur:
            print(f"  {key:<14} MISSING in current output", file=sys.stderr)
            failed = True
            continue
        b = base[key]
        c = cur[key]
        if b == 0:
            continue
        delta = (c - b) / abs(b)
        flag = ""
        if key in LOWER_IS_BETTER and delta > args.threshold:
            flag = "  <-- REGRESSION"
            failed = True
        elif key not in LOWER_IS_BETTER and delta < -args.threshold:
            flag = "  <-- REGRESSION"
            failed = True
        print(f"  {key:<14} {b:>12.3f} {c:>12.3f} {delta:>+8.1%}{flag}")

    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
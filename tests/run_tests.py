#!/usr/bin/env python3

import os
import sys
import io
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xplus.compiler import compile_source, make_vm
from xplus.errors import XPlusError

EXAMPLES = os.path.join(ROOT, "examples")

TESTS = [
    ("hello.xp", ["Log: User Dimas has a speed of 999 ns",
                  "Hello, Dimas! Speed: 999"]),
    ("math_vec.xp", ["Sin(45 deg): 0.70710678", "Ln(e^3): 3.0"]),
    ("loops.xp", ["stopped at count = 5", "Sum 0..10 (exclusive) = 45",
                  "Sum 1..=10 (inclusive) = 55", "Array iteration (skip 30) = 70",
                  "Grade: B"]),
    ("struct_demo.xp", ["Player Dimas has speed 999.0",
                        "After buff: Dimas speed 1000.0", "Magnitude (3,4) = 5.0",
                        "Player level: 99"]),
    ("enum_match.xp", ["Server is active and blazing fast!", "Grade: B (solid)"]),
    ("pipeline.xp", ["Sin(30 deg) via pipeline = 0.5000", "|(3,4)| = 5.0",
                     "processed: ["]),
    ("comptime.xp", ["5! = 120", "sin(0 deg) = 0.0000", "sin(90 deg) = 1.0000",
                     "MAX_BUFFER = 4096"]),
    ("guard_result.xp", ["10 / 2  = 5.0", "10 / 4 = 2.5 (Ok)",
                         "1 / 0 -> Err(division by zero)",
                         "ratio = 75.0", "failed: division by zero",
                         "Found: Dimas", "ID 9: not found (without segfault!)"]),
    ("vec_simd.xp", ["v1 + v2 = [6.0, 8.0, 10.0, 12.0]", "v1 .x v2 (dot) = 70.0",
                     "|a| = 5.0"]),
    ("matmul.xp", ["W x X = [1.0, 2.1]", "A x B = [[19.0, 22.0], [43.0, 50.0]]"]),
    ("arena_demo.xp", ["mean   = 35.0", "arena peak"]),
    ("rational_demo.xp", ["1/3 + 2/7 = 13/21", "3/4 + 1/6 = 11/12",
                          "3/4 * 1/6 = 1/8", "3/4 / 1/6 = 9/2"]),
    ("smv_demo.xp", ["@core_count", "@compiler_ver", "@target_arch", "from function"]),
    ("iot_led.xp", ["[GPIO] pin 2 -> mode 1", "[GPIO] pin 2 -> HIGH",
                    "Blink complete: 5 cycles"]),
    ("interrupt_demo.xp", ["[MMIO] GPIOA_ODR @ 0x40020014", "GPIOA_ODR reread: 1",
                           "asm! executed"]),
    ("task_demo.xp", ["2 tasks registered", "async result:"]),
    ("benchmark_sim.xp", ["System energy after 100 steps:"]),
    ("library_user.xp", ["math::sin_deg(30) = 0.5000", "strng::length", "statistics::mean",
                         "@grad(x^2+3) at x=4.0 = 8.0"]),
    ("fizzbuzz.xp", ["FizzBuzz", "Fizz", "Buzz", "\n14\n"]),
]

ERROR_TESTS = [
    ("tensor_shape_err.xp", "E204"),
]


def run_one(fname, expect_substrings):
    path = os.path.join(EXAMPLES, fname)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    out = io.StringIO()
    err = io.StringIO()
    res = compile_source(src, fname, EXAMPLES)
    if not res.ok:
        return False, "COMPILE ERROR:\n" + "\n".join(e.render(color=False) for e in res.errors)
    vm = make_vm(res, out=out, err=err)
    try:
        code = vm.run_main()
    except XPlusError as e:
        return False, "RUNTIME PANIC:\n" + e.render(color=False)
    output = out.getvalue()
    missing = [s for s in expect_substrings if s not in output]
    if missing:
        return False, f"OUTPUT MISMATCH — missing: {missing}\n--- output ---\n{output}"
    return True, output


def run_error_one(fname, expect_code):
    path = os.path.join(EXAMPLES, fname)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    res = compile_source(src, fname, EXAMPLES)
    if res.ok:
        return False, "expected a COMPILE ERROR but it succeeded"
    codes = [e.code for e in res.errors]
    if expect_code not in codes:
        return False, f"error codes {codes} do not contain {expect_code}"
    return True, codes


def main():
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    passed = failed = 0
    print(f"{'=' * 64}\n X+ Test Suite — {len(TESTS)} output tests + {len(ERROR_TESTS)} error tests\n{'=' * 64}")

    for fname, expects in TESTS:
        if flt and flt not in fname:
            continue
        ok, info = run_one(fname, expects)
        if ok:
            passed += 1
            print(f"  \033[32m✓ PASS\033[0m {fname}")
        else:
            failed += 1
            print(f"  \033[31m✗ FAIL\033[0m {fname}")
            for line in info.splitlines():
                print(f"      {line}")

    for fname, code in ERROR_TESTS:
        if flt and flt not in fname:
            continue
        ok, info = run_error_one(fname, code)
        if ok:
            passed += 1
            print(f"  \033[32m✓ PASS\033[0m {fname} (correctly rejected: {info})")
        else:
            failed += 1
            print(f"  \033[31m✗ FAIL\033[0m {fname}: {info}")

    print(f"{'=' * 64}")
    total = passed + failed
    color = "\033[32m" if failed == 0 else "\033[31m"
    print(f" {color}{passed}/{total} tests passed\033[0m" +
          (f" — {failed} FAILED" if failed else " — ALL GREEN ✓"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
from __future__ import annotations

import math
import os
import platform
import time
import random as _random

def detect_cpu_features() -> dict:
    feats = {"arch": platform.machine() or "unknown",
             "avx512": False, "avx2": False, "fma3": False, "neon": False,
             "model": platform.processor() or "generic"}
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for block in f.read().split("\n\n"):
                    if "flags" in block and ":" in block:
                        flags = block.split("flags")[1].split(":")[1].split()
                        feats["avx512"] = "avx512f" in flags
                        feats["avx2"] = "avx2" in flags
                        feats["fma3"] = "fma3" in flags or "fma" in flags
                        feats["neon"] = "neon" in flags or "asimd" in flags
                        for line in block.splitlines():
                            if "model name" in line:
                                feats["model"] = line.split(":", 1)[1].strip()
                        break
                    if "Features" in block and ":" in block:
                        flags = block.split("Features")[1].split(":")[1].split()
                        feats["neon"] = "asimd" in flags or "neon" in flags
                        feats["fma3"] = "asimdhp" in flags or "fphp" in flags
                        for line in block.splitlines():
                            if "Model" in line or "model" in line:
                                feats["model"] = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    if platform.machine() in ("arm64", "aarch64"):
        feats["neon"] = True
    return feats


CPU = detect_cpu_features()


def simd_target_name() -> str:
    arch = CPU["arch"]
    if CPU["avx512"]:
        return "x86_64-AVX512"
    if CPU["avx2"]:
        return "x86_64-AVX2"
    if CPU["neon"]:
        return "aarch64-NEON"
    if arch in ("arm64", "aarch64"):
        return "aarch64-NEON"
    return "x86_64-SSE"


def simd_width_bits() -> int:
    if CPU["avx512"]:
        return 512
    if CPU["avx2"]:
        return 256
    if CPU["neon"]:
        return 128
    return 128


def simd_lane_count() -> int:
    return simd_width_bits() // 64

def _x_div(a, b):
    if b == 0:
        from .errors import XRuntimeError
        raise XRuntimeError("division by zero", hint="use 'guard b != 0.0 else { ... }'")
    return a / b


def _x_mod(a, b):
    if b == 0:
        from .errors import XRuntimeError
        raise XRuntimeError("modulo by zero")
    return a % b


def _x_rational(num, den):
    from fractions import Fraction
    return Fraction(int(num), int(den))


def _x_push(arr, val):
    arr.append(val)
    return arr


def to_string(v) -> str:
    from .vm import x_repr
    return x_repr(v)


NATIVE_TABLE: dict = {
    "sin":        {"fn": math.sin,  "ret": "f64", "desc": "sinus (radian)"},
    "cos":        {"fn": math.cos,  "ret": "f64", "desc": "cosinus (radian)"},
    "tan":        {"fn": math.tan,  "ret": "f64", "desc": "tangent (radian)"},
    "asin":       {"fn": math.asin, "ret": "f64"},
    "acos":       {"fn": math.acos, "ret": "f64"},
    "atan":       {"fn": math.atan, "ret": "f64"},
    "atan2":      {"fn": math.atan2, "ret": "f64"},
    "sqrt":       {"fn": math.sqrt, "ret": "f64"},
    "ln":         {"fn": math.log,  "ret": "f64"},
    "log":        {"fn": math.log10, "ret": "f64"},
    "log2":       {"fn": math.log2, "ret": "f64"},
    "exp":        {"fn": math.exp,  "ret": "f64"},
    "abs":        {"fn": abs,       "ret": "f64"},
    "floor":      {"fn": math.floor, "ret": "int"},
    "ceil":       {"fn": math.ceil, "ret": "int"},
    "round":      {"fn": round,     "ret": "int"},
    "min":        {"fn": min,       "ret": "f64"},
    "max":        {"fn": max,       "ret": "f64"},
    "pow":        {"fn": pow,       "ret": "f64"},
    "sum":        {"fn": lambda xs: sum(xs), "ret": "f64"},
    "length":     {"fn": lambda xs: len(xs), "ret": "int"},
    "len":        {"fn": lambda xs: len(xs), "ret": "int"},
    "push":       {"fn": _x_push, "ret": "int"},
    "__rational": {"fn": _x_rational, "ret": "rational"},
    "append":     {"fn": _x_push, "ret": "int"},
    "concat":     {"fn": lambda a, b: a + b, "ret": "array"},
    "to_string":  {"fn": to_string, "ret": "string"},
    "to_upper":   {"fn": lambda s: str(s).upper(), "ret": "string"},
    "to_lower":   {"fn": lambda s: str(s).lower(), "ret": "string"},
    "trim":       {"fn": lambda s: str(s).strip(), "ret": "string"},
    "contains":   {"fn": lambda s, sub: sub in s, "ret": "bool"},
    "replace":    {"fn": lambda s, a, b: str(s).replace(a, b), "ret": "string"},
    "char_at":    {"fn": lambda s, i: s[i], "ret": "string"},
    "time_ns":    {"fn": lambda: time.time_ns(), "ret": "int"},
    "now_ms":     {"fn": lambda: time.time_ns() // 1_000_000, "ret": "int"},
    "random_entropy": {"fn": lambda: int.from_bytes(os.urandom(8), "little"), "ret": "int"},
    "random_f64": {"fn": _random.random, "ret": "f64"},
}

for _n in ("sin", "cos", "tan", "sqrt", "ln", "exp", "log"):
    NATIVE_TABLE[f"__builtin_fast_{_n}"] = NATIVE_TABLE[_n]

NATIVE_TABLE["__builtin_str"] = NATIVE_TABLE["to_string"]
NATIVE_TABLE["__builtin_push"] = NATIVE_TABLE["push"]
NATIVE_TABLE["__gpio_mode"] = {"fn": lambda p, m: None, "ret": "void"}
NATIVE_TABLE["__gpio_write"] = {"fn": lambda p, v: None, "ret": "void"}
NATIVE_TABLE["__gpio_read"] = {"fn": lambda p: 0, "ret": "int"}
NATIVE_TABLE["__gpio_toggle"] = {"fn": lambda p: 0, "ret": "int"}
NATIVE_TABLE["__delay_ms"] = {"fn": lambda ms: None, "ret": "void"}
NATIVE_TABLE["__task_spawn"] = {"fn": lambda f, s, c: None, "ret": "void"}
NATIVES = NATIVE_TABLE

_STARTUP_NS = time.time_ns()

SMV_TABLE: dict = {
    "pi":         ("f64", "3.141592653589793 — konstanta \u03c0"),
    "euler":      ("f64", "2.718281828459045 — konstanta e"),
    "tau":        ("f64", "6.283185307179586 — 2\u03c0"),
    "phi":        ("f64", "1.618033988749895 — golden ratio"),
    "simd_width": ("int", "lebar register SIMD (bit)"),
    "simd_lane_count": ("int", "jumlah lane SIMD paralel"),
    "is_simd_active": ("bool", "status eksekusi SIMD"),
    "core_count": ("int", "jumlah core CPU"),
    "cpu_freq_mhz": ("int", "clock CPU (MHz, estimasi)"),
    "thread_id":  ("int", "ID thread aktif"),
    "time_ns":    ("int", "timestamp nanodetik"),
    "uptime_ms":  ("int", "uptime runtime X+ (ms)"),
    "random_entropy": ("int", "TRNG hardware (os.urandom)"),
    "arena_free": ("int", "sisa budget arena aktif (bytes, simulasi)"),
    "arena_total": ("int", "total alokasi arena aktif"),
    "arena_peak": ("int", "puncak pemakaian arena"),
    "stack_usage": ("int", "kedalaman stack frame saat ini"),
    "heap_allocated": ("int", "alokasi persistent"),
    "is_unsafe":  ("bool", "sedang di dalam blok unsafe"),
    "compiler_ver": ("string", "versi compiler xpc"),
    "target_arch": ("string", "arsitektur target"),
    "target_os":  ("string", "OS target"),
    "build_profile": ("string", "debug | release"),
    "opt_level":  ("int", "level optimasi (0-3)"),
    "tensor_backend": ("string", "backend akselerator aktif"),
    "matrix_precision": ("string", "mode presisi tensor"),
    "ctfe_active": ("bool", "sedang dievaluasi compiler (CTFE)"),
    "git_commit": ("string", "hash commit build"),
    "build_timestamp": ("int", "unix time kompilasi"),
    "file":       ("string", "nama file .xp"),
    "line":       ("int", "current source line"),
    "col":        ("int", "current source column"),
    "function":   ("string", "nama fungsi aktif"),
    "call_depth": ("int", "kedalaman call stack"),
    "app_name":   ("string", "nama proyek (xpc.toml)"),
    "environment": ("string", "development | production"),
    "timezone":   ("string", "zona waktu sistem"),
    "locale":     ("string", "locale sistem"),
    "log_level":  ("string", "INFO"),
    "terminal_width": ("int", "lebar terminal"),
    "debug_mode": ("bool", "mode debug"),
    "batch_size": ("int", "ukuran batch tensor aktif"),
    "epoch_index": ("int", "indeks epoch training"),
    "learning_rate": ("f64", "hyperparameter LR"),
    "loss_value": ("f64", "loss terakhir"),
}

SMV_COMPUTE = {
    "pi": lambda vm, fr: math.pi,
    "euler": lambda vm, fr: math.e,
    "tau": lambda vm, fr: math.tau,
    "phi": lambda vm, fr: (1 + 5 ** 0.5) / 2,
    "simd_width": lambda vm, fr: simd_width_bits(),
    "simd_lane_count": lambda vm, fr: simd_lane_count(),
    "is_simd_active": lambda vm, fr: True,
    "core_count": lambda vm, fr: os.cpu_count() or 1,
    "cpu_freq_mhz": lambda vm, fr: 3500,
    "thread_id": lambda vm, fr: 0,
    "time_ns": lambda vm, fr: time.time_ns(),
    "uptime_ms": lambda vm, fr: (time.time_ns() - _STARTUP_NS) // 1_000_000,
    "random_entropy": lambda vm, fr: int.from_bytes(os.urandom(8), "little"),
    "arena_free": lambda vm, fr: vm.arena_budget,
    "arena_total": lambda vm, fr: vm.arena_total,
    "arena_peak": lambda vm, fr: vm.arena_peak,
    "stack_usage": lambda vm, fr: len(vm.frames) * 256,
    "heap_allocated": lambda vm, fr: vm.arena_total,
    "is_unsafe": lambda vm, fr: vm.unsafe_depth > 0,
    "compiler_ver": lambda vm, fr: vm.compiler_ver,
    "target_arch": lambda vm, fr: vm.target_arch,
    "target_os": lambda vm, fr: vm.target_os,
    "build_profile": lambda vm, fr: vm.build_profile,
    "opt_level": lambda vm, fr: vm.opt_level,
    "tensor_backend": lambda vm, fr: vm.tensor_backend,
    "matrix_precision": lambda vm, fr: "mixed",
    "ctfe_active": lambda vm, fr: vm.ctfe_active,
    "git_commit": lambda vm, fr: vm.git_commit,
    "build_timestamp": lambda vm, fr: vm.build_timestamp,
    "file": lambda vm, fr: vm.file,
    "line": lambda vm, fr: fr.current_line,
    "col": lambda vm, fr: 0,
    "function": lambda vm, fr: (fr.func.get("name", "main") if isinstance(fr.func, dict) else "main"),
    "call_depth": lambda vm, fr: len(vm.frames),
    "app_name": lambda vm, fr: vm.app_name,
    "environment": lambda vm, fr: "development",
    "timezone": lambda vm, fr: "Asia/Jakarta",
    "locale": lambda vm, fr: "id_ID",
    "log_level": lambda vm, fr: "INFO",
    "terminal_width": lambda vm, fr: 80,
    "debug_mode": lambda vm, fr: False,
    "batch_size": lambda vm, fr: vm.batch_size,
    "epoch_index": lambda vm, fr: vm.epoch_index,
    "learning_rate": lambda vm, fr: 0.001,
    "loss_value": lambda vm, fr: 0.0,
}
from __future__ import annotations

import sys
import time
import math

from .builtins import NATIVES, SMV_COMPUTE, simd_target_name
from .errors import XRuntimeError
from . import VERSION

INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1
WRAP_MASK = (1 << 64) - 1


class Frame:
    __slots__ = ("func", "regs", "ip", "call_dst", "current_line")

    def __init__(self, func, call_dst=None):
        self.func = func
        self.regs = {}
        self.ip = 0
        self.call_dst = call_dst
        self.current_line = 0


class XStruct:
    __slots__ = ("name", "fields")

    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

    def __eq__(self, other):
        return isinstance(other, XStruct) and self.name == other.name and self.fields == other.fields

    def __repr__(self):
        inner = ", ".join(f"{k}: {x_repr(v)}" for k, v in self.fields.items())
        return f"{self.name}{{{inner}}}"


class XEnumVal:
    __slots__ = ("enum", "variant")

    def __init__(self, enum, variant):
        self.enum = enum
        self.variant = variant

    def __eq__(self, other):
        return isinstance(other, XEnumVal) and self.enum == other.enum and self.variant == other.variant

    def __repr__(self):
        return f"{self.enum}::{self.variant}"


class XResult:
    __slots__ = ("tag", "value")

    def __init__(self, tag, value):
        self.tag = tag          # "ok" | "err"
        self.value = value

    def __repr__(self):
        return f"Ok({x_repr(self.value)})" if self.tag == "ok" else f"Err({x_repr(self.value)})"


class XOptional:
    __slots__ = ("some", "value")

    def __init__(self, some, value=None):
        self.some = some
        self.value = value

    def __repr__(self):
        return f"Some({x_repr(self.value)})" if self.some else "None"


class XTask:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Task({x_repr(self.value)})"


class XGrad:
    __slots__ = ("fn_name",)

    def __init__(self, fn_name):
        self.fn_name = fn_name

    def __repr__(self):
        return f"@grad({self.fn_name})"

def x_repr(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15 and not math.isinf(v):
            return f"{v:.1f}"
        return repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "[" + ", ".join(x_repr(i) for i in v) + "]"
    if isinstance(v, XStruct):
        return repr(v)
    if isinstance(v, XEnumVal):
        return repr(v)
    if isinstance(v, XResult):
        return repr(v)
    if isinstance(v, XOptional):
        return repr(v)
    if isinstance(v, XTask):
        return repr(v)
    if isinstance(v, XGrad):
        return repr(v)
    if v is None:
        return "void"
    return str(v)


def format_value(v, spec=None) -> str:
    if spec:
        s = spec.lstrip(":")
        try:
            if "." in s or s.startswith(("<", ">", "=", "^")):
                return format(float(v), s + ("f" if "." in s and not any(c.isalpha() for c in s) else ""))
            return format(v, s)
        except (ValueError, TypeError):
            return x_repr(v)
    return x_repr(v)


def apply_fmt(fmt: str, args, specs) -> str:
    out = []
    i = 0
    n = 0
    while i < len(fmt):
        ch = fmt[i]
        if ch == "{" and i + 1 < len(fmt) and fmt[i + 1] != "{":
            j = fmt.index("}", i)
            inner = fmt[i + 1:j]
            if ";" in inner:
                idx_s, spec = inner.split(";", 1)
                spec = ":" + spec
            else:
                idx_s, spec = inner, (specs[n] if n < len(specs) else None)
            idx = int(idx_s)
            if idx >= len(args):
                raise XRuntimeError(f"format slot {idx} out of range ({len(args)} args injected)")
            out.append(format_value(args[idx], spec))
            n += 1
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)

class XVM:
    def __init__(self, funcs, consts_meta, meta=None, out=None, err=None, board=False):
        self.funcs = funcs or {}
        self.consts_meta = consts_meta or {}
        self.meta = meta or {}
        self.globals = {}
        self.frames: list[Frame] = []
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self.steps = 0
        self.max_steps = None
        self.ctfe_active = False
        self.unsafe_depth = 0
        self.arena_budget = 65536
        self.arena_total = 0
        self.arena_peak = 0
        self.mmio = {}
        self.gpio_pins = {}
        self.gpio_log = []
        self.board_mode = board
        self.task_registry = []
        self.compiler_ver = f"xpc {VERSION}"
        self.target_arch = simd_target_name()
        self.target_os = self.meta.get("target_os", "linux")
        self.build_profile = self.meta.get("profile", "release")
        self.opt_level = self.meta.get("opt_level", 3)
        self.tensor_backend = "cpu-simd"
        self.git_commit = self.meta.get("git_commit", "v0.1-local")
        self.build_timestamp = self.meta.get("build_timestamp", int(time.time()))
        self.file = self.meta.get("file", "main.xp")
        self.app_name = self.meta.get("app_name", "x-plus-app")
        self.batch_size = 0
        self.epoch_index = 0
        self._start_ns = time.time_ns()

    def panic(self, code, msg, hint=None):
        raise XRuntimeError(msg, file=self.meta.get("file", "main.xp"),
                            line=self.frames[-1].current_line if self.frames else None,
                            hint=hint, code=code)

    def load_consts(self):
        for name, info in self.consts_meta.items():
            if name in self.globals:
                continue
            if info.get("literal") is not None:
                self.globals[name] = _deep_literal(info["literal"])
            elif info.get("init_fn"):
                self.globals[name] = self.call_fn(info["init_fn"], [], ctfe=True)

    def call_fn(self, name, args, ctfe=False, max_steps=None):
        if name not in self.funcs:
            raise XRuntimeError(f"E202 undefined function '{name}'")
        prev_ctfe = self.ctfe_active
        prev_max = self.max_steps
        self.ctfe_active = ctfe
        self.max_steps = max_steps
        params = self.funcs[name].get("params", [])
        frame = Frame(self.funcs[name])
        for p, a in zip(params, args):
            frame.regs[p] = a
        self.frames.append(frame)
        try:
            self._run_frame(frame)
        finally:
            self.ctfe_active = prev_ctfe
            self.max_steps = prev_max
        return frame.regs.get("$ret")

    def run_main(self, entry="main"):
        if entry not in self.funcs:
            for name, f in self.funcs.items():
                if f.get("is_entry"):
                    entry = name
                    break
            else:
                raise XRuntimeError("no entry point: write 'main => int32 { ... }'",
                                    hint="the X+ default entry is the 'main' function")
        code = 0
        ret = self.call_fn(entry, [])
        if isinstance(ret, bool):
            code = int(ret)
        elif isinstance(ret, int):
            code = ret
        return code

    def _run_frame(self, fr: Frame):
        instrs = fr.func.get("instrs", [])
        n = len(instrs)
        while fr.ip < n:
            if self.max_steps is not None and self.steps >= self.max_steps:
                if self.ctfe_active:
                    self.panic("E220",
                               "CTFE step limit exceeded",
                               hint="the comptime function is likely an infinite loop")
                else:
                    raise XRuntimeError("execution step limit exceeded")
            ins = instrs[fr.ip]
            fr.ip += 1
            self.steps += 1
            op = ins[0]

            if op == "line":
                fr.current_line = ins[1]
            elif op == "const":
                fr.regs[ins[1]] = ins[2]
            elif op == "mov":
                fr.regs[ins[1]] = fr.regs.get(ins[2])
            elif op == "bin":
                fr.regs[ins[1]] = self._binop(ins[2], fr.regs.get(ins[3]), fr.regs.get(ins[4]), fr)
            elif op == "un":
                v = fr.regs.get(ins[3])
                fr.regs[ins[1]] = (not v) if ins[2] == "!" else -v
            elif op == "label":
                pass
            elif op == "jmp":
                fr.ip = self._label_ip(instrs, ins[1], n)
            elif op == "jz":
                if not fr.regs.get(ins[1]):
                    fr.ip = self._label_ip(instrs, ins[2], n)
            elif op == "jnz":
                if fr.regs.get(ins[1]):
                    fr.ip = self._label_ip(instrs, ins[2], n)
            elif op == "ret":
                fr.regs["$ret"] = fr.regs.get(ins[1]) if len(ins) > 1 else None
                if len(self.frames) > 1:
                    self.frames.pop()
                    dst = fr.call_dst
                    if dst:
                        self.frames[-1].regs[dst] = fr.regs.get("$ret")
                    return
                return
            elif op == "call":
                dst = ins[1]
                fname = ins[2]
                args = [fr.regs.get(a) for a in ins[3]]
                named = {k: fr.regs.get(v) for k, v in ins[4].items()} if len(ins) > 4 else {}
                fr.regs[dst] = self._invoke(fname, args, named, dst)
            elif op == "callv":
                v = fr.regs.get(ins[2])
                args = [fr.regs.get(a) for a in ins[3]]
                fr.regs[ins[1]] = self._call_value(v, args, ins[1])
            elif op == "native":
                args = [fr.regs.get(a) for a in ins[3]]
                fr.regs[ins[1]] = self._native(ins[2], args, fr)
            elif op == "show":
                chan = fr.regs.get(ins[1], ins[1])
                text = apply_fmt(ins[2], [fr.regs.get(a) for a in ins[3]], ins[4] if len(ins) > 4 else [])
                self._emit(chan, text)
            elif op == "gload":
                if ins[2] not in self.globals:
                    self.panic("E201", f"undefined const '{ins[2]}'")
                fr.regs[ins[1]] = self.globals[ins[2]]
            elif op == "gstore":
                self.globals[ins[1]] = fr.regs.get(ins[2])
            elif op == "load":
                self.panic("E201", f"undefined variable '{ins[2]}'")
            elif op == "smv":
                fr.regs[ins[1]] = self._smv(ins[2], fr)
            elif op == "index_get":
                obj = fr.regs.get(ins[2])
                idx = fr.regs.get(ins[3])
                fr.regs[ins[1]] = self._index_get(obj, idx, fr)
            elif op == "index_set":
                obj = fr.regs.get(ins[1])
                idx = fr.regs.get(ins[2])
                self._index_set(obj, idx, fr.regs.get(ins[3]), fr)
            elif op == "member":
                obj = fr.regs.get(ins[2])
                if isinstance(obj, XStruct):
                    if ins[3] not in obj.fields:
                        self.panic("E210", f"struct '{obj.name}' has no field '{ins[3]}'")
                    fr.regs[ins[1]] = obj.fields[ins[3]]
                elif isinstance(obj, dict):
                    fr.regs[ins[1]] = obj.get(ins[3])
                else:
                    self.panic("E210", f"cannot access field '{ins[3]}' on {x_repr(obj)[:30]}")
            elif op == "set_member":
                obj = fr.regs.get(ins[1])
                if isinstance(obj, XStruct):
                    obj.fields[ins[2]] = fr.regs.get(ins[3])
                else:
                    self.panic("E210", "cannot set field on non-struct")
            elif op == "array":
                fr.regs[ins[1]] = [fr.regs.get(a) for a in ins[2]]
                self._arena_alloc(1)
            elif op == "fill_array":
                v = fr.regs.get(ins[2])
                cnt = int(fr.regs.get(ins[3]) or 0)
                fr.regs[ins[1]] = [v] * cnt
                self._arena_alloc(cnt)
            elif op == "struct":
                fr.regs[ins[1]] = XStruct(ins[2], {k: fr.regs.get(s) for k, s in ins[3]})
            elif op == "enum":
                fr.regs[ins[1]] = XEnumVal(ins[2], ins[3])
            elif op == "wrap_ok":
                fr.regs[ins[1]] = XResult("ok", fr.regs.get(ins[2]))
            elif op == "wrap_err":
                fr.regs[ins[1]] = XResult("err", fr.regs.get(ins[2]))
            elif op == "wrap_some":
                fr.regs[ins[1]] = XOptional(True, fr.regs.get(ins[2]))
            elif op == "wrap_none":
                fr.regs[ins[1]] = XOptional(False)
            elif op == "propagate":
                v = fr.regs.get(ins[1])
                if isinstance(v, XResult) and v.tag == "err":
                    fr.regs["$ret"] = v
                    if len(self.frames) > 1:
                        self.frames.pop()
                        if fr.call_dst:
                            self.frames[-1].regs[fr.call_dst] = v
                    return
                if isinstance(v, XOptional) and not v.some:
                    fr.regs["$ret"] = v
                    if len(self.frames) > 1:
                        self.frames.pop()
                        if fr.call_dst:
                            self.frames[-1].regs[fr.call_dst] = v
                    return
            elif op == "extract":
                v = fr.regs.get(ins[2])
                fr.regs[ins[1]] = self._extract(v, fr)
            elif op == "cast":
                fr.regs[ins[1]] = self._cast(fr.regs.get(ins[2]), ins[3], fr)
            elif op == "range":
                fr.regs[ins[1]] = ("__range__", fr.regs.get(ins[2]), fr.regs.get(ins[3]), ins[4])
            elif op == "arena_enter":
                self.arena_budget = 65536
                self._arena_mark = self.arena_total
            elif op == "arena_exit":
                freed = self.arena_total - getattr(self, "_arena_mark", 0)
                self.arena_total = getattr(self, "_arena_mark", 0)
            elif op == "unsafe_enter":
                self.unsafe_depth += 1
            elif op == "unsafe_exit":
                self.unsafe_depth = max(0, self.unsafe_depth - 1)
            elif op == "spawn":
                named = ins[2] if len(ins) > 2 else {}
                self.task_registry.append((ins[1], named))
                core = named.get("core", "?")
                stack = named.get("stack_size", 2048)
                print(f"[task] spawned @{ins[1]} (core: {core}, stack: {stack}B) [v0.1: registered]", file=self.out)
            elif op == "wrap_task":
                pass
            elif op == "unwrap_task":
                v = fr.regs.get(ins[2])
                fr.regs[ins[1]] = v.value if isinstance(v, XTask) else v
            elif op == "asm":
                if self.board_mode or self.unsafe_depth > 0:
                    for s in ins[1]:
                        print(f"[asm] {s}", file=self.out)
            elif op == "reg_write":
                val = fr.regs.get(ins[2])
                self.mmio[ins[1]] = val
                print(f"[MMIO] {ins[1]} @ {self._reg_addr(ins[1])} <= {self._fmt_reg(val)}", file=self.out)
            elif op == "reg_read":
                fr.regs[ins[1]] = self.mmio.get(ins[2], 0)
            elif op == "verify":
                if not fr.regs.get(ins[1]):
                    self.panic("E221", "@verify formal check FAILED: post-condition violated",
                               hint="fix the function logic or relax the @verify contract")
            elif op == "tensor_note":
                print(f"[{ins[1]}] offloaded to backend '{self.tensor_backend}' (v0.1 simulated)", file=self.out)
            elif op == "panic":
                self.panic(ins[1], ins[2])
            elif op == "nop":
                pass
            else:
                self.panic("E9xx", f"unknown XIR opcode '{op}'")

    def _label_ip(self, instrs, label, n):
        for k in range(n):
            if instrs[k][0] == "label" and instrs[k][1] == label:
                return k
        self.panic("E9xx", f"missing label '{label}' (compiler bug)")

    def _invoke(self, fname, args, named, dst):
        if len(self.frames) > 8000:
            self.panic("E3xx", f"stack overflow: call depth exceeded 8000 frames in '{fname}'",
                       hint="check the recursion base case")
        base = fname.split("::")[-1]
        if fname in self.funcs:
            f = self.funcs[fname]
            frame = Frame(f, call_dst=dst)
            params = f.get("params", [])
            for p, a in zip(params, args):
                frame.regs[p] = a
            for k, v in named.items():
                pk = f"%v_{k}"
                if pk in params:
                    frame.regs[pk] = v
            self.frames.append(frame)
            self._run_frame(frame)
            return frame.regs.get("$ret")
        if base in NATIVES:
            return self._native(base, args + list(named.values()))
        if args and isinstance(args[0], dict):
            ref = args[0]
            if "__fnref__" in ref:
                if fname in ("task::spawn", "spawn"):
                    return None
                return self._invoke(ref["__fnref__"], args[1:], {}, dst)
            if "__grad__" in ref:
                return XGrad(ref["__grad__"])
        self.panic("E202", f"undefined function '{fname}'",
                   hint=f"try 'xpc search {base}' in the central registry")

    def _call_value(self, v, args, dst):
        if isinstance(v, dict) and "__fnref__" in v:
            return self._invoke(v["__fnref__"], args, {}, dst)
        if isinstance(v, XGrad) or (isinstance(v, dict) and "__grad__" in v):
            fn_name = v.fn_name if isinstance(v, XGrad) else v["__grad__"]
            x = args[0] if args else 0.0
            h = 1e-6
            f_plus = self._invoke(fn_name, [x + h], {}, dst)
            f_min = self._invoke(fn_name, [x - h], {}, dst)
            return (f_plus - f_min) / (2 * h)
        self.panic("E202", f"{x_repr(v)[:30]} is not callable")

    def _native(self, name, args, fr=None):
        if name in ("__gpio_mode", "__gpio_write", "__gpio_read", "__gpio_toggle"):
            return self._gpio(name, args)
        if name == "__delay_ms":
            ms = args[0] if args else 0
            time.sleep(min(ms, 200) / 1000.0)
            return None
        if name == "delay_ms":
            ms = args[0] if args else 0
            time.sleep(min(ms, 200) / 1000.0)
            return None
        if name == "spawn":
            self.task_registry.append((x_repr(args[0]) if args else "?", {}))
            print(f"[task] spawned [v0.1: registered]", file=self.out)
            return None
        if name == "__task_spawn":
            fnref = args[0] if args else None
            stack = args[1] if len(args) > 1 else 2048
            core = args[2] if len(args) > 2 else 0
            fname = x_repr(fnref) if fnref else "?"
            self.task_registry.append((fname, {"stack_size": stack, "core": core}))
            print(f"[task] spawned @{fname} (core: {core}, stack: {stack}B) [v0.1: registered]", file=self.out)
            return None
        if name in NATIVES:
            try:
                return NATIVES[name]["fn"](*args)
            except ZeroDivisionError:
                self.panic("E302", "division by zero in native call",
                           hint="use 'guard b != 0.0 else { ... }'")
            except ValueError as e:
                self.panic("E3xx", f"math domain error: {e}")
        self.panic("E202", f"undefined native function '{name}'")

    def _gpio(self, name, args):
        if name == "__gpio_mode":
            pin, mode = args
            self.gpio_pins[pin] = {"mode": mode, "val": 0}
            self.gpio_log.append(("mode", pin, mode))
            print(f"[GPIO] pin {pin} -> mode {mode}", file=self.out)
            return None
        if name == "__gpio_write":
            pin, val = args
            self.gpio_pins.setdefault(pin, {"mode": "OUTPUT", "val": 0})["val"] = val
            self.gpio_log.append(("write", pin, val))
            print(f"[GPIO] pin {pin} -> {'HIGH' if val else 'LOW'}", file=self.out)
            return None
        if name == "__gpio_read":
            return self.gpio_pins.get(args[0], {}).get("val", 0)
        if name == "__gpio_toggle":
            pin = args[0]
            cur = self.gpio_pins.setdefault(pin, {"mode": "OUTPUT", "val": 0})
            cur["val"] = 0 if cur["val"] else 1
            self.gpio_log.append(("toggle", pin, cur["val"]))
            print(f"[GPIO] pin {pin} -> {'HIGH' if cur['val'] else 'LOW'}", file=self.out)
            return cur["val"]
        return None

    def _binop(self, op, a, b, fr):
        if op == "matmul":
            return self._matmul(a, b, fr)
        if op == "dot":
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    self.panic("E204", f"dot product length mismatch: {len(a)} vs {len(b)}")
                return sum(x * y for x, y in zip(a, b))
            self.panic("E204", "'.x' requires two vectors")
        if isinstance(a, list) and isinstance(b, list) and op in ("+", "-", "*", "/"):
            if len(a) != len(b):
                self.panic("E204", f"vector length mismatch: {len(a)} vs {len(b)}")
            if op == "+": return [x + y for x, y in zip(a, b)]
            if op == "-": return [x - y for x, y in zip(a, b)]
            if op == "*": return [x * y for x, y in zip(a, b)]
            return [x / y for x, y in zip(a, b)]
        if isinstance(a, list) and isinstance(b, (int, float)) and op in ("+", "-", "*", "/", "^"):
            if op == "+": return [x + b for x in a]
            if op == "-": return [x - b for x in a]
            if op == "*": return [x * b for x in a]
            if op == "/": return [x / b for x in a]
            return [x ** b for x in a]
        if op == "+":
            if a is None: return b
            if isinstance(a, str) or isinstance(b, str):
                if isinstance(a, str) and isinstance(b, str):
                    return a + b
                self.panic("E206", f"cannot add string with {type(b).__name__}",
                           hint="use show(\"...{var}...\") for interpolation")
            r = a + b
            if isinstance(r, int) and (r > INT64_MAX or r < INT64_MIN):
                self.panic("E301", "integer overflow on '+'",
                           hint="use '+%' for explicit wrap-around")
            return r
        if op == "+%":
            return ((a + b) & WRAP_MASK)
        if op == "-%":
            return ((a - b) & WRAP_MASK)
        if op == "*%":
            return ((a * b) & WRAP_MASK)
        if op == "-":
            r = a - b
            if isinstance(r, int) and (r > INT64_MAX or r < INT64_MIN):
                self.panic("E301", "integer overflow on '-'", hint="use '-%'")
            return r
        if op == "*":
            r = a * b
            if isinstance(r, int) and not isinstance(a, bool) and not isinstance(b, bool) \
                    and (r > INT64_MAX or r < INT64_MIN):
                self.panic("E301", "integer overflow on '*'", hint="use '*%'")
            return r
        if op == "/":
            if b == 0:
                self.panic("E302", "division by zero",
                           hint="use 'guard b != 0.0 else { ... }'")
            return a / b
        if op == "%":
            if b == 0:
                self.panic("E302", "modulo by zero")
            return a % b
        if op == "^":
            return a ** b
        if op == "==":
            return self._eq(a, b)
        if op == "==tag":
            if isinstance(a, XResult):
                return b in ("Ok", "Err") and a.tag == ("ok" if b == "Ok" else "err")
            if isinstance(a, XOptional):
                return b in ("Some", "None") and a.some == (b == "Some")
            return False
        if op == "!=":
            return not self._eq(a, b)
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        if op == ">=":
            return a >= b
        self.panic("E9xx", f"unknown binop '{op}'")

    def _eq(self, a, b):
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(self._eq(x, y) for x, y in zip(a, b))
        return a == b

    def _matmul(self, a, b, fr):
        if isinstance(a, list) and isinstance(a[0] if a else None, list) \
                and isinstance(b, list) and not (b and isinstance(b[0], list)):
            rows = len(a)
            cols = len(a[0]) if a else 0
            if cols != len(b):
                self.panic("E204",
                           f"Dimension Mismatch in Matrix Multiplication: [{rows} x {cols}] x [1 x {len(b)}]",
                           hint=f"matrix columns ({cols}) must equal the vector length ({len(b)})")
            return [sum(r[j] * b[j] for j in range(cols)) for r in a]
        if isinstance(a, list) and isinstance(a[0] if a else None, list) \
                and isinstance(b, list) and isinstance(b[0] if b else None, list):
            rows_a, cols_a = len(a), len(a[0]) if a else 0
            rows_b, cols_b = len(b), len(b[0]) if b else 0
            if cols_a != rows_b:
                self.panic("E204",
                           f"Dimension Mismatch in Matrix Multiplication: [{rows_a} x {cols_a}] x [{rows_b} x {cols_b}]",
                           hint=f"columns of A ({cols_a}) must equal rows of B ({rows_b})")
            return [[sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
                    for i in range(rows_a)]
        if isinstance(a, list) and isinstance(b, list) and isinstance(b[0] if b else None, list):
            return [
                sum(a[k] * b[k][j] for k in range(len(a))) for j in range(len(b[0]))]
        self.panic("E204", "matmul 'x' requires mat/vec operands")

    def _index_get(self, obj, idx, fr):
        if isinstance(obj, (list, str)):
            i = int(idx)
            if i < 0 or i >= len(obj):
                self.panic("E303", f"index out of bounds: {i} (length {len(obj)})",
                           hint="bounds-check is permanently on in X+ — no segfaults")
            return obj[i]
        self.panic("E303", f"cannot index {x_repr(obj)[:30]}")

    def _index_set(self, obj, idx, val, fr):
        if isinstance(obj, list):
            i = int(idx)
            if i < 0 or i >= len(obj):
                self.panic("E303", f"index out of bounds: {i} (length {len(obj)})")
            obj[i] = val
            return
        self.panic("E303", "cannot index-assign non-array")

    def _extract(self, v, fr):
        if isinstance(v, XResult):
            return v.value
        if isinstance(v, XOptional):
            return v.value
        return v

    def _cast(self, v, tname, fr):
        base = tname.lstrip("[").split(";")[0].strip()
        try:
            if base in ("i8", "int8"):
                return self._checked_int(int(v), -128, 127, tname)
            if base in ("i16", "int16"):
                return self._checked_int(int(v), -32768, 32767, tname)
            if base in ("i32", "int32"):
                return self._checked_int(int(v), -2147483648, 2147483647, tname)
            if base in ("i64", "int64", "int"):
                return self._checked_int(int(v), INT64_MIN, INT64_MAX, tname)
            if base == "u8":
                return self._checked_int(int(v), 0, 255, tname)
            if base == "u16":
                return self._checked_int(int(v), 0, 65535, tname)
            if base == "u32":
                return self._checked_int(int(v), 0, 4294967295, tname)
            if base in ("u64", "usize"):
                return self._checked_int(int(v), 0, 18446744073709551615, tname)
            if base in ("f32", "f64"):
                return float(v)
            if base == "string":
                return x_repr(v)
            if base == "bool":
                return bool(v)
            if base == "rational":
                from fractions import Fraction
                return Fraction(str(v))
        except (ValueError, OverflowError):
            self.panic("E306", f"cast failed: {x_repr(v)[:30]} as {tname}")
        return v

    def _checked_int(self, val, lo, hi, tname):
        if val < lo or val > hi:
            self.panic("E306", f"cast overflow: {val} does not fit in {tname} [{lo}..{hi}]",
                       hint="checked-by-default; use a wider type or a wrapping op")
        return val

    def _smv(self, name, fr):
        fn = SMV_COMPUTE.get(name)
        if fn is None:
            self.panic("E120", f"unknown System Magic Variable '@{name}'")
        return fn(self, fr)

    def _emit(self, chan, text):
        if chan == "sys::err":
            self.err.write(text)
            self.err.flush()
        elif chan == "sys::uart":
            self.out.write("[UART] " + text)
            self.out.flush()
        else:
            self.out.write(text)
            self.out.flush()

    def _arena_alloc(self, n_elems):
        self.arena_total += n_elems * 8
        self.arena_peak = max(self.arena_peak, self.arena_total)

    def _reg_addr(self, name):
        addr = (self.meta.get("registers") or {}).get(name, {}).get("addr")
        return hex(addr) if addr else "0x?"

    def _fmt_reg(self, val):
        if isinstance(val, int) and not isinstance(val, bool):
            return f"{val} (0b{bin(val)[2:]})"
        return x_repr(val)


def _deep_literal(v):
    return v
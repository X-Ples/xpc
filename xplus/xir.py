from __future__ import annotations

import json
import zlib
import struct
import time

from .astnodes import *
from .astnodes import Program, Param
from .parser import Parser as MiniParser, XVecType, XMatType
from .builtins import NATIVES
from .errors import XSyntaxError


class XIRFunc:
    def __init__(self, name, params, ret, instrs, is_entry=False):
        self.name = name
        self.params = params
        self.ret = ret
        self.instrs = instrs
        self.is_entry = is_entry

    def render(self):
        out = [f"func @{self.name}({', '.join('%' + p for p in self.params)})"
               f"{(' -> ' + str(self.ret)) if self.ret else ''} {{"]
        for ins in self.instrs:
            out.append("    " + render_instr(ins))
        out.append("}")
        return "\n".join(out)


def render_instr(ins) -> str:
    op = ins[0]
    try:
        if op == "const":
            return f"{ins[1]} = const {ins[2]!r}"
        if op == "bin":
            return f"{ins[1]} = {ins[2]} {ins[3]}, {ins[4]}"
        if op == "un":
            return f"{ins[1]} = {ins[2]} {ins[3]}"
        if op == "call":
            return f"{ins[1]} = call @{ins[2]}({', '.join(ins[3])})"
        if op == "native":
            return f"{ins[1]} = native {ins[2]}({', '.join(ins[3])})"
        if op == "jmp":
            return f"jmp {ins[1]}"
        if op == "jz":
            return f"jz {ins[1]}, {ins[2]}"
        if op == "jnz":
            return f"jnz {ins[1]}, {ins[2]}"
        if op == "label":
            return f"{ins[1]}:"
        if op == "ret":
            return f"ret {ins[1] if len(ins) > 1 else 'void'}"
        if op == "show":
            specs = ins[4] if len(ins) > 4 else []
            s = f"show {ins[1]}, {ins[2]!r}"
            if ins[3]:
                s += f", args=[{', '.join(ins[3])}]"
            if specs:
                s += f", specs={specs}"
            return s
        if op == "index_get":
            return f"{ins[1]} = index {ins[2]}[{ins[3]}]"
        if op == "index_set":
            return f"index_set {ins[1]}[{ins[2]}] = {ins[3]}"
        if op == "member":
            return f"{ins[1]} = member {ins[2]}.{ins[3]}"
        if op == "set_member":
            return f"set_member {ins[1]}.{ins[2]} = {ins[3]}"
        if op == "array":
            return f"{ins[1]} = array [{', '.join(ins[2])}]"
        if op == "fill_array":
            return f"{ins[1]} = fill {ins[2]}, n={ins[3]}"
        if op == "struct":
            fields = ", ".join(f"{k}={v}" for k, v in ins[3])
            return f"{ins[1]} = struct {ins[2]} {{{fields}}}"
        if op == "enum":
            return f"{ins[1]} = enum {ins[2]}::{ins[3]}({ins[4] if len(ins) > 4 else ''})"
        if op == "wrap_ok":
            return f"{ins[1]} = result.wrap_ok {ins[2]}"
        if op == "wrap_err":
            return f"{ins[1]} = result.wrap_err {ins[2]}"
        if op == "propagate":
            return f"result.propagate {ins[1]}"
        if op == "cast":
            return f"{ins[1]} = cast {ins[2]} as {ins[3]}"
        if op == "gload":
            return f"{ins[1]} = gload {ins[2]}"
        if op == "arena_enter":
            return "arena.enter"
        if op == "arena_exit":
            return "arena.exit"
        if op == "spawn":
            return f"task.spawn @{ins[1]} {ins[2] if len(ins) > 2 else ''}"
        if op == "await_":
            return f"{ins[1]} = await {ins[2]}"
        if op == "asm":
            return f"asm {[i for i in ins[1]]}"
        if op == "reg_write":
            return f"reg_write {ins[1]} <- {ins[2]}"
        if op == "reg_read":
            return f"{ins[1]} = reg_read {ins[2]}"
        if op == "verify":
            return f"verify {ins[1]}"
        if op == "tensor_note":
            return f"; tensor.backend {ins[1]}"
        if op == "nop":
            return "nop"
        if op == "mov":
            return f"{ins[1]} = mov {ins[2]}"
        if op == "gstore":
            return f"gstore {ins[1]} <- {ins[2]}"
        if op == "load":
            return f"{ins[1]} = load {ins[2]}"
        if op == "smv":
            return f"{ins[1]} = smv @{ins[2]}"
        if op == "range":
            return f"{ins[1]} = range {ins[2]}..{ins[3]}"
        if op == "extract":
            return f"{ins[1]} = extract {ins[2]}"
        if op == "wrap_task":
            return f"{ins[1]} = task.wrap {ins[2]}"
        if op == "unwrap_task":
            return f"{ins[1]} = task.unwrap {ins[2]}"
        if op == "unsafe_enter":
            return "unsafe.enter"
        if op == "unsafe_exit":
            return "unsafe.exit"
        if op == "panic":
            return f"panic {ins[1]} {ins[2]!r}"
        if op == "line":
            return f"; line {ins[1]}"
        return f"; ?{op}"
    except Exception:
        return f"; {ins!r}"

class Codegen:
    def __init__(self, table, namespace=None):
        self.table = table
        self.namespace = namespace
        self.funcs: dict[str, XIRFunc] = {}
        self.globals_meta = {}
        self.comptime_fns = []
        self._tmp = 0
        self._label = 0
        self._loop_stack = []
        self._async_count = 0

    def tmp(self):
        self._tmp += 1
        return f"%{self._tmp - 1}"

    def label(self, hint="L"):
        self._label += 1
        return f"{hint}{self._label}"

    def vreg(self, name):
        return f"%v_{name}"

    def gen_module(self, prog: Program):
        consts = [d for d in prog.decls if isinstance(d, ConstDecl)]
        fns = [d for d in prog.decls if isinstance(d, FnDecl)]
        structs = [d for d in prog.decls if isinstance(d, StructDecl)]
        enums = [d for d in prog.decls if isinstance(d, EnumDecl)]
        regs = [d for d in prog.decls if isinstance(d, RegisterDecl)]

        for c in consts:
            self.gen_const(c)

        for f in fns:
            key = f"{self.namespace}::{f.name}" if self.namespace else f.name
            self.gen_fn(f, key)

        return {
            "funcs": self.funcs,
            "consts": self.globals_meta,
            "structs": {s.name: [f.name for f in s.fields] for s in structs},
            "enums": {e.name: e.variants for e in enums},
            "registers": {r.name: {"type": str(r.type), "addr": r.address} for r in regs},
        }

    def gen_const(self, c: ConstDecl):
        gname = f"{self.namespace}::{c.name}" if self.namespace else c.name
        if isinstance(c.value, ComptimeBlock):
            fn_name = f"@comptime::{gname}"
            body = c.value.body
            code = self._compile_fn_body(body, [], None, fn_name)
            self.funcs[fn_name] = XIRFunc(fn_name, [], None, code)
            self.globals_meta[gname] = {"comptime": True, "init_fn": fn_name, "literal": None}
        else:
            lit = try_literal(c.value)
            if lit is not None:
                self.globals_meta[gname] = {"comptime": True, "init_fn": None, "literal": lit}
            else:
                fn_name = f"@comptime::{gname}"
                code = self._compile_fn_body([ReturnStmt(c.value, c.line)], [], None, fn_name)
                self.funcs[fn_name] = XIRFunc(fn_name, [], None, code)
                self.globals_meta[gname] = {"comptime": True, "init_fn": fn_name, "literal": None}

    def gen_fn(self, f: FnDecl, key: str):
        param_regs = [self.vreg(p.name) for p in f.params]
        code = self._compile_fn_body(f.body, f.params, f, key)
        self.funcs[key] = XIRFunc(key, param_regs, f.ret_type, code, is_entry=f.is_entry)

    def _compile_fn_body(self, stmts, params, fnode, key):
        self._tmp = 0
        code = []
        verify_cond = None
        if fnode is not None:
            for a in fnode.attrs:
                if a.name == "verify" and a.args:
                    try:
                        verify_cond = MiniParser(a.args, self.table.src_files.get(key, "main.xp") if hasattr(self.table, "src_files") else "main.xp").parse_expr(allow_struct=True)
                    except XSyntaxError:
                        verify_cond = None
        scope = {p.name: self.vreg(p.name) for p in (params or [])}
        self._gen_stmts(stmts, code, scope, fnode)
        has_ret = any(i[0] == "ret" for i in code)
        if not has_ret:
            t = self.tmp()
            code.append(("const", t, 0))
            code.append(("ret", t))
        return code

    def _gen_stmts(self, stmts, code, scope, fnode):
        for st in stmts:
            self._gen_stmt(st, code, scope, fnode)

    def _gen_stmt(self, st, code, scope, fnode):
        code.append(("line", getattr(st, "line", 0) or 0))
        if isinstance(st, VarDecl) and st.type is not None and str(st.type) == "rational"                 and isinstance(st.init, BinOp) and st.init.op == "/"                 and isinstance(st.init.left, Literal) and isinstance(st.init.right, Literal):
            num = self.tmp(); code.append(("const", num, st.init.left.value))
            den = self.tmp(); code.append(("const", den, st.init.right.value))
            t = self.tmp(); code.append(("native", t, "__rational", [num, den]))
            reg = self.vreg(st.name)
            code.append(("mov", reg, t))
            scope[st.name] = reg
            return
        if isinstance(st, VarDecl):
            src = self._gen_expr(st.init, code, scope) if st.init is not None else None
            reg = self.vreg(st.name)
            if src is not None:
                code.append(("mov", reg, src)) if src != reg else None
            scope[st.name] = reg
            return
        if isinstance(st, ConstDecl):
            src = self._gen_expr(st.value, code, scope) if not isinstance(st.value, ComptimeBlock) else None
            if src is not None:
                code.append(("gstore", st.name, src))
            scope[st.name] = self.vreg(st.name)
            return
        if isinstance(st, Assign):
            self._gen_assign(st, code, scope)
            return
        if isinstance(st, ExprStmt):
            self._gen_expr(st.expr, code, scope)
            return
        if isinstance(st, ShowBlock):
            self._gen_show_block(st, code, scope)
            return
        if isinstance(st, IfStmt):
            end_lbl = self.label("ifend")
            for i, (cond, body) in enumerate(st.branches):
                c = self._gen_expr(cond, code, scope)
                else_lbl = self.label("ifelse")
                code.append(("jz", c, else_lbl))
                self._gen_stmts(body, code, scope, fnode)
                code.append(("jmp", end_lbl))
                code.append(("label", else_lbl))
            if st.else_body:
                self._gen_stmts(st.else_body, code, scope, fnode)
            code.append(("label", end_lbl))
            return
        if isinstance(st, LoopStmt):
            start = self.label("loop")
            cont = self.label("next")
            brk = self.label("brk")
            self._loop_stack.append((cont, brk))
            code.append(("label", start))
            if st.cond is not None:
                c = self._gen_expr(st.cond, code, scope)
                code.append(("jz", c, brk))
            self._gen_stmts(st.body, code, scope, fnode)
            code.append(("label", cont))
            code.append(("jmp", start))
            code.append(("label", brk))
            self._loop_stack.pop()
            return
        if isinstance(st, ForStmt):
            self._gen_for(st, code, scope, fnode)
            return
        if isinstance(st, BreakStmt):
            if self._loop_stack:
                code.append(("jmp", self._loop_stack[-1][1]))
            return
        if isinstance(st, ContinueStmt):
            if self._loop_stack:
                code.append(("jmp", self._loop_stack[-1][0]))
            return
        if isinstance(st, ReturnStmt):
            if st.value is not None:
                v = self._gen_expr(st.value, code, scope)
                if fnode is not None and any(a.name == "verify" for a in fnode.attrs):
                    verify = next(a for a in fnode.attrs if a.name == "verify")
                    if verify.args:
                        try:
                            src = self.table.src_files.get(key := fnode.name, "main.xp") if hasattr(self.table, "src_files") else "main.xp"
                            vc = MiniParser(f"( {verify.args} )").parse_expr(allow_struct=True)
                            scope2 = dict(scope)
                            scope2["result"] = v
                            cv = self._gen_expr(vc, code, scope2)
                            code.append(("verify", cv))
                        except XSyntaxError:
                            pass
                code.append(("ret", v))
            else:
                code.append(("ret",))
            return
        if isinstance(st, GuardStmt):
            ok_lbl = self.label("guard_ok")
            c = self._gen_expr(st.cond, code, scope)
            code.append(("jnz", c, ok_lbl))
            self._gen_stmts(st.else_body, code, scope, fnode)
            code.append(("label", ok_lbl))
            return
        if isinstance(st, MatchStmt):
            self._gen_match(st, code, scope, fnode)
            return
        if isinstance(st, ArenaBlock):
            code.append(("arena_enter",))
            self._gen_stmts(st.body, code, scope, fnode)
            code.append(("arena_exit",))
            return
        if isinstance(st, UnsafeBlock):
            code.append(("unsafe_enter",))
            self._gen_stmts(st.body, code, scope, fnode)
            code.append(("unsafe_exit",))
            return
        if isinstance(st, ComptimeBlock):
            self._gen_stmts(st.body, code, scope, fnode)
            return
        if isinstance(st, TensorCoreBlock):
            code.append(("tensor_note", st.backend))
            self._gen_stmts(st.body, code, scope, fnode)
            return
        if isinstance(st, AsmBlock):
            code.append(("asm", st.instrs))
            return
        return

    def _gen_assign(self, st: Assign, code, scope):
        v = self._gen_expr(st.value, code, scope)
        tgt = st.target
        if isinstance(tgt, Ident):
            if tgt.name in self.table.registers:
                code.append(("reg_write", tgt.name, v))
                return
            reg = scope.get(tgt.name) or self.vreg(tgt.name)
            scope[tgt.name] = reg
            if st.op == "=":
                code.append(("mov", reg, v))
            else:
                bop = {"+=": "+", "-=": "-", "*=": "*", "/=": "/",
                       "+%": "+%", "-%": "-%", "*%": "*%"}.get(st.op, "+")
                code.append(("bin", reg, bop, reg, v))
            return
        if isinstance(tgt, Index):
            obj = self._gen_expr(tgt.obj, code, scope)
            idx = self._gen_expr(tgt.index, code, scope)
            code.append(("index_set", obj, idx, v))
            return
        if isinstance(tgt, Member):
            obj = self._gen_expr(tgt.obj, code, scope)
            code.append(("set_member", obj, tgt.name, v))
            return
        raise XSyntaxError("invalid assignment target", line=st.line)

    def _gen_for(self, st: ForStmt, code, scope, fnode):
        var_reg = self.vreg(st.var)
        cont = self.label("next")
        brk = self.label("brk")
        self._loop_stack.append((cont, brk))
        if st.range_expr is not None:
            s = self._gen_expr(st.range_expr.start, code, scope)
            e = self._gen_expr(st.range_expr.end, code, scope)
            code.append(("mov", var_reg, s))
            cond = self.label("fcond")
            step = self.tmp()
            code.append(("const", step, 1))
            code.append(("label", cond))
            cmp_t = self.tmp()
            code.append(("bin", cmp_t, "<=" if st.range_expr.inclusive else "<", var_reg, e))
            code.append(("jz", cmp_t, brk))
            scope[st.var] = var_reg
            self._gen_stmts(st.body, code, scope, fnode)
            code.append(("label", cont))
            code.append(("bin", var_reg, "+", var_reg, step))
            code.append(("jmp", cond))
            code.append(("label", brk))
        else:
            obj = self._gen_expr(st.iter_expr, code, scope)
            ivar = self.vreg(f"_i_{st.var}")
            zero = self.tmp(); code.append(("const", zero, 0))
            code.append(("mov", ivar, zero))
            ln = self.tmp(); code.append(("native", ln, "length", [obj]))
            cond = self.label("fcond")
            code.append(("label", cond))
            cmp_t = self.tmp()
            code.append(("bin", cmp_t, "<", ivar, ln))
            code.append(("jz", cmp_t, brk))
            elem = self.tmp()
            code.append(("index_get", elem, obj, ivar))
            code.append(("mov", var_reg, elem))
            scope[st.var] = var_reg
            self._gen_stmts(st.body, code, scope, fnode)
            code.append(("label", cont))
            one = self.tmp(); code.append(("const", one, 1))
            code.append(("bin", ivar, "+", ivar, one))
            code.append(("jmp", cond))
            code.append(("label", brk))
        self._loop_stack.pop()

    def _gen_match(self, st: MatchStmt, code, scope, fnode):
        subj = self._gen_expr(st.subject, code, scope)
        end_lbl = self.label("mend")
        for (pat, binding, body) in st.arms:
            next_lbl = self.label("mnext")
            if pat == "_":
                self._gen_match_body(binding, subj, body, code, scope, fnode)
                code.append(("jmp", end_lbl))
                continue
            cond = None
            if isinstance(pat, Literal):
                t = self.tmp()
                code.append(("const", t, pat.value))
                cond = self.tmp()
                code.append(("bin", cond, "==", subj, t))
            elif isinstance(pat, str) and "::" in pat:
                ename, variant = pat.split("::", 1)
                t = self.tmp()
                code.append(("enum", t, ename, variant, None))
                cond = self.tmp()
                code.append(("bin", cond, "==", subj, t))
            elif isinstance(pat, str) and pat in ("Ok", "Err", "Some", "None"):
                t = self.tmp()
                code.append(("const", t, pat))
                cond = self.tmp()
                code.append(("bin", cond, "==tag", subj, t))
            elif isinstance(pat, str):
                t = self.tmp()
                code.append(("enum", t, "?", pat, None))
                cond = self.tmp()
                code.append(("bin", cond, "==", subj, t))
            code.append(("jz", cond, next_lbl))
            self._gen_match_body(binding, subj, body, code, scope, fnode)
            code.append(("jmp", end_lbl))
            code.append(("label", next_lbl))
        code.append(("panic", "E304", "match fell through: no arm matched"))
        code.append(("label", end_lbl))

    def _gen_match_body(self, binding, subj, body, code, scope, fnode):
        scope2 = dict(scope)
        if binding:
            b = self.tmp()
            code.append(("extract", b, subj))
            scope2[binding] = b
        self._gen_stmts(body, code, scope2, fnode)

    def _gen_show_block(self, st: ShowBlock, code, scope):
        args = []
        injections = sorted(st.injections, key=lambda x: x[0])
        for (_, expr, _tann) in injections:
            args.append(self._gen_expr(expr, code, scope))
        code.append(("show", st.channel, st.fmt, args, []))

    def _gen_expr(self, e, code, scope):
        if isinstance(e, Literal):
            t = self.tmp()
            code.append(("const", t, e.value))
            return t
        if isinstance(e, SMV):
            t = self.tmp()
            code.append(("smv", t, e.name))
            return t
        if isinstance(e, Ident):
            if e.name in scope:
                return scope[e.name]
            if "::" in e.name:
                ename, variant = e.name.split("::", 1)
                if ename in self.table.enums:
                    t = self.tmp()
                    code.append(("enum", t, ename, variant, None))
                    return t
                if e.name in self.table.consts or e.name in self.globals_meta:
                    t = self.tmp()
                    code.append(("gload", t, e.name))
                    return t
                t = self.tmp()
                code.append(("const", t, {"__fnref__": e.name}))
                return t
            nsname = f"{self.namespace}::{e.name}" if self.namespace else None
            if e.name in self.table.consts:
                t = self.tmp()
                key = e.name
                if key not in self.globals_meta and nsname and nsname in self.globals_meta:
                    key = nsname
                code.append(("gload", t, key))
                return t
            if nsname and nsname in self.table.consts:
                t = self.tmp()
                code.append(("gload", t, nsname))
                return t
            if e.name in self.table.registers:
                t = self.tmp()
                code.append(("reg_read", t, e.name))
                return t
            if e.name in self.table.functions:
                t = self.tmp()
                code.append(("const", t, {"__fnref__": e.name}))
                return t
            t = self.tmp()
            code.append(("load", t, e.name))
            return t
        if isinstance(e, BinOp):
            return self._gen_binop(e, code, scope)
        if isinstance(e, UnOp):
            v = self._gen_expr(e.operand, code, scope)
            t = self.tmp()
            code.append(("un", t, e.op, v))
            return t
        if isinstance(e, Cast):
            v = self._gen_expr(e.expr, code, scope)
            t = self.tmp()
            code.append(("cast", t, v, str(e.type)))
            return t
        if isinstance(e, Call):
            return self._gen_call(e, code, scope)
        if isinstance(e, Member):
            obj = self._gen_expr(e.obj, code, scope)
            t = self.tmp()
            code.append(("member", t, obj, e.name))
            return t
        if isinstance(e, Index):
            obj = self._gen_expr(e.obj, code, scope)
            idx = self._gen_expr(e.index, code, scope)
            t = self.tmp()
            code.append(("index_get", t, obj, idx))
            return t
        if isinstance(e, RangeExpr):
            s = self._gen_expr(e.start, code, scope)
            en = self._gen_expr(e.end, code, scope)
            t = self.tmp()
            code.append(("range", t, s, en, e.inclusive))
            return t
        if isinstance(e, Pipeline):
            return self._gen_pipeline(e, code, scope)
        if isinstance(e, Propagate):
            v = self._gen_expr(e.expr, code, scope)
            code.append(("propagate", v))
            t = self.tmp()
            code.append(("extract", t, v))
            return t
        if isinstance(e, StructLit):
            fields = []
            for fname, fexpr in e.fields.items():
                fields.append((fname, self._gen_expr(fexpr, code, scope)))
            t = self.tmp()
            code.append(("struct", t, e.name, fields))
            return t
        if isinstance(e, EnumLit):
            t = self.tmp()
            code.append(("enum", t, e.name, e.variant, None))
            return t
        if isinstance(e, ArrayLit):
            items = [self._gen_expr(i, code, scope) for i in e.items]
            t = self.tmp()
            code.append(("array", t, items))
            return t
        if isinstance(e, ArrayFill):
            v = self._gen_expr(e.value, code, scope)
            n = self._gen_expr(e.size, code, scope)
            t = self.tmp()
            code.append(("fill_array", t, v, n))
            return t
        if isinstance(e, AsyncExpr):
            fn_name = f"@async::{self._async_count}"
            self._async_count += 1
            code_blk = self._compile_fn_body(e.body, [], None, fn_name)
            self.funcs[fn_name] = XIRFunc(fn_name, [], None, code_blk)
            t = self.tmp()
            code.append(("call", t, fn_name, []))
            code.append(("wrap_task", t, t))
            return t
        if isinstance(e, AwaitExpr):
            v = self._gen_expr(e.expr, code, scope)
            t = self.tmp()
            code.append(("unwrap_task", t, v))
            return t
        if isinstance(e, GradExpr):
            t = self.tmp()
            code.append(("const", t, {"__grad__": e.fn_name}))
            return t
        return self.tmp()

    def _gen_binop(self, e: BinOp, code, scope):
        if e.op in ("&&", "||"):
            l = self._gen_expr(e.left, code, scope)
            t = self.tmp()
            end = self.label("sc")
            if e.op == "&&":
                code.append(("mov", t, l))
                code.append(("jz", t, end))
                r = self._gen_expr(e.right, code, scope)
                code.append(("mov", t, r))
            else:
                code.append(("mov", t, l))
                code.append(("jnz", t, end))
                r = self._gen_expr(e.right, code, scope)
                code.append(("mov", t, r))
            code.append(("label", end))
            return t
        l = self._gen_expr(e.left, code, scope)
        r = self._gen_expr(e.right, code, scope)
        t = self.tmp()
        op = e.op
        if op == "x":
            op = "matmul"
        elif op == ".x":
            op = "dot"
        code.append(("bin", t, op, l, r))
        return t

    def _gen_pipeline(self, e: Pipeline, code, scope):
        left = self._gen_expr(e.left, code, scope)
        if isinstance(e.right, Call):
            args = [self._gen_expr(a, code, scope) for a in e.right.args]
            args = [left] + args
            t = self.tmp()
            if e.right.named_args:
                code.append(("call", t, e.right.callee, args, dict(e.right.named_args)))
            else:
                code.append(("call", t, e.right.callee, args))
            return t
        callee = e.right.name if isinstance(e.right, Ident) else str(e.right)
        t = self.tmp()
        code.append(("call", t, callee, [left]))
        return t

    def _gen_call(self, e: Call, code, scope):
        if e.callee == "Ok" and len(e.args) == 1:
            v = self._gen_expr(e.args[0], code, scope)
            t = self.tmp()
            code.append(("wrap_ok", t, v))
            return t
        if e.callee == "Err" and len(e.args) == 1:
            v = self._gen_expr(e.args[0], code, scope)
            t = self.tmp()
            code.append(("wrap_err", t, v))
            return t
        if e.callee == "Some" and len(e.args) == 1:
            v = self._gen_expr(e.args[0], code, scope)
            t = self.tmp()
            code.append(("wrap_some", t, v))
            return t
        if e.callee == "None" and not e.args:
            t = self.tmp()
            code.append(("wrap_none", t))
            return t
        if e.callee == "show" and e.args and isinstance(e.args[1], Literal) and e.args[1].kind == "string":
            chan = e.args[0].name if isinstance(e.args[0], Ident) else str(e.args[0])
            fmt, specs, fmt_args = self._compile_format(e.args[1].value, code, scope)
            args = list(fmt_args)
            for a in e.args[2:]:
                args.append(self._gen_expr(a, code, scope))
            t = self.tmp()
            code.append(("show", chan, fmt, args, specs))
            return t
        callee = e.callee
        if "::" not in callee and callee in scope:
            args2 = [self._gen_expr(a, code, scope) for a in e.args]
            t = self.tmp()
            code.append(("callv", t, scope[callee], args2))
            return t
        if "::" not in callee and self.namespace:
            from .builtins import NATIVES as _NAT
            if e.callee.split("::")[-1] not in _NAT:
                nsname = f"{self.namespace}::{callee}"
                if nsname in self.table.functions:
                    callee = nsname
        args = [self._gen_expr(a, code, scope) for a in e.args]
        named = {k: self._gen_expr(v, code, scope) for k, v in e.named_args.items()}
        t = self.tmp()
        if named:
            code.append(("call", t, callee, args, named))
        else:
            code.append(("call", t, callee, args))
        return t

    def _compile_format(self, fmt: str, code, scope):
        out = []
        specs = []
        args = []
        i = 0
        while i < len(fmt):
            ch = fmt[i]
            if ch == "{" and i + 1 < len(fmt) and fmt[i + 1] != "{":
                j = fmt.index("}", i)
                inner = fmt[i + 1:j]
                expr_part, cast_t, spec = parse_placeholder(inner)
                if cast_t:
                    ast = MiniParser(f"({expr_part}) as {cast_t}").parse_expr(allow_struct=True)
                else:
                    ast = MiniParser(f"({expr_part})").parse_expr(allow_struct=True)
                src = self._gen_expr(ast, code, scope)
                specs.append(spec)
                args.append(src)
                out.append("{" + str(len(args) - 1) + (";" + spec if spec else "") + "}")
                i = j + 1
                continue
            out.append(ch)
            i += 1
        return "".join(out), specs, args


def parse_placeholder(inner: str):
    cast_t = None
    if " as " in inner:
        inner, cast_t = inner.split(" as ", 1)
        cast_t = cast_t.strip()
        if ":" in cast_t:
            cast_t, spec0 = cast_t.split(":", 1)
            return inner.strip(), cast_t.strip(), ":" + spec0.strip()
    spec = None
    for idx in range(len(inner) - 1):
        if inner[idx] == ":" and inner[idx + 1] != ":" and (idx == 0 or inner[idx - 1] != ":"):
            spec = ":" + inner[idx + 1:].strip()
            inner = inner[:idx]
            break
    return inner.strip(), cast_t, spec


def try_literal(e):
    if isinstance(e, Literal):
        return e.value
    if isinstance(e, UnOp) and e.op == "-" and isinstance(e.operand, Literal):
        return -e.operand.value
    if isinstance(e, ArrayLit):
        vals = [try_literal(i) for i in e.items]
        if all(v is not None for v in vals):
            return vals
    return None

def funcs_to_json(funcs: dict):
    out = {}
    for name, f in funcs.items():
        if isinstance(f, XIRFunc):
            out[name] = {
                "params": f.params,
                "ret": str(f.ret) if f.ret is not None else None,
                "is_entry": f.is_entry,
                "instrs": [list(i) for i in f.instrs],
            }
        else:
            out[name] = {
                "params": f.get("params", []),
                "ret": f.get("ret"),
                "is_entry": f.get("is_entry", False),
                "instrs": f.get("instrs", []),
            }
    return out


def serialize_xbc(module_ir: dict, meta: dict) -> bytes:
    body = json.dumps({"funcs": funcs_to_json(module_ir["funcs"]),
                       "consts": module_ir["consts"],
                       "structs": module_ir["structs"],
                       "enums": module_ir["enums"],
                       "registers": module_ir["registers"]}).encode()
    body_z = zlib.compress(body, 6)
    header = json.dumps(meta).encode()
    buf = bytearray()
    buf += b"XBC1"
    buf += struct.pack("<BBH", 1, 0, 0)
    buf += struct.pack("<I", len(header))
    buf += header
    buf += struct.pack("<I", len(body_z))
    buf += body_z
    return bytes(buf)


def deserialize_xbc(data: bytes):
    if data[:4] != b"XBC1":
        raise ValueError("not an XBC file (magic mismatch)")
    ver, flags, _res = struct.unpack("<BBH", data[4:8])
    hlen = struct.unpack("<I", data[8:12])[0]
    header = json.loads(data[12:12 + hlen])
    blen = struct.unpack("<I", data[12 + hlen:16 + hlen])[0]
    body = json.loads(zlib.decompress(data[16 + hlen:16 + hlen + blen]))
    return header, body
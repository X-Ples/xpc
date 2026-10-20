from __future__ import annotations

from .astnodes import *
from .astnodes import Program, Param
from .parser import (XType, XNamedType, XVecType, XMatType, XMatShapeType,
                     XArrayType, XGenericType, XOptionalType, PRIMITIVE_TYPES)
from .errors import XTypeError, XWarning
from .builtins import NATIVES, SMV_TABLE

INT_TYPES = {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "usize", "int"}
FLOAT_TYPES = {"f32", "f64"}

MONAD_CALLS = {"Ok", "Err", "Some", "None"}


class FnSig:
    def __init__(self, name, params, ret, pub=False, ns=None, node=None):
        self.name = name
        self.params = params
        self.ret = ret
        self.pub = pub
        self.ns = ns
        self.node = node


class ProgramTable:
    def __init__(self):
        self.functions: dict[str, FnSig] = {}
        self.structs: dict[str, dict] = {}
        self.enums: dict[str, list] = {}
        self.consts: dict[str, tuple] = {}
        self.registers: dict[str, XType] = {}
        self.modules: dict[str, set] = {}


class Checker:
    def __init__(self, table: ProgramTable):
        self.table = table
        self.errors: list = []
        self.warnings: list = []
        self.vector_candidates: list = []
        self.const_eval_order: list = []

    def err(self, file, line, col, code, msg, hint=None):
        self.errors.append(XTypeError(msg, file=file, line=line, col=col,
                                      hint=hint, code=code))

    def warn(self, file, line, code, msg, hint=None):
        self.warnings.append(XWarning(code, msg, file=file, line=line, hint=hint))

    def check(self, prog: Program) -> None:
        for decl in prog.decls:
            self.collect_decl(decl, prog)

        for decl in prog.decls:
            if isinstance(decl, FnDecl):
                self.check_fn(decl, prog)

        self.scan_vectorization(prog)

    def collect_decl(self, d, prog):
        file = prog.file
        ns = d.namespace if hasattr(d, "namespace") else None
        if isinstance(d, ConstDecl):
            key = f"{ns}::{d.name}" if ns else d.name
            self.table.consts[key] = (d.type, d, d.comptime)
        elif isinstance(d, StructDecl):
            fields = {}
            for f in d.fields:
                fields[f.name] = f.type
            self.table.structs[d.name] = fields
        elif isinstance(d, EnumDecl):
            self.table.enums[d.name] = d.variants
        elif isinstance(d, RegisterDecl):
            self.table.registers[d.name] = d.type
        elif isinstance(d, FnDecl):
            params = [(p.name, p.type) for p in d.params]
            sig = FnSig(d.name, params, d.ret_type, pub=d.is_pub, ns=ns, node=d)
            key = f"{ns}::{d.name}" if ns else d.name
            self.table.functions[key] = sig

    def check_fn(self, fn: FnDecl, prog):
        file = prog.file
        scope = {}
        mutated = set()
        returned = [False]

        for a in fn.attrs:
            if a.name not in self._known_attrs():
                self.warn(file, a.line, "W101",
                          f"attribute '@{a.name}' recognized but not enforced in v0.1",
                          hint="this feature is on the v0.5 roadmap (LLVM backend)")

        for (pname, ptype) in [(p.name, p.type) for p in fn.params]:
            scope[pname] = ptype

        self.check_block(fn.body, scope, mutated, returned, file, fn)

        for vname in scope:
            if vname not in mutated and vname not in ("_",):
                pass
    def _known_attrs(self):
        return {"entry", "interrupt", "hotpatch", "tmr", "cluster", "enclave",
                "verify", "grad", "gpu", "tensor_core", "hard_realtime", "kalman",
                "kernel", "resilient", "deterministic", "no_std", "inline"}

    def check_block(self, stmts, scope, mutated, returned, file, fn):
        for st in stmts:
            if returned[0]:
                self.warn(file, getattr(st, "line", 0), "W102",
                          "unreachable code after return")
            self.check_stmt(st, scope, mutated, returned, file, fn)
    def check_stmt(self, st, scope, mutated, returned, file, fn):
        if isinstance(st, VarDecl):
            if st.type is not None and str(st.type) == "rational" \
                    and isinstance(st.init, BinOp) and st.init.op == "/" \
                    and isinstance(st.init.left, Literal) and isinstance(st.init.right, Literal):
                scope[st.name] = st.type
                return
            t = self.infer(st.init, scope, file) if st.init is not None else st.type
            if st.type is not None and st.init is not None:
                it = self.infer(st.init, scope, file)
                self.check_assignable(st.type, it, st.line, file, context=f"variabel '{st.name}'")
                t = st.type
            if t is None and st.init is None:
                self.err(file, st.line, 0, "E206",
                         f"variable '{st.name}' has no type and no initializer")
                t = XNamedType("any")
            if st.name in scope:
                self.warn(file, st.line, "W103",
                          f"variable '{st.name}' shadows an earlier declaration")
            scope[st.name] = t
            if st.name in self.table.consts:
                self.err(file, st.line, 0, "E208",
                         f"cannot shadow global const '{st.name}'")
        elif isinstance(st, Assign):
            self.check_assign(st, scope, mutated, file)
        elif isinstance(st, ExprStmt):
            self.infer(st.expr, scope, file)
        elif isinstance(st, ShowBlock):
            for (_, expr, _) in st.injections:
                self.infer(expr, scope, file)
        elif isinstance(st, IfStmt):
            for (cond, body) in st.branches:
                ct = self.infer(cond, scope, file)
                self.expect_bool(ct, cond, file, "kondisi if")
                self.check_block(body, scope, mutated, returned, file, fn)
            if st.else_body:
                self.check_block(st.else_body, scope, mutated, returned, file, fn)
        elif isinstance(st, LoopStmt):
            if st.cond is not None:
                ct = self.infer(st.cond, scope, file)
                self.expect_bool(ct, st.cond, file, "kondisi loop")
            self.check_block(st.body, scope, mutated, returned, file, fn)
        elif isinstance(st, ForStmt):
            inner = dict(scope)
            mutated2 = set(mutated)
            if st.range_expr is not None:
                st_t = self.infer(st.range_expr.start, scope, file)
                en_t = self.infer(st.range_expr.end, scope, file)
                inner[st.var] = st_t
            elif st.iter_expr is not None:
                it = self.infer(st.iter_expr, scope, file)
                inner[st.var] = self.element_type(it, st.line, file)
            self.check_block(st.body, inner, mutated2, returned, file, fn)
        elif isinstance(st, (BreakStmt, ContinueStmt)):
            pass
        elif isinstance(st, ReturnStmt):
            returned[0] = True
            rt = self.infer(st.value, scope, file) if st.value is not None else XNamedType("void")
            if fn is not None and fn.ret_type is not None and st.value is not None:
                self.check_assignable(fn.ret_type, rt, st.line, file,
                                      context=f"return of '{fn.name}'")
        elif isinstance(st, GuardStmt):
            ct = self.infer(st.cond, scope, file)
            self.expect_bool(ct, st.cond, file, "kondisi guard")
            self.check_block(st.else_body, scope, mutated, list(returned), file, fn)
        elif isinstance(st, MatchStmt):
            subj_t = self.infer(st.subject, scope, file)
            has_wild = False
            for (pat, binding, body) in st.arms:
                if pat == "_":
                    has_wild = True
                else:
                    self.check_pattern(pat, binding, subj_t, scope, file, st.line)
                inner = dict(scope)
                if binding:
                    inner[binding] = self.binding_type(subj_t, pat)
                self.check_block(body, inner, set(), returned, file, fn)
            if not has_wild and subj_t is not None:
                self.warn(file, st.line, "W104",
                          "match without a '_' wildcard can panic E304 when no arm matches")
        elif isinstance(st, ArenaBlock):
            self.check_block(st.body, scope, mutated, returned, file, fn)
        elif isinstance(st, UnsafeBlock):
            self.check_block(st.body, scope, mutated, returned, file, fn)
        elif isinstance(st, ComptimeBlock):
            self.check_block(st.body, scope, mutated, returned, file, fn)
        elif isinstance(st, TensorCoreBlock):
            self.check_block(st.body, scope, mutated, returned, file, fn)
        elif isinstance(st, AsmBlock):
            pass
        elif isinstance(st, ConstDecl):
            self.table.consts[st.name] = (st.type, st, st.comptime)
        else:
            pass

    def check_assign(self, st: Assign, scope, mutated, file):
        tgt = st.target
        val_t = self.infer(st.value, scope, file)
        if isinstance(tgt, Ident):
            name = tgt.name
            if name in self.table.consts:
                self.err(file, st.line, 0, "E208",
                         f"cannot assign to const '{name}'",
                         hint="const bersifat immutable (CTFE)")
                return
            if name in self.table.registers:
                mutated.add(name)
                return
            if name not in scope:
                self.err(file, st.line, tgt.line, "E201",
                         f"undefined variable '{name}'",
                         hint="declare it first with 'name := value;' or 'name: type = value;'")
                return
            mutated.add(name)
            cur = scope[name]
            self.check_assignable(cur, val_t, st.line, file, context=f"assign to '{name}'")
        elif isinstance(tgt, Index):
            self.infer(tgt.obj, scope, file)
            self.infer(tgt.index, scope, file)
            if isinstance(tgt.obj, Ident) and tgt.obj.name in scope:
                mutated.add(tgt.obj.name)
        elif isinstance(tgt, Member):
            self.infer(tgt.obj, scope, file)
            if isinstance(tgt.obj, Ident) and tgt.obj.name in scope:
                mutated.add(tgt.obj.name)
        else:
            self.err(file, st.line, 0, "E110", "invalid assignment target")

    def check_assignable(self, target_t, val_t, line, file, context=""):
        if target_t is None or val_t is None:
            return
        ts, vs = str(target_t), str(val_t)
        if ts == vs or ts == "any" or vs == "any" or ts == "void":
            return
        if ts in FLOAT_TYPES and (vs in INT_TYPES or vs == "int"):
            return
        if ts in INT_TYPES and vs in INT_TYPES:
            return
        if ts == "f64" and vs == "f32":
            return
        if ts in ("vec4", "vec2", "vec3", "vec8", "vec16") and vs == "array":
            return
        if ts.startswith("mat") and vs == "array":
            return
        self.err(file, line, 0, "E206",
                 f"type mismatch in {context}: expected '{ts}', got '{vs}'",
                 hint="use 'as' for an explicit cast, e.g. (x as f64)")

    def check_pattern(self, pat, binding, subj_t, scope, file, line):
        if isinstance(pat, Literal):
            return
        if isinstance(pat, str):
            if "::" in pat:
                ename, variant = pat.split("::", 1)
                if ename not in self.table.enums:
                    self.err(file, line, 0, "E211", f"unknown enum '{ename}'")
                elif variant not in self.table.enums[ename]:
                    self.err(file, line, 0, "E211",
                             f"unknown variant '{variant}' in enum '{ename}'",
                             hint=f"variants: {', '.join(self.table.enums[ename])}")
            elif pat in MONAD_CALLS:
                pass
            elif subj_t is not None and str(subj_t) in self.table.enums:
                if pat not in self.table.enums[str(subj_t)]:
                    self.err(file, line, 0, "E211",
                             f"unknown variant '{pat}' in enum '{subj_t}'")
            elif pat not in ("true", "false"):
                self.warn(file, line, "W105", f"pattern '{pat}' tidak dikenal — diasumsikan wildcard")

    def binding_type(self, subj_t, pat=None):
        if isinstance(subj_t, XOptionalType):
            return subj_t.inner
        if isinstance(subj_t, XGenericType) and subj_t.name == "Result":
            if pat == "Err":
                return subj_t.args[1] if len(subj_t.args) > 1 else XNamedType("string")
            return subj_t.args[0]
        return XNamedType("any")

    def element_type(self, it, line, file):
        if isinstance(it, XVecType):
            return XNamedType("f64")
        if isinstance(it, XArrayType):
            return it.elem
        if isinstance(it, XNamedType) and it.name == "string":
            return XNamedType("string")
        if it is None or (isinstance(it, XNamedType) and it.name == "any"):
            return XNamedType("any")
        self.warn(file, line, "W106", f"type '{it}' bukan iterable (array/vec/string)")
        return XNamedType("any")

    def expect_bool(self, t, expr, file, ctx):
        if t is None:
            return
        s = str(t)
        if s != "bool" and s != "any":
            self.err(file, expr.line, 0, "E206",
                     f"{ctx} must be of type bool, got '{s}'",
                     hint="contoh: if speed > 500 { ... }")

    def infer(self, e, scope, file):
        if e is None:
            return None
        if isinstance(e, Literal):
            return {"int": XNamedType("int"), "float": XNamedType("f64"),
                    "string": XNamedType("string"), "bool": XNamedType("bool")}[e.kind]
        if isinstance(e, SMV):
            if e.name not in SMV_TABLE:
                close = [k for k in SMV_TABLE if k.startswith(e.name[:4])]
                self.err(file, e.line, 0, "E120",
                         f"unknown System Magic Variable '@{e.name}'",
                         hint=f"SMV tersedia: {', '.join(sorted(SMV_TABLE)[:20])} ...")
            return XNamedType(SMV_TABLE.get(e.name, ("any",))[0])
        if isinstance(e, Ident):
            name = e.name
            if "::" in name and name not in scope:
                return self.infer_path_ident(e, scope, file)
            if name in scope:
                return scope[name]
            nsname = f"{self.current_ns}::{name}" if self.current_ns else None
            if name in self.table.consts:
                ct = self.table.consts[name][0]
                return ct or XNamedType("any")
            if nsname and nsname in self.table.consts:
                ct = self.table.consts[nsname][0]
                return ct or XNamedType("any")
            if name in self.table.functions or (nsname and nsname in self.table.functions):
                return XNamedType("fn")
            if name in self.table.registers:
                return self.table.registers[name]
            self.err(file, e.line, 0, "E201", f"undefined variable '{name}'",
                     hint="check the spelling or declare it first")
            return None
        if isinstance(e, BinOp):
            lt = self.infer(e.left, scope, file)
            rt = self.infer(e.right, scope, file)
            return self.infer_binop(e, lt, rt, scope, file)
        if isinstance(e, UnOp):
            t = self.infer(e.operand, scope, file)
            if e.op == "!":
                return XNamedType("bool")
            return t
        if isinstance(e, Cast):
            self.infer(e.expr, scope, file)
            return e.type
        if isinstance(e, Call):
            return self.infer_call(e, scope, file)
        if isinstance(e, Member):
            ot = self.infer(e.obj, scope, file)
            return self.infer_member(e, ot, scope, file)
        if isinstance(e, Index):
            ot = self.infer(e.obj, scope, file)
            self.infer(e.index, scope, file)
            if isinstance(ot, XArrayType):
                return ot.elem
            if isinstance(ot, XVecType):
                return XNamedType("f64")
            if isinstance(ot, XNamedType) and ot.name == "string":
                return XNamedType("string")
            return XNamedType("any")
        if isinstance(e, RangeExpr):
            self.infer(e.start, scope, file)
            self.infer(e.end, scope, file)
            return XNamedType("range")
        if isinstance(e, Pipeline):
            lt = self.infer(e.left, scope, file)
            self.infer(e.right, scope, file) if not isinstance(e.right, Call) else self.infer_pipeline_call(e.right, scope, file)
            rt = None
            if isinstance(e.right, Call):
                sig = self.table.functions.get(e.right.callee)
                rt = sig.ret if sig else None
            return rt or XNamedType("any")
        if isinstance(e, Propagate):
            t = self.infer(e.expr, scope, file)
            if isinstance(t, XGenericType) and t.name == "Result":
                return t.args[0]
            if isinstance(t, XOptionalType):
                return t.inner
            return XNamedType("any")
        if isinstance(e, StructLit):
            fields = self.table.structs.get(e.name)
            if fields is None:
                self.err(file, e.line, 0, "E210", f"unknown struct '{e.name}'")
                return None
            for fname, fexpr in e.fields.items():
                if fname not in fields:
                    self.err(file, e.line, 0, "E210",
                             f"struct '{e.name}' has no field '{fname}'",
                             hint=f"fields: {', '.join(fields)}")
                else:
                    ft = self.infer(fexpr, scope, file)
                    if fields[fname] is not None:
                        self.check_assignable(fields[fname], ft, fexpr.line if hasattr(fexpr, 'line') else e.line,
                                              file, context=f"field '{e.name}.{fname}'")
            return XNamedType(e.name)
        if isinstance(e, EnumLit):
            if e.name not in self.table.enums:
                self.err(file, e.line, 0, "E211", f"unknown enum '{e.name}'")
            elif e.variant not in self.table.enums[e.name]:
                self.err(file, e.line, 0, "E211",
                         f"unknown variant '{e.variant}' in enum '{e.name}'")
            return XNamedType(e.name)
        if isinstance(e, ArrayLit):
            for it in e.items:
                self.infer(it, scope, file)
            return XNamedType("array")
        if isinstance(e, ArrayFill):
            self.infer(e.value, scope, file)
            self.infer(e.size, scope, file)
            return XNamedType("array")
        if isinstance(e, AsyncExpr):
            self.check_block(e.body, scope, set(), [False], file, None)
            return XNamedType("task")
        if isinstance(e, AwaitExpr):
            self.infer(e.expr, scope, file)
            return XNamedType("any")
        if isinstance(e, GradExpr):
            if e.fn_name not in self.table.functions:
                self.err(file, e.line, 0, "E202", f"undefined function '{e.fn_name}' for @grad")
            return XNamedType("fn")
        return None

    def infer_path_ident(self, e: Ident, scope, file):
        ename, variant = e.name.split("::", 1)
        if ename in self.table.enums:
            if variant not in self.table.enums[ename]:
                self.err(file, e.line, 0, "E211",
                         f"unknown variant '{variant}' in enum '{ename}'",
                         hint=f"variants: {', '.join(self.table.enums[ename])}")
            return XNamedType(ename)
        if ename in self.table.modules:
            if e.name in self.table.consts:
                ct = self.table.consts[e.name][0]
                return ct or XNamedType("any")
            if variant in self.table.modules[ename]:
                return XNamedType("fn")
            self.err(file, e.line, 0, "E205",
                     f"'{variant}' is not a public symbol of module '{ename}'")
            return None
        self.err(file, e.line, 0, "E211", f"unknown enum or module '{ename}'",
                 hint=f"try 'xpc install {ename}'")
        return None

    def infer_binop(self, e: BinOp, lt, rt, scope, file):
        ls, rs = str(lt) if lt else "?", str(rt) if rt else "?"
        line = e.line
        if e.op in ("&&", "||"):
            return XNamedType("bool")
        if e.op in ("==", "!=", "<", ">", "<=", ">="):
            return XNamedType("bool")

        if e.op in ("x", ".x"):
            return self.check_matmul(e, lt, rt, file)

        if e.op in ("+", "-", "*", "/", "%", "^"):
            if e.op == "+" and (ls == "string" or rs == "string"):
                if ls == "any" or rs == "any" or (ls == "string" and rs == "string"):
                    return XNamedType("string")
                self.err(file, line, 0, "E206",
                         f"cannot add '{ls}' + '{rs}'",
                         hint="manual conversion: to_string(x) or use show with {var}")
                return XNamedType("string")
            if ls.startswith("vec") and rs.startswith("vec"):
                if ls != rs:
                    self.err(file, line, 0, "E204",
                             f"vector length mismatch: '{ls}' vs '{rs}'",
                             hint="SIMD elementwise ops need equal lengths")
                return lt
            if ls.startswith("vec") and rs not in ("array",):
                return lt
            if rs.startswith("vec") and ls not in ("array",):
                return rt
            if ls == "rational" or rs == "rational":
                return XNamedType("rational")
            if ls in FLOAT_TYPES or rs in FLOAT_TYPES or ls == "f64":
                return XNamedType("f64")
            if ls in INT_TYPES and rs in INT_TYPES:
                return XNamedType("int")
            if ls == "int" or rs == "int":
                return XNamedType("int")
            if ls == "array" or rs == "array":
                self.err(file, line, 0, "E206",
                         "elementwise array arithmetic is not supported in v0.1 — use vecN",
                         hint="vec4 = [1.0, 2.0, 3.0, 4.0] lalu v1 + v2 → SIMD")
            return XNamedType("f64")
        return None

    def check_matmul(self, e, lt, rt, file):
        line = e.line
        if e.op == ".x":
            if not (isinstance(lt, XVecType) and isinstance(rt, XVecType)):
                self.warn(file, line, "W107", "'.x' dot product requires vec x vec")
            return XNamedType("f64")
        if isinstance(lt, XMatShapeType) and isinstance(rt, XMatShapeType):
            lc = lt.dims[1] if len(lt.dims) > 1 else None
            rr = rt.dims[0] if rt.dims else None
            if lc is not None and rr is not None and str(lc) != str(rr):
                self.err(file, line, 0, "E204",
                         f"Dimension Mismatch in Matrix Multiplication: [{lt.dims[0]} x {lc}] x [{rr} x {rt.dims[1] if len(rt.dims) > 1 else '?'}]",
                         hint=f"columns of A ({lc}) must equal rows of B ({rr})")
            return XMatShapeType([lt.dims[0], rt.dims[1] if len(rt.dims) > 1 else "*"])
        if isinstance(lt, XMatType) and isinstance(rt, XVecType):
            if lt.c != rt.n:
                self.err(file, line, 0, "E204",
                         f"Dimension Mismatch: mat{lt.r}x{lt.c} x vec{rt.n}",
                         hint=f"matrix columns ({lt.c}) must equal the vector length ({rt.n})")
            return XVecType(lt.r)
        if isinstance(lt, XMatType) and isinstance(rt, XMatType):
            if lt.c != rt.r:
                self.err(file, line, 0, "E204",
                         f"Dimension Mismatch: mat{lt.r}x{lt.c} x mat{rt.r}x{rt.c}",
                         hint=f"columns of A ({lt.c}) must equal rows of B ({rt.r})")
            return XMatType(lt.r, rt.c)
        self.warn(file, line, "W107", f"matmul 'x' requires mat/vec operands, got '{lt}' x '{rt}'")
        return XNamedType("any")

    def infer_call(self, e: Call, scope, file):
        name = e.callee
        args_to_check = e.args[1:] if name == "show" else e.args
        for a in args_to_check:
            self.infer(a, scope, file)
        for k, v in e.named_args.items():
            self.infer(v, scope, file)

        if name in MONAD_CALLS:
            if name == "Ok":
                return XGenericType("Result", [self.infer(e.args[0], scope, file) if e.args else XNamedType("any"), XNamedType("string")])
            if name == "Some":
                return XOptionalType(self.infer(e.args[0], scope, file) if e.args else XNamedType("any"))
            if name in ("Err", "None"):
                return XNamedType("any")

        if name == "show":
            return XNamedType("void")

        base = name.split("::")[-1]
        if "::" in name:
            ns = name.split("::")[0]
            if ns not in self.table.modules:
                self.err(file, e.line, 0, "E203",
                         f"unknown module '{ns}'",
                         hint=f"run 'xpc install {ns}' or check the using line at the top of the file")
                return None
            if base not in self.table.modules[ns]:
                pub = self.table.modules[ns]
                self.err(file, e.line, 0, "E205",
                         f"'{base}' is private to module '{ns}'" if base in self.all_module_syms(ns)
                         else f"module '{ns}' has no public symbol '{base}'",
                         hint=f"pub: {', '.join(sorted(pub)) if pub else '(none)'}")
                return None

        sig = self.table.functions.get(name)
        if sig is None and self.current_ns and "::" not in name:
            sig = self.table.functions.get(f"{self.current_ns}::{name}")
        if sig is None:
            if name in scope:
                return XNamedType("any")
            if base in NATIVES:
                return XNamedType(NATIVES[base].get("ret", "f64"))
            self.err(file, e.line, 0, "E202", f"undefined function '{name}'",
                     hint=f"try 'xpc search {base}' in the central registry")
            return None

        n_pos = len(e.args)
        n_named = len(e.named_args)
        n_params = len(sig.params)
        if n_pos + n_named != n_params:
            self.err(file, e.line, 0, "E209",
                     f"function '{name}' expects {n_params} argument(s), got {n_pos + n_named}",
                     hint="named args count too (task::spawn(fn, core: 0))")
        return sig.ret

    def infer_pipeline_call(self, call: Call, scope, file):
        for a in call.args:
            self.infer(a, scope, file)
        return None

    def all_module_syms(self, ns):
        return {n.split("::")[-1] for n in self.table.functions if n.startswith(f"{ns}::")}

    def infer_member(self, e: Member, ot, scope, file):
        if ot is not None and isinstance(ot, XNamedType) and ot.name in self.table.structs:
            fields = self.table.structs[ot.name]
            if e.name not in fields:
                self.err(file, e.line, 0, "E210",
                         f"struct '{ot.name}' has no field '{e.name}'",
                         hint=f"fields: {', '.join(fields)}")
                return None
            return fields[e.name]
        return XNamedType("any")

    def scan_vectorization(self, prog: Program):
        def walk(stmts):
            for st in stmts:
                if isinstance(st, ForStmt) and st.range_expr is not None:
                    for s2 in st.body:
                        if isinstance(s2, (Assign, VarDecl)) and isinstance(
                                getattr(s2, "value", None) or getattr(s2, "init", None), BinOp):
                            self.vector_candidates.append(
                                (st.line, "arithmetic loop → SIMD auto-vectorize"))
                if isinstance(st, (IfStmt, LoopStmt, ForStmt, ArenaBlock, UnsafeBlock)):
                    walk(getattr(st, "body", []) or [])
                    if isinstance(st, IfStmt):
                        for (_, b) in st.branches:
                            walk(b)
                        if st.else_body:
                            walk(st.else_body)

        for d in prog.decls:
            if isinstance(d, FnDecl):
                walk(d.body)
from __future__ import annotations

import time
import os

from .parser import parse_program
from .astnodes import Program, FnDecl, ConstDecl, StructDecl, EnumDecl, RegisterDecl, UseStmt
from .typesys import Checker, ProgramTable
from .xir import Codegen, serialize_xbc, deserialize_xbc, XIRFunc
from .resolver import resolve_module, module_name
from .errors import XPlusError, XSyntaxError, XTypeError
from .vm import XVM
from . import VERSION, CTFE_STEP_LIMIT


class CompileResult:
    def __init__(self):
        self.programs = []
        self.table: ProgramTable = ProgramTable()
        self.funcs = {}
        self.consts_meta = {}
        self.structs = {}
        self.enums = {}
        self.registers = {}
        self.errors = []
        self.warnings = []
        self.vector_candidates = []
        self.compile_ms = 0.0
        self.instr_count = 0
        self.xbc = None
        self.meta = {}

    @property
    def ok(self):
        return not self.errors

def _assign_namespace(prog: Program, ns: str):
    for d in prog.decls:
        if isinstance(d, (FnDecl, ConstDecl, StructDecl, EnumDecl)):
            d.namespace = ns


def compile_source(source: str, filename: str = "main.xp", project_dir: str = ".",
                   fetch_imports: bool = True) -> CompileResult:
    t0 = time.perf_counter()
    res = CompileResult()
    res.meta = {"file": filename}
    try:
        main_prog = parse_program(source, filename)
    except XSyntaxError as e:
        res.errors.append(e)
        res.compile_ms = (time.perf_counter() - t0) * 1000
        return res
    res.programs.append(main_prog)
    modules = {}
    queue = [module_name(u.path) for u in main_prog.uses]
    visited = set()
    missing = []
    while queue:
        mod = queue.pop(0)
        if mod in visited:
            continue
        visited.add(mod)
        if fetch_imports:
            r = resolve_module(mod, project_dir)
            if r is None:
                missing.append(mod)
                continue
            try:
                lp = parse_program(r.source, os.path.basename(r.file))
            except XSyntaxError as e:
                res.errors.append(XTypeError(
                    f"library '{mod}' failed to parse: {e.message}",
                    file=e.file, line=e.line, hint="report this to the library maintainer"))
                continue
            _assign_namespace(lp, mod)
            modules[mod] = lp
            res.programs.append(lp)
            for u in lp.uses:
                queue.append(module_name(u.path))

    for mod in missing:
        first_use = next((u for u in main_prog.uses if module_name(u.path) == mod), None)
        res.errors.append(XTypeError(
            f"unknown module '{mod}'",
            file=filename, line=first_use.line if first_use else 1,
            hint=f"run 'xpc install {mod.split('::')[0]}' — the resolver searches: "
                 f"proyek lokal → ~/.xpc/libs → stdlib → GitHub X-Ples/Liblary"))

    table = res.table
    for mod, lp in modules.items():
        pub = set()
        for d in lp.decls:
            if isinstance(d, FnDecl) and d.is_pub:
                pub.add(d.name)
            elif isinstance(d, ConstDecl) and getattr(d, "is_pub", False):
                pub.add(d.name)
        table.modules[mod] = pub
    checker = Checker(table)
    all_progs = [main_prog] + [modules[m] for m in modules]
    for prog in all_progs:
        for d in prog.decls:
            checker.collect_decl(d, prog)
    for prog in all_progs:
        checker.current_ns = None
        for d in prog.decls:
            if getattr(d, "namespace", None):
                checker.current_ns = d.namespace
                break
        try:
            checker.check(prog)
        except RecursionError:
            res.errors.append(XTypeError("type checker stack overflow",
                                         file=prog.file))
    res.errors.extend(checker.errors)
    res.warnings.extend(checker.warnings)
    res.vector_candidates = checker.vector_candidates
    if not res.ok:
        res.compile_ms = (time.perf_counter() - t0) * 1000
        return res

    all_funcs = {}
    consts_meta = {}
    structs, enums, registers = {}, {}, {}
    try:
        cg_main = Codegen(table)
        ir = cg_main.gen_module(main_prog)
        all_funcs.update(ir["funcs"])
        consts_meta.update(ir["consts"])
        structs.update(ir["structs"])
        enums.update(ir["enums"])
        registers.update(ir["registers"])
        for mod, lp in modules.items():
            cg = Codegen(table, namespace=mod)
            ir = cg.gen_module(lp)
            all_funcs.update(ir["funcs"])
            for k, v in ir["consts"].items():
                consts_meta[k] = v
            structs.update(ir["structs"])
            enums.update(ir["enums"])
    except XSyntaxError as e:
        res.errors.append(e)
        res.compile_ms = (time.perf_counter() - t0) * 1000
        return res
    norm = {}
    for name, f in all_funcs.items():
        if isinstance(f, XIRFunc):
            norm[name] = {"name": name, "params": f.params, "ret": str(f.ret) if f.ret else None,
                          "instrs": [list(i) for i in f.instrs], "is_entry": f.is_entry}
        else:
            norm[name] = f
    all_funcs = norm
    res.funcs = all_funcs
    res.consts_meta = consts_meta
    res.structs = structs
    res.enums = enums
    res.registers = registers
    res.instr_count = sum(len(f["instrs"]) for f in all_funcs.values())
    res.compile_ms = (time.perf_counter() - t0) * 1000
    res.meta.update({
        "xpc_version": VERSION,
        "target": "native",
        "fast_math": True,
        "opt_level": 3,
        "profile": "release",
        "build_timestamp": int(time.time()),
        "structs": structs,
        "enums": enums,
        "registers": registers,
    })
    res.xbc = serialize_xbc(
        {"funcs": all_funcs, "consts": consts_meta, "structs": structs,
         "enums": enums, "registers": registers},
        res.meta)
    return res

def make_vm(res: CompileResult, out=None, err=None, board=False) -> XVM:
    vm = XVM(res.funcs, res.consts_meta, meta=res.meta, out=out, err=err, board=board)
    vm.load_consts()
    return vm

def run_source(source: str, filename: str = "main.xp", project_dir: str = ".",
               out=None, err=None, board=False, max_steps=None):
    res = compile_source(source, filename, project_dir)
    if not res.ok:
        raise res.errors[0]
    vm = make_vm(res, out=out, err=err, board=board)
    if max_steps:
        vm.max_steps = max_steps
    code = vm.run_main()
    return code, res, vm


def load_xbc_file(path: str):
    with open(path, "rb") as f:
        return deserialize_xbc(f.read())


def xbc_to_vm(data: bytes, out=None, err=None, board=False) -> XVM:
    header, body = deserialize_xbc(data)
    vm = XVM(body["funcs"], body["consts"], meta=header, out=out, err=err, board=board)
    vm.load_consts()
    return vm
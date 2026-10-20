from __future__ import annotations

import sys

from . import VERSION
from .compiler import compile_source, make_vm
from .errors import XPlusError
from .builtins import simd_target_name
from .tokens import T

HEADER = f"""X+ REPL v{VERSION} — JIT in RAM (Cranelift-free preview)
Target : {simd_target_name()} | Fast-Math: Enabled (permanent)
Type   : degrees := 45.0; (degrees * @pi / 180.0) |> math::sin
Exit   : .exit / .quit / Ctrl+C | Reset session: .reset | Help: .help
"""


class ReplSession:
    def __init__(self):
        self.decls = []
        self.var_lines = []
        self.vm = None

    def build_source(self, last_expr=None):
        body = "\n".join("    " + l for l in self.var_lines)
        if last_expr:
            body += f"\n    show(sys::out, \"= {{{'{'}_repl{'}'}}}\\n\"); _repl := {last_expr};"
            body = "\n".join("    " + l for l in self.var_lines)
            body += f"\n    _repl := {last_expr};"
            body += f"\n    show(sys::out, \"= {{_repl}}\\n\");"
        top = "\n".join(self.decls)
        src = ""
        if top:
            src += top + "\n\n"
        src += "using = <<math.xp>>\n\nmain => int32 {\n"
        src += body if body else "    yield 0;"
        src += "\n    yield 0;\n}\n"
        return src

    def eval(self, text) -> str:
        import io
        buf = io.StringIO()
        stripped = text.strip().rstrip(";").strip()
        is_top = (stripped.startswith("fn ") or stripped.startswith("pub ")
                  or stripped.startswith("const ") or stripped.startswith("struct ")
                  or stripped.startswith("enum "))

        # a plain statement?
        last_expr = None
        is_expr_like = not is_top and not stripped.endswith("{") and ";" not in stripped
        if is_expr_like and stripped and not any(stripped.startswith(k) for k in
                ("show", "if", "loop", "for", "match", "guard", "arena", "unsafe",
                 "return", "break", "continue", "parfor", "yield", "@")):
            try:
                from .parser import Parser as _RP
                _rp = _RP(stripped, "repl_probe.xp")
                _rp.parse_expr(allow_struct=True, no_range=True)
                if _rp.peek().type == T.EOF:
                    last_expr = stripped
            except Exception:
                last_expr = None
        if last_expr is None:
            if stripped and not stripped.endswith(";"):
                stripped += ";"
            self.var_lines.append(stripped)

        if is_top:
            self.decls.append(stripped)
            self.var_lines = self.var_lines

        src = self.build_source(last_expr)
        res = compile_source(src, "repl.xp")
        if not res.ok:
            err = res.errors[0]
            if last_expr is None and self.var_lines:
                self.var_lines.pop()
            if is_top and self.decls:
                self.decls.pop()
            return err.render()
        self.vm = make_vm(res, out=buf)
        try:
            self.vm.run_main()
        except XPlusError as e:
            if last_expr is None and self.var_lines:
                self.var_lines.pop()
            if is_top and self.decls:
                self.decls.pop()
            return e.render()
        return buf.getvalue()


def _is_pure_expr(cand: str) -> bool:
    try:
        from .parser import Parser as _RP
        from .tokens import T as _T
        rp = _RP(cand, "repl_probe.xp")
        rp.parse_expr(allow_struct=True, no_range=True)
        return rp.peek().type == _T.EOF
    except Exception:
        return False


def run_repl():
    print(HEADER)
    session = ReplSession()
    buf = ""
    while True:
        try:
            prompt = "x+> " if not buf else "...> "
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\n" + "bye — zero-waste: all REPL builds dropped from RAM. ✓")
            break
        cmd = line.strip()
        if cmd in (".exit", ".quit", "exit", "quit"):
            print("bye — zero-waste: all REPL builds dropped from RAM. ✓")
            break
        if cmd == ".reset":
            session = ReplSession()
            buf = ""
            print("session reset. ✓")
            continue
        if cmd == ".help":
            print("Commands: .reset, .exit | Examples:")
            print("  degrees := 45.0;")
            print("  (degrees * @pi / 180.0) |> math::sin")
            print("  for i in 0..3 { show(sys::out, \"i={i}\\n\"); };")
            continue
        if cmd.startswith("."):
            print(f"unknown command: {cmd}")
            continue

        buf += line + "\n"
        balanced = buf.count("{") == buf.count("}")
        ends = buf.rstrip().endswith(";") or buf.rstrip().endswith("}")
        if not (balanced and ends):
            cand = buf.strip().rstrip(";").strip()
            if not (balanced and cand and _is_pure_expr(cand)):
                continue

        src_in = buf.strip()
        buf = ""
        out = session.eval(src_in)
        if out:
            sys.stdout.write(out)
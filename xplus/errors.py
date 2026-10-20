from __future__ import annotations


class XPlusError(Exception):

    kind = "Error"
    code = "E000"

    def __init__(self, message, file=None, line=None, col=None, hint=None, source=None, code=None):
        super().__init__(message)
        self.message = message
        self.file = file or "<main.xp>"
        self.line = line
        self.col = col
        self.hint = hint
        self.source = source
        if code:
            self.code = code

    # ------------------------------------------------------------------
    def render(self, color=True):
        out = []
        red = lambda s: f"\033[31;1m{s}\033[0m" if color else s
        bold = lambda s: f"\033[1m{s}\033[0m" if color else s
        dim = lambda s: f"\033[2m{s}\033[0m" if color else s
        cyan = lambda s: f"\033[36m{s}\033[0m" if color else s

        out.append(red(self.message))
        if self.line is not None:
            loc = f" --> {self.file}:{self.line}"
            if self.col:
                loc += f":{self.col}"
            out.append(" " + cyan(loc))
            out.append("  " + dim("|"))

            line_text = self._get_line()
            if line_text is not None:
                num = str(self.line)
                pad = " " * max(len(num), 2)
                out.append(f" {num} | {line_text}")
                if self.col and self.col > 0:
                    caret_col = min(self.col, len(line_text) + 1)
                    marker = "^" + "-" * max(len(self.message), 8)
                    out.append(f" {pad} | " + " " * (caret_col - 1) + red(marker))
        if self.hint:
            out.append("  " + dim("= ") + self.hint)
        return "\n".join(out)

    def _get_line(self):
        if self.source:
            lines = self.source.splitlines()
            if 1 <= self.line <= len(lines):
                return lines[self.line - 1]
        return None

    def __str__(self):
        return self.render(color=False)


class XSyntaxError(XPlusError):
    kind = "Syntax Error"
    code = "E1xx"


class XTypeError(XPlusError):
    kind = "Type Error"
    code = "E2xx"


class XRuntimeError(XPlusError):
    kind = "Runtime Panic"
    code = "E3xx"


class XCompileAbort(XPlusError):
    kind = "Compiler Abort"
    code = "E9xx"


class XWarning:
    def __init__(self, code, message, file=None, line=None, col=None, hint=None):
        self.code = code
        self.message = message
        self.file = file
        self.line = line
        self.col = col
        self.hint = hint

    def render(self, color=True):
        yellow = lambda s: f"\033[33;1m{s}\033[0m" if color else s
        dim = lambda s: f"\033[2m{s}\033[0m" if color else s
        loc = f" --> {self.file}:{self.line}" if self.line else ""
        s = yellow(f"Warning: {self.code} {self.message}")
        if loc:
            s += "\n " + dim(loc)
        if self.hint:
            s += "\n  " + dim("= ") + self.hint
        return s

ERROR_CATALOG = {
    "E101": "unexpected character",
    "E102": "unterminated string literal",
    "E103": "invalid number literal",
    "E104": "unterminated import path <<...>>",
    "E110": "unexpected token",
    "E111": "expected token",
    "E112": "unknown statement / declaration",
    "E120": "unknown System Magic Variable (SMV)",
    "E201": "undefined variable",
    "E202": "undefined function",
    "E203": "unknown module / library (suggest: xpc install <pkg>)",
    "E204": "tensor dimension mismatch",
    "E205": "access to private (non-pub) symbol of module",
    "E206": "type mismatch",
    "E207": "invalid vector length literal",
    "E208": "assignment to immutable const / global",
    "E209": "arity mismatch in function call",
    "E210": "unknown struct field",
    "E211": "unknown enum variant",
    "E212": "non-exhaustive match without wildcard",
    "E220": "CTFE step limit exceeded (possible infinite loop in comptime)",
    "E221": "CTFE purity violation (I/O or side-effect in comptime)",
    "E301": "integer overflow (checked-by-default; use +% / -% / *% to wrap)",
    "E302": "division by zero (use guard)",
    "E303": "index out of bounds",
    "E304": "match fell through (no arm matched)",
    "E305": "propagated Err without Result return type",
    "E306": "cast overflow",
}
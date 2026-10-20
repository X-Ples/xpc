from __future__ import annotations

from .tokens import T, Token, KEYWORDS
from .errors import XSyntaxError

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", '"': '"', "\\": "\\", "'": "'"}


class Lexer:
    def __init__(self, source: str, filename: str = "main.xp"):
        self.src = source
        self.file = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def error(self, code, msg, line=None, col=None, hint=None):
        raise XSyntaxError(msg, file=self.file, line=line or self.line,
                           col=col or self.col, hint=hint, source=self.src)

    def _peek(self, off=0):
        p = self.pos + off
        return self.src[p] if p < len(self.src) else ""

    def _advance(self, n=1):
        for _ in range(n):
            if self.pos < len(self.src):
                if self.src[self.pos] == "\n":
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1

    def _add(self, ttype, value, line=None, col=None):
        self.tokens.append(Token(ttype, value, line if line is not None else self.line,
                                 col if col is not None else self.col))

    def tokenize(self) -> list[Token]:
        while self.pos < len(self.src):
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
                continue
            if ch == "/" and self._peek(1) == "/":
                while self.pos < len(self.src) and self._peek() != "\n":
                    self._advance()
                continue
            if ch == "/" and self._peek(1) == "*":
                sl, sc = self.line, self.col
                self._advance(2)
                while self.pos < len(self.src) and not (self._peek() == "*" and self._peek(1) == "/"):
                    self._advance()
                if self.pos >= len(self.src):
                    self.error("E104", "unterminated block comment", sl, sc)
                self._advance(2)
                continue
            if ch == "<" and self._peek(1) == "<":
                sl, sc = self.line, self.col
                self._advance(2)
                start = self.pos
                while self.pos < len(self.src) and not (self._peek() == ">" and self._peek(1) == ">"):
                    self._advance()
                if self.pos >= len(self.src):
                    self.error("E104", "unterminated import path '<<...>>'", sl, sc,
                               hint="close it with '>>'")
                path = self.src[start:self.pos].strip()
                if not path:
                    self.error("E104", "empty import path", sl, sc)
                self._advance(2)
                self._add(T.IMPORT_PATH, path, sl, sc)
                continue
            if ch == '"':
                self._string()
                continue
            if ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                self._number()
                continue
            if ch.isalpha() or ch == "_":
                sl, sc = self.line, self.col
                start = self.pos
                while self._peek().isalnum() or self._peek() == "_":
                    self._advance()
                word = self.src[start:self.pos]

                if word in KEYWORDS:
                    self._add(KEYWORDS[word], word, sl, sc)
                else:
                    self._add(T.IDENT, word, sl, sc)
                continue
            if ch == ":" and self._peek(1) == ":":
                self._add(T.COLONCOLON, "::")
                self._advance(2); continue
            if ch == ":" and self._peek(1) == "=":
                self._add(T.VARDECL, ":=")
                self._advance(2); continue
            if ch == "=" and self._peek(1) == "=":
                self._add(T.EQ, "==")
                self._advance(2); continue
            if ch == "=" and self._peek(1) == ">":
                self._add(T.ARROW, "=>")
                self._advance(2); continue
            if ch == "-" and self._peek(1) == ">":
                self._add(T.RARROW, "->")
                self._advance(2); continue
            if ch == "!" and self._peek(1) == "=":
                self._add(T.NEQ, "!=")
                self._advance(2); continue
            if ch == "<" and self._peek(1) == "=":
                self._add(T.LE, "<=")
                self._advance(2); continue
            if ch == ">" and self._peek(1) == "=":
                self._add(T.GE, ">=")
                self._advance(2); continue
            if ch == "<" and self._peek(1) == "<":
                self._add(T.SHL, "<<")
                self._advance(2); continue
            if ch == ">" and self._peek(1) == ">":
                self._add(T.SHR, ">>")
                self._advance(2); continue
            if ch == "&" and self._peek(1) == "&":
                self._add(T.AND, "&&")
                self._advance(2); continue
            if ch == "|" and self._peek(1) == ">":
                self._add(T.PIPE, "|>")
                self._advance(2); continue
            if ch == "." and self._peek(1) == "." and self._peek(2) == "=":
                self._add(T.DOTDOTEQ, "..=")
                self._advance(3); continue
            if ch == "." and self._peek(1) == ".":
                self._add(T.DOTDOT, "..")
                self._advance(2); continue
            if ch == "." and self._peek(1) == "x" \
                    and not (self._peek(2).isalnum() or self._peek(2) == "_") \
                    and self.pos > 0 and self.src[self.pos - 1] in " \t":
                self._add(T.DOTX, ".x")
                self._advance(2); continue
            if ch == "+" and self._peek(1) == "%":
                self._add(T.WRAP_ADD, "+%")
                self._advance(2); continue
            if ch == "-" and self._peek(1) == "%":
                self._add(T.WRAP_SUB, "-%")
                self._advance(2); continue
            if ch == "*" and self._peek(1) == "%":
                self._add(T.WRAP_MUL, "*%")
                self._advance(2); continue

            single = {
                "=": T.ASSIGN, "+": T.PLUS, "-": T.MINUS, "*": T.STAR,
                "/": T.SLASH, "%": T.PERCENT, "^": T.CARET, "!": T.NOT,
                "<": T.LT, ">": T.GT, "(": T.LPAREN, ")": T.RPAREN,
                "{": T.LBRACE, "}": T.RBRACE, "[": T.LBRACKET, "]": T.RBRACKET,
                ",": T.COMMA, ";": T.SEMI, ":": T.COLON, "?": T.QUESTION,
                "@": T.AT, ".": T.DOT, "|": T.PIPE,
            }
            if ch in single:
                self._add(single[ch], ch)
                self._advance()
                continue

            self.error("E101", f"unexpected character {ch!r}",
                       hint="karakter ini tidak dikenal lexer X+ v0.1")

        self._add(T.EOF, None)
        return self.tokens

    def _string(self):
        sl, sc = self.line, self.col
        self._advance()
        buf = []
        while True:
            if self.pos >= len(self.src):
                self.error("E102", "unterminated string literal", sl, sc)
            ch = self._peek()
            if ch == '"':
                self._advance()
                break
            if ch == "\\":
                self._advance()
                esc = self._peek()
                if esc not in ESCAPES:
                    self.error("E102", f"unknown escape '\\{esc}'", hint="use \\n \\t \\r \\0 \\\" \\\\")
                buf.append(ESCAPES[esc])
                self._advance()
                continue
            buf.append(ch)
            self._advance()
        self._add(T.STRING, "".join(buf), sl, sc)

    def _number(self):
        sl, sc = self.line, self.col
        start = self.pos
        if self._peek() == "0" and self._peek(1) in "xX":
            self._advance(2)
            hx = self.pos
            while self._peek() in "0123456789abcdefABCDEF_":
                self._advance()
            txt = self.src[hx:self.pos].replace("_", "")
            if not txt:
                self.error("E103", "invalid hex literal", sl, sc)
            self._add(T.INT, int(txt, 16), sl, sc)
            return
        if self._peek() == "0" and self._peek(1) in "bB":
            self._advance(2)
            bx = self.pos
            while self._peek() in "01_":
                self._advance()
            txt = self.src[bx:self.pos].replace("_", "")
            if not txt:
                self.error("E103", "invalid binary literal", sl, sc)
            self._add(T.INT, int(txt, 2), sl, sc)
            return

        is_float = False
        while self._peek().isdigit() or self._peek() == "_":
            self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while self._peek().isdigit() or self._peek() == "_":
                self._advance()
        if self._peek() in "eE" and (self._peek(1).isdigit() or
                                     (self._peek(1) in "+-" and self._peek(2).isdigit())):
            is_float = True
            self._advance()
            if self._peek() in "+-":
                self._advance()
            while self._peek().isdigit():
                self._advance()
        txt = self.src[start:self.pos].replace("_", "")
        try:
            if is_float:
                self._add(T.FLOAT, float(txt), sl, sc)
            else:
                self._add(T.INT, int(txt), sl, sc)
        except ValueError:
            self.error("E103", f"invalid number literal '{txt}'", sl, sc)


def tokenize(source: str, filename: str = "main.xp") -> list[Token]:
    return Lexer(source, filename).tokenize()
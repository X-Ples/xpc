from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class T(Enum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IDENT = auto()
    IMPORT_PATH = auto()
    USING = auto()
    CONST = auto()
    STRUCT = auto()
    ENUM = auto()
    FN = auto()
    PUB = auto()
    EXPORT = auto()
    RETURN = auto()
    YIELD = auto()
    IF = auto()
    ELSE = auto()
    LOOP = auto()
    FOR = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    MATCH = auto()
    GUARD = auto()
    ARENA = auto()
    COMPTIME = auto()
    ASYNC = auto()
    AWAIT = auto()
    PARFOR = auto()
    UNSAFE = auto()
    REGISTER = auto()
    AS = auto()
    TRUE = auto()
    FALSE = auto()
    MAIN_ARROW_IDENT = auto()
    ASSIGN = auto()           # =
    VARDECL = auto()          # :=
    ARROW = auto()            # =>
    RARROW = auto()           # ->
    EQ = auto()               # ==
    NEQ = auto()              # !=
    LE = auto()               # <=
    GE = auto()               # >=
    LT = auto()               # <
    GT = auto()               # >
    PLUS = auto()             # +
    MINUS = auto()            # -
    STAR = auto()             # *
    SLASH = auto()            # /
    PERCENT = auto()          # %
    CARET = auto()            # ^
    WRAP_ADD = auto()         # +%
    WRAP_SUB = auto()         # -%
    WRAP_MUL = auto()         # *%
    AND = auto()              # &&
    OR = auto()               # ||
    NOT = auto()              # !
    PIPE = auto()             # |>
    DOT = auto()              # .
    DOTDOT = auto()           # ..
    DOTDOTEQ = auto()         # ..=
    DOTX = auto()             # .x
    COMMA = auto()            # ,
    SEMI = auto()             # ;
    COLON = auto()            # :
    COLONCOLON = auto()       # ::
    QUESTION = auto()         # ?
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    AT = auto()               # @ (SMV / atribut / makro)
    BANG_LBRACE = auto()
    BANG = auto()
    SHL = auto()              # <<
    SHR = auto()              # >>
    IDENT_X = auto()
    EOF = auto()


KEYWORDS = {
    "using": T.USING, "const": T.CONST, "struct": T.STRUCT, "enum": T.ENUM,
    "fn": T.FN, "pub": T.PUB, "export": T.EXPORT, "return": T.RETURN,
    "yield": T.YIELD, "if": T.IF, "else": T.ELSE, "loop": T.LOOP,
    "for": T.FOR, "in": T.IN, "break": T.BREAK, "continue": T.CONTINUE,
    "match": T.MATCH, "guard": T.GUARD, "arena": T.ARENA, "comptime": T.COMPTIME,
    "async": T.ASYNC, "await": T.AWAIT, "parfor": T.PARFOR, "unsafe": T.UNSAFE,
    "register": T.REGISTER, "as": T.AS, "true": T.TRUE, "false": T.FALSE,
}


@dataclass
class Token:
    type: T
    value: object
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"
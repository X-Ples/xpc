from __future__ import annotations

from .tokens import T, Token
from .lexer import tokenize
from .astnodes import *
from .astnodes import Program, UseStmt, Param
from .errors import XSyntaxError

PRIMITIVE_TYPES = {
    "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
    "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
    "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64",
    "usize": "usize", "f32": "f32", "f64": "f64",
    "bool": "bool", "string": "string", "str": "string", "void": "void",
    "rational": "rational", "decimal256": "decimal256",
    "any": "any",
}

ATTR_BLOCK_NAMES = {"tensor_core", "gpu", "deterministic", "resilient", "enclave"}

COMPOUND_OPS = {"+=": "ADD_ASSIGN", "-=": "SUB_ASSIGN", "*=": "MUL_ASSIGN",
                "/=": "DIV_ASSIGN"}


class Parser:
    def __init__(self, source: str, filename: str = "main.xp"):
        self.file = filename
        self.src = source
        self.toks = tokenize(source, filename)
        self.i = 0

    def peek(self, off=0) -> Token:
        j = min(self.i + off, len(self.toks) - 1)
        return self.toks[j]

    def next(self) -> Token:
        tok = self.toks[self.i]
        if self.i < len(self.toks) - 1:
            self.i += 1
        return tok

    def at(self, *types) -> bool:
        return self.peek().type in types

    def expect(self, ttype, what="") -> Token:
        tok = self.peek()
        if tok.type != ttype:
            self.err("E111", f"expected {what or ttype.name}, found {self.tok_desc(tok)}",
                     tok, hint=self.context_hint())
        return self.next()

    def tok_desc(self, tok):
        if tok.type == T.EOF:
            return "end of file"
        return f"'{tok.value}'"

    def err(self, code, msg, tok=None, hint=None):
        tok = tok or self.peek()
        raise XSyntaxError(msg, file=self.file, line=tok.line, col=tok.col,
                           hint=hint, source=self.src)

    def context_hint(self):
        return None

    def parse(self) -> Program:
        prog = Program(file=self.file)
        pending_attrs: list[Attribute] = []
        while not self.at(T.EOF):
            tok = self.peek()
            if tok.type == T.USING:
                prog.uses.append(self.parse_using())
            elif tok.type == T.AT and self._is_statement_attr_block():
                prog.decls.append(self.parse_statement_attr_block())
            elif tok.type == T.AT:
                pending_attrs.append(self.parse_attribute())
            elif tok.type == T.CONST or (tok.type == T.PUB and self.peek(1).type == T.CONST):
                is_pub = False
                if tok.type == T.PUB:
                    self.next()
                    is_pub = True
                d = self.parse_const()
                d.is_pub = is_pub
                prog.decls.append(d)
            elif tok.type == T.STRUCT:
                d = self.parse_struct()
                d.attrs = pending_attrs; pending_attrs = []
                prog.decls.append(d)
            elif tok.type == T.ENUM:
                d = self.parse_enum()
                d.attrs = pending_attrs; pending_attrs = []
                prog.decls.append(d)
            elif tok.type == T.REGISTER:
                d = self.parse_register()
                d.attrs = pending_attrs; pending_attrs = []
                prog.decls.append(d)
            elif tok.type in (T.PUB, T.EXPORT, T.FN):
                d = self.parse_fn(pending_attrs)
                pending_attrs = []
                prog.decls.append(d)
            elif tok.type == T.IDENT and self.peek(1).type == T.LPAREN:
                d = self.parse_fn(pending_attrs, implicit=True)
                pending_attrs = []
                prog.decls.append(d)
            elif tok.type == T.IDENT and tok.value == "main" and self.peek(1).type == T.ARROW:
                d = self.parse_entry(pending_attrs)
                pending_attrs = []
                prog.decls.append(d)
            elif tok.type == T.SEMI:
                self.next()
            else:
                self.err("E112", f"unexpected {self.tok_desc(tok)} at top level",
                         tok, hint="top-level: using / const / struct / enum / fn / register / main =>")
        prog.meta["attrs"] = pending_attrs
        return prog

    def _is_statement_attr_block(self) -> bool:
        name = self.peek(1)
        if name.type != T.IDENT:
            return False
        if name.value in ("show",):
            return True
        if name.value in ATTR_BLOCK_NAMES and self.peek(2).type == T.LBRACE:
            return True
        return False

    def parse_using(self) -> UseStmt:
        self.expect(T.USING, "'using'")
        self.expect(T.ASSIGN, "'='")
        tok = self.expect(T.IMPORT_PATH, "import path <<math.xp>>")
        self._opt_semi()
        return UseStmt(path=tok.value, line=tok.line)

    def parse_attribute(self) -> Attribute:
        at = self.expect(T.AT, "'@'")
        name = self.expect(T.IDENT, "attribute name")
        args = None
        if self.at(T.LPAREN):
            args = self.parse_raw_parens()
        return Attribute(name=name.value, args=args, line=at.line)

    def parse_raw_parens(self):
        self.expect(T.LPAREN, "'('")
        depth = 1
        parts = []
        while depth > 0 and not self.at(T.EOF):
            tok = self.next()
            if tok.type == T.LPAREN:
                depth += 1
            elif tok.type == T.RPAREN:
                depth -= 1
                if depth == 0:
                    break
            parts.append(str(tok.value) if tok.value is not None else "")
        return " ".join(p for p in parts if p)

    def parse_const(self) -> ConstDecl:
        self.expect(T.CONST, "'const'")
        name = self.expect(T.IDENT, "constant name")
        ctype = None
        if self.at(T.COLON):
            self.next()
            ctype = self.parse_type()
        self.expect(T.ASSIGN, "'='")
        comptime = False
        if self.at(T.COMPTIME) and self.peek(1).type == T.LBRACE:
            self.next()
            comptime = True
            body = self.parse_block_stmts()
            value = ComptimeBlock(body=body, line=name.line)
        else:
            value = self.parse_expr(allow_struct=True)
        self._opt_semi()
        return ConstDecl(name=name.value, type=ctype, value=value,
                         comptime=comptime, line=name.line)

    def parse_struct(self) -> StructDecl:
        self.expect(T.STRUCT, "'struct'")
        name = self.expect(T.IDENT, "struct name")
        self.expect(T.LBRACE, "'{'")
        fields = []
        while not self.at(T.RBRACE):
            fname = self.expect(T.IDENT, "field name")
            self.expect(T.COLON, "':'")
            ftype = self.parse_type()
            fields.append(Param(fname.value, ftype, fname.line))
            if self.at(T.COMMA):
                self.next()
            elif not self.at(T.RBRACE):
                self.expect(T.SEMI, "';' or ','")
        self.expect(T.RBRACE, "'}'")
        self._opt_semi()
        return StructDecl(name=name.value, fields=fields, line=name.line)

    def parse_enum(self) -> EnumDecl:
        self.expect(T.ENUM, "'enum'")
        name = self.expect(T.IDENT, "enum name")
        self.expect(T.LBRACE, "'{'")
        variants = []
        while not self.at(T.RBRACE):
            v = self.expect(T.IDENT, "variant name")
            variants.append(v.value)
            if self.at(T.COMMA):
                self.next()
        self.expect(T.RBRACE, "'}'")
        self._opt_semi()
        return EnumDecl(name=name.value, variants=variants, line=name.line)

    def parse_register(self) -> RegisterDecl:
        self.expect(T.REGISTER, "'register'")
        name = self.expect(T.IDENT, "register name")
        self.expect(T.COLON, "':'")
        rtype = self.parse_type()
        self.expect(T.AT, "'@'")
        addr = self.expect(T.INT, "address literal")
        self._opt_semi()
        return RegisterDecl(name.value, rtype, addr.value, name.line)

    def parse_fn(self, attrs, implicit=False) -> FnDecl:
        is_pub = False
        export_c = False
        if self.at(T.EXPORT):
            self.next()
            if self.at(T.STRING):
                s = self.next()
                export_c = (s.value == "C")
            if self.at(T.FN):
                self.next()
        elif self.at(T.PUB):
            self.next()
            is_pub = True
            if self.at(T.FN):
                self.next()
        elif self.at(T.FN):
            self.next()

        name = self.expect(T.IDENT, "function name")
        params = []
        if self.at(T.LPAREN):
            params = self.parse_params()
        ret_type = None
        if self.at(T.RARROW) or self.at(T.ARROW):
            self.next()
            ret_type = self.parse_type()
        body = self.parse_block_stmts()
        return FnDecl(name=name.value, params=params, ret_type=ret_type, body=body,
                      attrs=attrs, is_pub=is_pub, export_c=export_c, line=name.line)

    def parse_entry(self, attrs) -> FnDecl:
        name = self.expect(T.IDENT, "'main'")
        self.expect(T.ARROW, "'=>'")
        ret_type = self.parse_type()
        body = self.parse_block_stmts()
        has_entry = any(a.name == "entry" for a in attrs)
        if not has_entry:
            attrs = list(attrs) + [Attribute(name="entry", line=name.line)]
        return FnDecl(name="main", params=[], ret_type=ret_type, body=body,
                      attrs=attrs, is_entry=True, line=name.line)

    def parse_params(self) -> list[Param]:
        self.expect(T.LPAREN, "'('")
        params = []
        while not self.at(T.RPAREN):
            pname = self.expect(T.IDENT, "parameter name")
            ptype = None
            if self.at(T.COLON):
                self.next()
                ptype = self.parse_type()
            params.append(Param(pname.value, ptype, pname.line))
            if self.at(T.COMMA):
                self.next()
        self.expect(T.RPAREN, "')'")
        return params

    def parse_type(self):
        tok = self.peek()
        if tok.type == T.LBRACKET:
            self.next()
            inner = self.parse_type()
            self.expect(T.SEMI, "';' dalam tipe array [T; N]")
            size = self.expect(T.INT, "ukuran array")
            self.expect(T.RBRACKET, "']'")
            return XArrayType(inner, size.value)
        if tok.type != T.IDENT:
            self.err("E111", f"expected type, found {self.tok_desc(tok)}", tok)
        self.next()
        name = tok.value
        if name in PRIMITIVE_TYPES:
            base = XNamedType(PRIMITIVE_TYPES[name])
        elif name.startswith("vec") and name[3:].isdigit():
            base = XVecType(int(name[3:]))
        elif name.startswith("mat") and name[3:].isdigit():
            base = XMatType(int(name[3:]), int(name[3:]))
        elif name == "mat" or name == "tensor":
            dims = []
            if self.at(T.LBRACKET):
                self.next()
                while not self.at(T.RBRACKET):
                    if self.at(T.STAR):
                        self.next()
                        dims.append("*")
                    else:
                        dims.append(self.expect(T.INT, "dimensi").value)
                    if self.at(T.COMMA):
                        self.next()
                self.expect(T.RBRACKET, "']'")
            if name == "mat":
                base = XMatShapeType(dims)
            else:
                base = XTensorType(dims)
        else:
            base = XNamedType(name)
        if self.at(T.LT):
            self.next()
            args = [self.parse_type()]
            while self.at(T.COMMA):
                self.next()
                args.append(self.parse_type())
            self.expect(T.GT, "'>'")
            base = XGenericType(name, args)
        if self.at(T.QUESTION):
            self.next()
            base = XOptionalType(base)
        return base

    def parse_block_stmts(self) -> list:
        self.expect(T.LBRACE, "'{'")
        body = self.parse_stmts_until(T.RBRACE)
        self.expect(T.RBRACE, "'}'")
        return body

    def parse_stmts_until(self, *end_types) -> list:
        stmts = []
        while not self.at(*end_types) and not self.at(T.EOF):
            stmts.append(self.parse_stmt())
        return stmts

    def _opt_semi(self):
        if self.at(T.SEMI):
            self.next()

    def _block_like(self, stmt) -> bool:
        return isinstance(stmt, (IfStmt, LoopStmt, ForStmt, MatchStmt, ArenaBlock,
                                 UnsafeBlock, ShowBlock, TensorCoreBlock, AsmBlock,
                                 ComptimeBlock))

    def parse_stmt(self):
        tok = self.peek()

        if tok.type == T.SEMI:
            self.next()
            return ExprStmt(Literal(0, "int", tok.line), tok.line)

        if tok.type == T.AT:
            return self.parse_at_stmt()

        if tok.type == T.IF:
            s = self.parse_if(); self._opt_semi(); return s
        if tok.type == T.LOOP:
            s = self.parse_loop(); self._opt_semi(); return s
        if tok.type == T.FOR or tok.type == T.PARFOR:
            s = self.parse_for(); self._opt_semi(); return s
        if tok.type == T.BREAK:
            t = self.next(); self._opt_semi(); return BreakStmt(t.line)
        if tok.type == T.CONTINUE:
            t = self.next(); self._opt_semi(); return ContinueStmt(t.line)
        if tok.type in (T.RETURN, T.YIELD):
            self.next()
            value = None
            if not self.at(T.SEMI) and not self.at(T.RBRACE):
                value = self.parse_expr(allow_struct=True)
            self._opt_semi()
            return ReturnStmt(value, tok.line)
        if tok.type == T.GUARD:
            return self.parse_guard()
        if tok.type == T.MATCH:
            return self.parse_match()
        if tok.type == T.ARENA:
            t = self.next()
            return ArenaBlock(body=self.parse_block_stmts(), line=t.line)
        if tok.type == T.UNSAFE:
            t = self.next()
            return UnsafeBlock(body=self.parse_block_stmts(), line=t.line)
        if tok.type == T.COMPTIME:
            t = self.next()
            return ComptimeBlock(body=self.parse_block_stmts(), line=t.line)
        if tok.type == T.CONST:
            return self.parse_const_local()

        if tok.type == T.IDENT and tok.value == "asm" and self.peek(1).type == T.NOT \
                and self.peek(2).type == T.LBRACE:
            self.next(); self.next()
            body = self.parse_block_stmts()
            instrs = []
            for s in body:
                if isinstance(s, ExprStmt) and isinstance(s.expr, Literal) and s.expr.kind == "string":
                    instrs.append(s.expr.value)
            return AsmBlock(instrs=instrs, line=tok.line)

        if tok.type == T.IDENT and self.peek(1).type == T.COLON and self.peek(2).type != T.COLON:
            return self.parse_typed_decl()

        if tok.type == T.IDENT and self.peek(1).type == T.VARDECL:
            name = self.next(); self.next()
            value = self.parse_expr(allow_struct=True)
            self._expect_semi_or_end()
            return VarDecl(name.value, None, value, mutable=True, line=name.line)

        if tok.type == T.IDENT and self.peek(1).type in (T.ASSIGN, T.WRAP_ADD, T.WRAP_SUB, T.WRAP_MUL):
            return self.parse_assignment()
        if tok.type == T.IDENT and self.peek(1).type in (T.DOT, T.LBRACKET):
            save = self.i
            try:
                target = self.parse_postfix_expr()
                if self.at(T.ASSIGN, T.WRAP_ADD, T.WRAP_SUB, T.WRAP_MUL):
                    return self.parse_assignment_from(target, tok.line)
                self.i = save
            except XSyntaxError:
                self.i = save

        expr = self.parse_expr(allow_struct=False)
        self._expect_semi_or_end()
        return ExprStmt(expr, tok.line)

    def _expect_semi_or_end(self):
        if self.at(T.SEMI):
            self.next()
        elif self.at(T.RBRACE) or self.at(T.EOF):
            pass
        else:
            self.err("E111", f"expected ';', found {self.tok_desc(self.peek())}",
                     hint="setiap statement diakhiri ';'")

    def parse_const_local(self) -> ConstDecl:
        d = self.parse_const()
        return d

    def parse_typed_decl(self):
        name = self.expect(T.IDENT, "variable name")
        self.expect(T.COLON, "':'")
        vtype = self.parse_type()
        init = None
        if self.at(T.ASSIGN):
            self.next()
            init = self.parse_expr(allow_struct=True)
        self._expect_semi_or_end()
        return VarDecl(name.value, vtype, init, mutable=True, line=name.line)

    def parse_assignment(self):
        target_tok = self.peek()
        target = self.parse_postfix_expr()
        return self.parse_assignment_from(target, target_tok.line)

    def parse_assignment_from(self, target, tline):
        op_tok = self.next()
        op = op_tok.value
        value = self.parse_expr(allow_struct=True)
        self._expect_semi_or_end()
        return Assign(target, op, value, tline)

    def parse_at_stmt(self):
        at_tok = self.expect(T.AT, "'@'")
        name_tok = self.expect(T.IDENT, "macro name")

        if name_tok.value == "show":
            self.expect(T.LPAREN, "'('")
            chan = self.parse_channel()
            self.expect(T.RPAREN, "')'")
            self.expect(T.LBRACE, "'{'")
            fmt_tok = self.expect(T.STRING, "format string")
            self.expect(T.SEMI, "';' setelah format string")
            injections = []
            while not self.at(T.RBRACE):
                slot = self.expect(T.INT, "slot index (0, 1, ...)")
                tann = None
                if self.at(T.COLONCOLON):
                    self.next()
                    tann = self.parse_type()
                self.expect(T.ARROW, "'=>'")
                expr = self.parse_expr()
                injections.append((slot.value, expr, tann))
                self.expect(T.SEMI, "';'")
            self.expect(T.RBRACE, "'}'")
            self._opt_semi()
            return ShowBlock(channel=chan, fmt=fmt_tok.value, injections=injections,
                             line=at_tok.line)

        if name_tok.value in ATTR_BLOCK_NAMES and self.at(T.LBRACE):
            body = self.parse_block_stmts()
            self._opt_semi()
            return TensorCoreBlock(backend=name_tok.value, body=body, line=at_tok.line)

        args = None
        if self.at(T.LPAREN):
            args = self.parse_raw_parens()
        inner = self.parse_stmt()
        if isinstance(inner, FnDecl):
            inner.attrs.append(Attribute(name_tok.value, args, at_tok.line))
            return inner
        return inner

    def parse_channel(self) -> str:
        parts = [self.expect(T.IDENT, "channel (sys)").value]
        while self.at(T.COLONCOLON):
            self.next()
            parts.append(self.expect(T.IDENT, "channel segment").value)
        return "::".join(parts)

    def parse_if(self) -> IfStmt:
        tok = self.expect(T.IF, "'if'")
        cond = self.parse_expr(allow_struct=False)
        if self.at(T.ARROW):
            self.next()
            inner = self.parse_stmt()
            body = [inner]
            branches = [(cond, body)]
            else_body = None
            if self.at(T.ELSE):
                self.next()
                else_body = [self.parse_stmt()] if not self.at(T.IF) else [self.parse_if()]
            return IfStmt(branches, else_body, tok.line)
        body = self.parse_block_stmts()
        branches = [(cond, body)]
        else_body = None
        while self.at(T.ELSE):
            self.next()
            if self.at(T.IF):
                else_body = [self.parse_if()]
                break
            else_body = self.parse_block_stmts()
            break
        return IfStmt(branches, else_body, tok.line)

    def parse_loop(self) -> LoopStmt:
        tok = self.expect(T.LOOP, "'loop'")
        cond = None
        if not self.at(T.LBRACE):
            cond = self.parse_expr(allow_struct=False)
        body = self.parse_block_stmts()
        return LoopStmt(cond, body, tok.line)

    def parse_for(self) -> ForStmt:
        tok = self.next()
        var = self.expect(T.IDENT, "loop variable")
        self.expect(T.IN, "'in'")
        start = self.parse_expr(allow_struct=False, no_range=True)
        if self.at(T.DOTDOT) or self.at(T.DOTDOTEQ):
            inc = self.next().type == T.DOTDOTEQ
            end = self.parse_expr(allow_struct=False)
            rng = RangeExpr(start, end, inc, tok.line)
            body = self.parse_block_stmts()
            return ForStmt(var.value, range_expr=rng, body=body,
                           parallel=(tok.type == T.PARFOR), line=tok.line)
        iter_expr = start
        while self.at(T.PIPE):
            rhs = self.parse_expr_atom()
            iter_expr = Pipeline(iter_expr, rhs, tok.line)
        body = self.parse_block_stmts()
        return ForStmt(var.value, iter_expr=iter_expr, body=body,
                       parallel=(tok.type == T.PARFOR), line=tok.line)

    def parse_guard(self) -> GuardStmt:
        tok = self.expect(T.GUARD, "'guard'")
        cond = self.parse_expr(allow_struct=False)
        self.expect(T.ELSE, "'else'")
        body = self.parse_block_stmts()
        self._opt_semi()
        return GuardStmt(cond, body, tok.line)

    def parse_match(self) -> MatchStmt:
        tok = self.expect(T.MATCH, "'match'")
        subject = self.parse_expr(allow_struct=False)
        self.expect(T.LBRACE, "'{'")
        arms = []
        while not self.at(T.RBRACE):
            pat, binding = self.parse_pattern()
            self.expect(T.ARROW, "'=>'")
            if self.at(T.LBRACE):
                body = self.parse_block_stmts()
            else:
                e = self.parse_expr(allow_struct=True)
                body = [ExprStmt(e, tok.line)] if not isinstance(e, ReturnStmt) else [e]
                self._opt_semi()
            arms.append((pat, binding, body))
        self.expect(T.RBRACE, "'}'")
        self._opt_semi()
        return MatchStmt(subject, arms, tok.line)

    def parse_pattern(self):
        tok = self.peek()
        if tok.type == T.IDENT and tok.value == "_" :
            self.next()
            return "_", None
        if tok.type in (T.INT, T.FLOAT, T.STRING, T.TRUE, T.FALSE):
            self.next()
            val = {"true": True, "false": False}.get(str(tok.value), tok.value)
            kind = {"true": "bool", "false": "bool"}.get(str(tok.value),
                   "string" if tok.type == T.STRING else ("float" if tok.type == T.FLOAT else "int"))
            return Literal(val, kind, tok.line), None
        if tok.type == T.IDENT:
            name = tok.value
            self.next()
            if self.at(T.LPAREN):
                self.next()
                bind = self.expect(T.IDENT, "binding variable").value
                self.expect(T.RPAREN, "')'")
                return name, bind
            if self.at(T.COLONCOLON):
                self.next()
                variant = self.expect(T.IDENT, "variant").value
                return f"{name}::{variant}", None
            return name, None
        self.err("E111", f"invalid match pattern {self.tok_desc(tok)}", tok)

    def parse_expr(self, allow_struct=False, no_range=False):
        left = self.parse_alt(allow_struct)
        if self.at(T.PIPE):
            while self.at(T.PIPE):
                self.next()
                rhs = self.parse_pipeline_rhs()
                left = Pipeline(left, rhs, left.line)
        if not no_range and (self.at(T.DOTDOT) or self.at(T.DOTDOTEQ)):
            inc = self.next().type == T.DOTDOTEQ
            right = self.parse_alt(allow_struct)
            left = RangeExpr(left, right, inc, left.line)
        return left

    def parse_pipeline_rhs(self):
        tok = self.peek()
        if tok.type == T.IDENT:
            path = self.parse_path()
            if self.at(T.LPAREN):
                args, named = self.parse_call_args()
                return Call(path, args, named, tok.line)
            return Call(path, [], {}, tok.line)
        return self.parse_expr_atom()

    def parse_alt(self, allow_struct):
        left = self.parse_and(allow_struct)
        while self.at(T.OR):
            op = self.next()
            right = self.parse_and(allow_struct)
            left = BinOp("||", left, right, op.line)
        return left

    def parse_and(self, allow_struct):
        left = self.parse_equality(allow_struct)
        while self.at(T.AND):
            op = self.next()
            right = self.parse_equality(allow_struct)
            left = BinOp("&&", left, right, op.line)
        return left

    def parse_equality(self, allow_struct):
        left = self.parse_comparison(allow_struct)
        while self.at(T.EQ, T.NEQ):
            op = self.next()
            right = self.parse_comparison(allow_struct)
            left = BinOp(op.value, left, right, op.line)
        return left

    def parse_comparison(self, allow_struct):
        left = self.parse_additive(allow_struct)
        while self.at(T.LT, T.GT, T.LE, T.GE):
            op = self.next()
            right = self.parse_additive(allow_struct)
            left = BinOp(op.value, left, right, op.line)
        return left

    def parse_additive(self, allow_struct):
        left = self.parse_multiplicative(allow_struct)
        while self.at(T.PLUS, T.MINUS):
            op = self.next()
            right = self.parse_multiplicative(allow_struct)
            left = BinOp(op.value, left, right, op.line)
        return left

    def parse_multiplicative(self, allow_struct):
        left = self.parse_matmul(allow_struct)
        while self.at(T.STAR, T.SLASH, T.PERCENT):
            op = self.next()
            right = self.parse_matmul(allow_struct)
            left = BinOp(op.value, left, right, op.line)
        return left

    def parse_matmul(self, allow_struct):
        left = self.parse_power(allow_struct)
        while True:
            tok = self.peek()
            if tok.type == T.IDENT and tok.value == "x":
                opname = "x"
            elif tok.type == T.DOTX:
                opname = ".x"
            else:
                break
            self.next()
            right = self.parse_power(allow_struct)
            left = BinOp(opname, left, right, tok.line)
        return left

    def parse_power(self, allow_struct):
        left = self.parse_unary(allow_struct)
        if self.at(T.CARET):
            op = self.next()
            right = self.parse_power(allow_struct)
            left = BinOp("^", left, right, op.line)
        return left

    def parse_unary(self, allow_struct):
        if self.at(T.MINUS):
            op = self.next()
            operand = self.parse_unary(allow_struct)
            return UnOp("-", operand, op.line)
        if self.at(T.NOT):
            op = self.next()
            operand = self.parse_unary(allow_struct)
            return UnOp("!", operand, op.line)
        if self.at(T.AWAIT):
            op = self.next()
            operand = self.parse_unary(allow_struct)
            return AwaitExpr(operand, op.line)
        return self.parse_postfix_expr(allow_struct)

    def parse_postfix_expr(self, allow_struct=False):
        expr = self.parse_expr_atom(allow_struct)
        while True:
            tok = self.peek()
            if tok.type == T.LPAREN:
                args, named = self.parse_call_args()
                callee = self._expr_to_callee(expr)
                expr = Call(callee, args, named, tok.line)
            elif tok.type == T.LBRACKET:
                self.next()
                idx = self.parse_expr(allow_struct=True)
                self.expect(T.RBRACKET, "']'")
                expr = Index(expr, idx, tok.line)
            elif tok.type == T.DOT:
                self.next()
                name = self.expect(T.IDENT, "field name")
                expr = Member(expr, name.value, tok.line)
            elif tok.type == T.DOTX and self._starts_operand(self.peek(1)):
                break
            elif tok.type == T.QUESTION:
                self.next()
                expr = Propagate(expr, tok.line)
            elif tok.type == T.AS:
                self.next()
                ctype = self.parse_type()
                expr = Cast(expr, ctype, tok.line)
            else:
                break
        return expr

    def _expr_to_callee(self, expr):
        if isinstance(expr, Ident):
            return expr.name
        if isinstance(expr, Call):
            return expr.callee
        self.err("E110", "expression is not callable", self.peek())

    def parse_call_args(self):
        self.expect(T.LPAREN, "'('")
        args, named = [], {}
        while not self.at(T.RPAREN):
            if self.at(T.IDENT) and self.peek(1).type == T.COLON and self.peek(2).type != T.COLON:
                key = self.next().value
                self.next()  # ':'
                named[key] = self.parse_expr(allow_struct=True)
            else:
                args.append(self.parse_expr(allow_struct=True))
            if self.at(T.COMMA):
                self.next()
        self.expect(T.RPAREN, "')'")
        return args, named

    def parse_path(self) -> str:
        tok = self.expect(T.IDENT, "identifier")
        parts = [tok.value]
        while self.at(T.COLONCOLON):
            self.next()
            parts.append(self.expect(T.IDENT, "path segment").value)
        return "::".join(parts)

    OPERAND_START = {T.INT, T.FLOAT, T.STRING, T.TRUE, T.FALSE, T.IDENT,
                     T.LPAREN, T.LBRACKET, T.AT, T.MINUS, T.NOT, T.ASYNC}

    def _starts_operand(self, tok) -> bool:
        return tok.type in self.OPERAND_START

    def parse_expr_atom(self, allow_struct=False):
        tok = self.peek()

        if tok.type == T.INT:
            self.next(); return Literal(tok.value, "int", tok.line)
        if tok.type == T.FLOAT:
            self.next(); return Literal(tok.value, "float", tok.line)
        if tok.type == T.STRING:
            self.next(); return Literal(tok.value, "string", tok.line)
        if tok.type == T.TRUE:
            self.next(); return Literal(True, "bool", tok.line)
        if tok.type == T.FALSE:
            self.next(); return Literal(False, "bool", tok.line)

        if tok.type == T.AT:
            self.next()
            name = self.expect(T.IDENT, "SMV name")
            if name.value == "grad" and self.at(T.LPAREN):
                self.next()
                fn = self.expect(T.IDENT, "function name").value
                self.expect(T.RPAREN, "')'")
                return GradExpr(fn, tok.line)
            return SMV(name.value, tok.line)

        if tok.type == T.ASYNC:
            self.next()
            body = self.parse_block_stmts()
            return AsyncExpr(body, tok.line)

        if tok.type == T.LPAREN:
            self.next()
            e = self.parse_expr(allow_struct=True)
            self.expect(T.RPAREN, "')'")
            return e

        if tok.type == T.LBRACKET:
            return self.parse_array_like()

        if tok.type == T.IDENT:
            path_tok = tok
            path = self.parse_path()
            if path == "None" and not self.at(T.LPAREN):
                return Call("None", [], {}, path_tok.line)
            if allow_struct and self.at(T.LBRACE) and self._looks_like_struct_lit():
                name_parts = path.split("::")
                fields = self.parse_struct_fields()
                return StructLit(name_parts[-1], fields, path_tok.line)
            return Ident(path, path_tok.line)

        self.err("E110", f"unexpected {self.tok_desc(tok)} in expression",
                 hint="expr: literal / ident / call / @smv / [array] / (paren)")


    def _looks_like_struct_lit(self) -> bool:
        t1, t2, t3 = self.peek(0), self.peek(1), self.peek(2)
        if t1.type != T.LBRACE:
            return False
        if t2.type == T.RBRACE:
            return True
        return t2.type == T.IDENT and t3.type == T.COLON

    def parse_struct_fields(self) -> dict:
        self.expect(T.LBRACE, "'{'")
        fields = {}
        while not self.at(T.RBRACE):
            fname = self.expect(T.IDENT, "field name")
            self.expect(T.COLON, "':'")
            fields[fname.value] = self.parse_expr(allow_struct=True)
            if self.at(T.COMMA):
                self.next()
        self.expect(T.RBRACE, "'}'")
        return fields

    def parse_array_like(self):
        tok = self.expect(T.LBRACKET, "'['")
        if self.at(T.RBRACKET):
            self.next()
            return ArrayLit([], tok.line)
        first = self.parse_expr(allow_struct=True)
        if self.at(T.SEMI):
            self.next()
            size = self.parse_expr(allow_struct=True)
            self.expect(T.RBRACKET, "']'")
            return ArrayFill(first, size, tok.line)
        items = [first]
        while self.at(T.COMMA):
            self.next()
            if self.at(T.RBRACKET):
                break
            items.append(self.parse_expr(allow_struct=True))
        self.expect(T.RBRACKET, "']'")
        return ArrayLit(items, tok.line)

class XType:
    def __str__(self):
        return getattr(self, "name", "?")


class XNamedType(XType):
    def __init__(self, name): self.name = name
    def __str__(self): return self.name


class XVecType(XType):
    def __init__(self, n): self.n = n
    def __str__(self): return f"vec{self.n}"


class XMatType(XType):
    def __init__(self, r, c): self.r, self.c = r, c
    def __str__(self): return f"mat{self.r}" if self.r == self.c else f"mat[{self.r},{self.c}]"


class XMatShapeType(XType):
    def __init__(self, dims): self.dims = dims
    def __str__(self): return f"mat[{', '.join(map(str, self.dims))}]"


class XTensorType(XType):
    def __init__(self, dims): self.dims = dims
    def __str__(self): return f"tensor[{', '.join(map(str, self.dims))}]"


class XArrayType(XType):
    def __init__(self, elem, size): self.elem, self.size = elem, size
    def __str__(self): return f"[{self.elem}; {self.size}]"


class XGenericType(XType):
    def __init__(self, name, args): self.name, self.args = name, args
    def __str__(self): return f"{self.name}<{', '.join(map(str, self.args))}>"


class XOptionalType(XType):
    def __init__(self, inner): self.inner = inner
    def __str__(self): return f"{self.inner}?"


def parse_program(source: str, filename: str = "main.xp") -> Program:
    return Parser(source, filename).parse()
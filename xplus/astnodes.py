from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class Program:
    file: str
    uses: list["UseStmt"] = field(default_factory=list)
    decls: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {"Program": {"file": self.file,
                            "uses": [u.path for u in self.uses],
                            "decls": [n.to_dict() for n in self.decls]}}


@dataclass
class UseStmt:
    path: str
    line: int = 0

@dataclass
class Param:
    name: str
    type: Any
    line: int = 0


@dataclass
class FnDecl:
    name: str
    params: list[Param] = field(default_factory=list)
    ret_type: Any = None
    body: list = field(default_factory=list)
    attrs: list["Attribute"] = field(default_factory=list)
    is_entry: bool = False
    is_pub: bool = False
    export_c: bool = False
    namespace: Optional[str] = None
    line: int = 0

    def to_dict(self):
        return {"FunctionDecl": {"name": self.name,
                                 "params": [{"name": p.name, "type": str(p.type)} for p in self.params],
                                 "return_type": str(self.ret_type),
                                 "attrs": [a.name for a in self.attrs],
                                 "entry": self.is_entry,
                                 "pub": self.is_pub}}


@dataclass
class ConstDecl:
    name: str
    type: Any
    value: Any
    comptime: bool = False
    is_pub: bool = False
    namespace: Optional[str] = None
    line: int = 0

    def to_dict(self):
        return {"ConstDecl": {"name": self.name, "type": str(self.type),
                              "comptime": self.comptime}}


@dataclass
class StructDecl:
    name: str
    fields: list[Param]
    namespace: Optional[str] = None
    line: int = 0

    def to_dict(self):
        return {"StructDecl": {"name": self.name,
                               "fields": [{"name": f.name, "type": str(f.type)} for f in self.fields]}}


@dataclass
class EnumDecl:
    name: str
    variants: list[str]
    namespace: Optional[str] = None
    line: int = 0

    def to_dict(self):
        return {"EnumDecl": {"name": self.name, "variants": self.variants}}


@dataclass
class RegisterDecl:
    name: str
    type: Any
    address: int
    line: int = 0

    def to_dict(self):
        return {"RegisterDecl": {"name": self.name, "type": str(self.type),
                                 "address": hex(self.address)}}


@dataclass
class Attribute:
    name: str
    args: Optional[Any] = None
    line: int = 0

    def to_dict(self):
        return {"Attribute": {"name": self.name, "args": str(self.args)}}

@dataclass
class VarDecl:
    name: str
    type: Any
    init: Any
    mutable: bool = True
    line: int = 0

    def to_dict(self):
        return {"VarDecl": {"name": self.name, "type": str(self.type) if self.type else "<infer>"}}


@dataclass
class Assign:
    target: Any
    op: str
    value: Any
    line: int = 0

    def to_dict(self):
        return {"Assign": {"op": self.op, "target": self.target.to_dict(),
                           "value": self.value.to_dict()}}


@dataclass
class ExprStmt:
    expr: Any
    line: int = 0

    def to_dict(self):
        return {"ExprStmt": self.expr.to_dict()}


@dataclass
class ShowBlock:
    channel: str
    fmt: str
    injections: list
    line: int = 0

    def to_dict(self):
        return {"ShowBlock": {"channel": self.channel, "fmt": self.fmt,
                              "injections": len(self.injections)}}


@dataclass
class IfStmt:
    branches: list
    else_body: Optional[list] = None
    line: int = 0

    def to_dict(self):
        return {"IfStmt": {"branches": len(self.branches),
                           "has_else": self.else_body is not None}}


@dataclass
class LoopStmt:
    cond: Optional[Any]
    body: list
    line: int = 0

    def to_dict(self):
        return {"LoopStmt": {"conditional": self.cond is not None}}


@dataclass
class ForStmt:
    var: str
    range_expr: Any = None
    iter_expr: Any = None
    body: list = field(default_factory=list)
    parallel: bool = False
    line: int = 0

    def to_dict(self):
        return {"ForStmt": {"var": self.var, "parallel": self.parallel}}


@dataclass
class BreakStmt:
    line: int = 0

    def to_dict(self):
        return {"Break": {}}


@dataclass
class ContinueStmt:
    line: int = 0

    def to_dict(self):
        return {"Continue": {}}


@dataclass
class ReturnStmt:
    value: Any = None
    line: int = 0

    def to_dict(self):
        return {"Return": {"value": self.value.to_dict() if self.value else None}}


@dataclass
class GuardStmt:
    cond: Any
    else_body: list
    line: int = 0

    def to_dict(self):
        return {"GuardStmt": {"cond": self.cond.to_dict()}}


@dataclass
class MatchStmt:
    subject: Any
    arms: list
    line: int = 0

    def to_dict(self):
        return {"MatchStmt": {"arms": len(self.arms)}}


@dataclass
class ArenaBlock:
    body: list
    line: int = 0

    def to_dict(self):
        return {"ArenaBlock": {"stmts": len(self.body)}}


@dataclass
class ComptimeBlock:
    body: list
    line: int = 0

    def to_dict(self):
        return {"ComptimeBlock": {"stmts": len(self.body)}}


@dataclass
class UnsafeBlock:
    body: list
    line: int = 0

    def to_dict(self):
        return {"UnsafeBlock": {"stmts": len(self.body)}}


@dataclass
class AsmBlock:
    instrs: list
    line: int = 0

    def to_dict(self):
        return {"AsmBlock": {"instrs": self.instrs}}


@dataclass
class TensorCoreBlock:
    backend: str
    body: list
    line: int = 0

    def to_dict(self):
        return {"TensorCoreBlock": {"backend": self.backend, "stmts": len(self.body)}}


@dataclass
class VerifyStmt:
    cond: Any
    line: int = 0

    def to_dict(self):
        return {"VerifyStmt": {"cond": self.cond.to_dict()}}

@dataclass
class Literal:
    value: Any
    kind: str
    line: int = 0

    def to_dict(self):
        return {"Literal": self.value}


@dataclass
class Ident:
    name: str
    line: int = 0

    def to_dict(self):
        return {"Ident": self.name}


@dataclass
class SMV:
    name: str
    line: int = 0

    def to_dict(self):
        return {"SMV": self.name}


@dataclass
class BinOp:
    op: str
    left: Any
    right: Any
    line: int = 0

    def to_dict(self):
        return {"BinaryOp": {"op": self.op, "left": self.left.to_dict(),
                             "right": self.right.to_dict()}}


@dataclass
class UnOp:
    op: str
    operand: Any
    line: int = 0

    def to_dict(self):
        return {"UnaryOp": {"op": self.op, "operand": self.operand.to_dict()}}


@dataclass
class Call:
    callee: str
    args: list = field(default_factory=list)
    named_args: dict = field(default_factory=dict)
    line: int = 0

    def to_dict(self):
        return {"Call": {"callee": self.callee, "args": [a.to_dict() for a in self.args],
                         "named": {k: v.to_dict() for k, v in self.named_args.items()}}}


@dataclass
class Member:
    obj: Any
    name: str
    line: int = 0

    def to_dict(self):
        return {"Member": {"obj": self.obj.to_dict(), "name": self.name}}


@dataclass
class Index:
    obj: Any
    index: Any
    line: int = 0

    def to_dict(self):
        return {"Index": {"obj": self.obj.to_dict(), "index": self.index.to_dict()}}


@dataclass
class RangeExpr:
    start: Any
    end: Any
    inclusive: bool = False
    line: int = 0

    def to_dict(self):
        return {"Range": {"start": self.start.to_dict(), "end": self.end.to_dict(),
                          "inclusive": self.inclusive}}


@dataclass
class Pipeline:
    left: Any
    right: Any
    line: int = 0

    def to_dict(self):
        return {"Pipeline": {"left": self.left.to_dict(), "right": self.right.to_dict()}}


@dataclass
class Cast:
    expr: Any
    type: Any
    line: int = 0

    def to_dict(self):
        return {"Cast": {"expr": self.expr.to_dict(), "type": str(self.type)}}


@dataclass
class Propagate:
    expr: Any
    line: int = 0

    def to_dict(self):
        return {"Propagate": self.expr.to_dict()}


@dataclass
class StructLit:
    name: str
    fields: dict
    line: int = 0

    def to_dict(self):
        return {"StructLit": {"name": self.name, "fields": list(self.fields.keys())}}


@dataclass
class EnumLit:
    name: str
    variant: str
    payload: Any = None
    line: int = 0

    def to_dict(self):
        return {"EnumLit": f"{self.name}::{self.variant}"}


@dataclass
class ArrayLit:
    items: list
    line: int = 0

    def to_dict(self):
        return {"ArrayLit": [i.to_dict() for i in self.items]}


@dataclass
class ArrayFill:
    value: Any
    size: Any
    line: int = 0

    def to_dict(self):
        return {"ArrayFill": {"size": self.size.to_dict()}}


@dataclass
class AsyncExpr:
    body: list
    line: int = 0

    def to_dict(self):
        return {"AsyncExpr": {"stmts": len(self.body)}}


@dataclass
class AwaitExpr:
    expr: Any
    line: int = 0

    def to_dict(self):
        return {"AwaitExpr": self.expr.to_dict()}


@dataclass
class GradExpr:
    fn_name: str
    line: int = 0

    def to_dict(self):
        return {"GradExpr": self.fn_name}


@dataclass
class VecLit:
    items: list
    line: int = 0

    def to_dict(self):
        return {"VecLit": [i.to_dict() for i in self.items]}
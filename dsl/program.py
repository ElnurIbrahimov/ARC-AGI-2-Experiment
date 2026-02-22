"""
DSLProgram — tree representation of a DSL program with token serialization.

Uses prefix notation for token serialization:
    OP ARG1 ARG2 ...
where each ARG can itself be a nested OP expression.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.dsl_config import (
    DSL_REGISTRY, TOKEN_TO_PRIMITIVE, PRIMITIVE_NAME_TO_ID,
    GRID_REF, CONST_INT, CONST_COLOR,
    color_to_token, token_to_color,
    const_to_token, token_to_const,
    is_primitive_token, is_color_token, is_const_token,
    COLOR_OFFSET, CONST_OFFSET,
)


@dataclass
class DSLNode:
    """A single node in the DSL program tree."""
    op: str                                          # primitive name or "__input__" / "__const__" / "__color__"
    args: List[Union[DSLNode, int, str]] = field(default_factory=list)

    def depth(self) -> int:
        """Maximum depth of this subtree."""
        if not self.args:
            return 1
        child_depths = []
        for a in self.args:
            if isinstance(a, DSLNode):
                child_depths.append(a.depth())
            else:
                child_depths.append(1)
        return 1 + max(child_depths)

    def size(self) -> int:
        """Total number of nodes in this subtree."""
        total = 1
        for a in self.args:
            if isinstance(a, DSLNode):
                total += a.size()
            else:
                total += 1  # leaf constants count as nodes
        return total

    def to_string(self, indent: int = 0) -> str:
        """Human-readable nested string."""
        if self.op == "__input__":
            return "INPUT"
        if self.op == "__const__":
            return str(self.args[0]) if self.args else "?"
        if self.op == "__color__":
            return f"C{self.args[0]}" if self.args else "C?"

        if not self.args:
            return self.op

        arg_strs = []
        for a in self.args:
            if isinstance(a, DSLNode):
                arg_strs.append(a.to_string(indent + 1))
            else:
                arg_strs.append(str(a))
        return f"{self.op}({', '.join(arg_strs)})"


@dataclass
class DSLProgram:
    """
    Complete DSL program with a root node.
    Supports serialization to/from token sequences (prefix notation).
    """
    root: DSLNode

    def to_tokens(self) -> List[int]:
        """Serialize program tree to a flat token sequence (prefix notation)."""
        tokens: List[int] = []
        self._node_to_tokens(self.root, tokens)
        return tokens

    def _node_to_tokens(self, node: DSLNode, tokens: List[int]) -> None:
        """Recursively serialize a node to tokens."""
        if node.op == "__input__":
            tokens.append(GRID_REF)
            return

        if node.op == "__const__":
            tokens.append(CONST_INT)
            val = int(node.args[0]) if node.args else 0
            tokens.append(const_to_token(val))
            return

        if node.op == "__color__":
            tokens.append(CONST_COLOR)
            val = int(node.args[0]) if node.args else 0
            tokens.append(color_to_token(val))
            return

        # It's a primitive — emit its token ID, then recurse on args
        if node.op not in PRIMITIVE_NAME_TO_ID:
            raise ValueError(f"Unknown primitive: {node.op}")
        tokens.append(PRIMITIVE_NAME_TO_ID[node.op])

        for arg in node.args:
            if isinstance(arg, DSLNode):
                self._node_to_tokens(arg, tokens)
            elif isinstance(arg, int):
                # Bare int -> treat as const
                tokens.append(CONST_INT)
                tokens.append(const_to_token(arg))
            else:
                raise ValueError(f"Unexpected arg type: {type(arg)}")

    @classmethod
    def from_tokens(cls, tokens: List[int]) -> DSLProgram:
        """Deserialize a token sequence (prefix notation) into a DSLProgram."""
        pos = [0]  # mutable index

        def _parse() -> DSLNode:
            if pos[0] >= len(tokens):
                raise ValueError("Unexpected end of token sequence")

            tok = tokens[pos[0]]
            pos[0] += 1

            # Input grid reference
            if tok == GRID_REF:
                return DSLNode(op="__input__")

            # Integer constant
            if tok == CONST_INT:
                if pos[0] >= len(tokens):
                    raise ValueError("Expected constant value after CONST_INT")
                val_tok = tokens[pos[0]]
                pos[0] += 1
                return DSLNode(op="__const__", args=[token_to_const(val_tok)])

            # Color constant
            if tok == CONST_COLOR:
                if pos[0] >= len(tokens):
                    raise ValueError("Expected color value after CONST_COLOR")
                val_tok = tokens[pos[0]]
                pos[0] += 1
                return DSLNode(op="__color__", args=[token_to_color(val_tok)])

            # Primitive
            if is_primitive_token(tok):
                prim = TOKEN_TO_PRIMITIVE[tok]
                args: List[Union[DSLNode, int, str]] = []
                for _ in range(prim.arity):
                    args.append(_parse())
                return DSLNode(op=prim.name, args=args)

            raise ValueError(f"Unexpected token ID: {tok}")

        root = _parse()

        # Ensure all tokens consumed
        if pos[0] != len(tokens):
            raise ValueError(
                f"Extra tokens after program end: consumed {pos[0]}/{len(tokens)}")

        return cls(root=root)

    def to_string(self) -> str:
        """Human-readable program representation."""
        return self.root.to_string()

    def depth(self) -> int:
        """Maximum depth of the program tree."""
        return self.root.depth()

    def size(self) -> int:
        """Total number of nodes in the program tree."""
        return self.root.size()

    def __repr__(self) -> str:
        return f"DSLProgram({self.to_string()})"


# ─── Convenience builders ────────────────────────────────────────────

def input_node() -> DSLNode:
    """Create a node representing the input grid."""
    return DSLNode(op="__input__")


def const_node(value: int) -> DSLNode:
    """Create a node representing an integer constant."""
    return DSLNode(op="__const__", args=[value])


def color_node(color: int) -> DSLNode:
    """Create a node representing a color constant."""
    return DSLNode(op="__color__", args=[color])


def prim_node(name: str, *args: Union[DSLNode, int]) -> DSLNode:
    """
    Create a primitive node. Raw ints in args are wrapped as const_node.
    """
    if name not in DSL_REGISTRY:
        raise ValueError(f"Unknown primitive: {name}")
    wrapped = []
    for a in args:
        if isinstance(a, DSLNode):
            wrapped.append(a)
        elif isinstance(a, int):
            wrapped.append(const_node(a))
        else:
            raise TypeError(f"Unexpected arg type: {type(a)}")
    return DSLNode(op=name, args=wrapped)

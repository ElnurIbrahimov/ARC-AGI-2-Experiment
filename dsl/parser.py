"""
Token sequence -> DSLProgram tree parser.

Parses prefix-notation token sequences produced by the model into
DSLProgram trees. Validates syntax and arities.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.dsl_config import (
    TOKEN_TO_PRIMITIVE, GRID_REF, CONST_INT, CONST_COLOR,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    is_primitive_token, is_color_token, is_const_token,
    token_to_color, token_to_const,
)
from dsl.program import DSLProgram, DSLNode


@dataclass
class ParseError:
    """Structured parse error."""
    position: int          # token index where error occurred
    message: str
    token_id: Optional[int] = None


@dataclass
class ParseResult:
    """Result of parsing a token sequence."""
    program: Optional[DSLProgram]
    success: bool
    error: Optional[ParseError] = None
    tokens_consumed: int = 0


class DSLParser:
    """
    Parse prefix-notation token sequences into DSLProgram trees.

    Token format (prefix notation):
        PRIMITIVE_TOKEN ARG1 ARG2 ...
        GRID_REF                         -> input grid leaf
        CONST_INT CONST_VAL_TOKEN        -> integer constant leaf
        CONST_COLOR COLOR_VAL_TOKEN      -> color constant leaf
    """

    def parse(self, tokens: List[int]) -> ParseResult:
        """
        Parse a token sequence into a DSLProgram.
        Strips BOS/EOS/PAD if present.
        """
        # Strip special framing tokens
        cleaned = self._strip_framing(tokens)

        if not cleaned:
            return ParseResult(
                program=None, success=False,
                error=ParseError(0, "Empty token sequence after stripping BOS/EOS/PAD"),
            )

        try:
            node, consumed = self._parse_node(cleaned, 0)
        except _ParseException as e:
            return ParseResult(
                program=None, success=False,
                error=ParseError(e.position, e.message, e.token_id),
            )

        if consumed != len(cleaned):
            return ParseResult(
                program=None, success=False,
                error=ParseError(
                    consumed,
                    f"Extra tokens after program end: consumed {consumed}/{len(cleaned)}",
                ),
                tokens_consumed=consumed,
            )

        return ParseResult(
            program=DSLProgram(root=node),
            success=True,
            tokens_consumed=consumed,
        )

    def validate_tokens(self, tokens: List[int]) -> List[ParseError]:
        """
        Validate a token sequence without building a tree.
        Returns list of errors found (empty = valid).
        """
        errors: List[ParseError] = []
        cleaned = self._strip_framing(tokens)

        if not cleaned:
            errors.append(ParseError(0, "Empty token sequence"))
            return errors

        try:
            _, consumed = self._parse_node(cleaned, 0)
            if consumed != len(cleaned):
                errors.append(ParseError(
                    consumed,
                    f"Extra tokens: consumed {consumed}/{len(cleaned)}",
                ))
        except _ParseException as e:
            errors.append(ParseError(e.position, e.message, e.token_id))

        return errors

    def _strip_framing(self, tokens: List[int]) -> List[int]:
        """Remove BOS, EOS, and PAD tokens."""
        return [t for t in tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]

    def _parse_node(self, tokens: List[int], pos: int) -> Tuple[DSLNode, int]:
        """
        Parse one node (and its subtree) starting at position pos.
        Returns (node, new_position).
        """
        if pos >= len(tokens):
            raise _ParseException(pos, "Unexpected end of token sequence")

        tok = tokens[pos]

        # ── Input grid reference ──
        if tok == GRID_REF:
            return DSLNode(op="__input__"), pos + 1

        # ── Integer constant ──
        if tok == CONST_INT:
            if pos + 1 >= len(tokens):
                raise _ParseException(pos, "Expected constant value after CONST_INT")
            val_tok = tokens[pos + 1]
            if not is_const_token(val_tok):
                raise _ParseException(
                    pos + 1,
                    f"Expected constant token (74-213) after CONST_INT, got {val_tok}",
                    val_tok,
                )
            return DSLNode(op="__const__", args=[token_to_const(val_tok)]), pos + 2

        # ── Color constant ──
        if tok == CONST_COLOR:
            if pos + 1 >= len(tokens):
                raise _ParseException(pos, "Expected color value after CONST_COLOR")
            val_tok = tokens[pos + 1]
            if not is_color_token(val_tok):
                raise _ParseException(
                    pos + 1,
                    f"Expected color token (4-13) after CONST_COLOR, got {val_tok}",
                    val_tok,
                )
            return DSLNode(op="__color__", args=[token_to_color(val_tok)]), pos + 2

        # ── Primitive ──
        if is_primitive_token(tok):
            prim = TOKEN_TO_PRIMITIVE[tok]
            args = []
            cur = pos + 1
            for i in range(prim.arity):
                if cur >= len(tokens):
                    raise _ParseException(
                        cur,
                        f"Primitive '{prim.name}' expects {prim.arity} args, "
                        f"but only {i} found before end of sequence",
                    )
                child, cur = self._parse_node(tokens, cur)
                args.append(child)
            return DSLNode(op=prim.name, args=args), cur

        # ── Unknown token ──
        raise _ParseException(pos, f"Unexpected token ID: {tok}", tok)


class _ParseException(Exception):
    """Internal exception for parse errors."""
    def __init__(self, position: int, message: str, token_id: Optional[int] = None):
        self.position = position
        self.message = message
        self.token_id = token_id
        super().__init__(message)

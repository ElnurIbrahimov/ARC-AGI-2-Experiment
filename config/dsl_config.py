"""
DSL primitive registry — defines all ~50 primitives with token IDs,
structural tokens, and the full vocabulary layout.

Vocab layout (aligned with model_config.py):
  0     PAD
  1     BOS
  2     EOS
  3     SEP
  4-13  Colors 0-9
  14-63 DSL primitives (~50)
  64-73 Structural tokens (parens, arg sep, grid ref, const markers)
  74-213 Reserved constants 0-139
  Total: 214
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Special token IDs ───────────────────────────────────────────────

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SEP_TOKEN = 3

COLOR_OFFSET = 4       # colors 0-9 at token IDs 4-13
PRIMITIVE_OFFSET = 14   # DSL primitives start here
STRUCT_OFFSET = 64      # structural tokens start here
CONST_OFFSET = 74       # constant values 0-139 at token IDs 74-213

VOCAB_SIZE = 214


# ─── DSL categories ──────────────────────────────────────────────────

class DSLCategory:
    SPATIAL = "SPATIAL"
    COLOR = "COLOR"
    OBJECT = "OBJECT"
    GRID = "GRID"
    SET = "SET"
    LOGIC = "LOGIC"
    COMPOSITION = "COMPOSITION"
    PATTERN = "PATTERN"


# ─── Structural tokens ───────────────────────────────────────────────

OPEN_PAREN = STRUCT_OFFSET + 0    # 64
CLOSE_PAREN = STRUCT_OFFSET + 1   # 65
ARG_SEP = STRUCT_OFFSET + 2       # 66
GRID_REF = STRUCT_OFFSET + 3      # 67  — refers to the input grid
CONST_INT = STRUCT_OFFSET + 4     # 68  — next token is a constant value
CONST_COLOR = STRUCT_OFFSET + 5   # 69  — next token is a color value
LIST_START = STRUCT_OFFSET + 6    # 70
LIST_END = STRUCT_OFFSET + 7      # 71
NULL_TOKEN = STRUCT_OFFSET + 8    # 72
MASK_REF = STRUCT_OFFSET + 9      # 73  — refers to a mask/boolean grid

STRUCTURAL_TOKENS = {
    "OPEN_PAREN": OPEN_PAREN,
    "CLOSE_PAREN": CLOSE_PAREN,
    "ARG_SEP": ARG_SEP,
    "GRID_REF": GRID_REF,
    "CONST_INT": CONST_INT,
    "CONST_COLOR": CONST_COLOR,
    "LIST_START": LIST_START,
    "LIST_END": LIST_END,
    "NULL": NULL_TOKEN,
    "MASK_REF": MASK_REF,
}


# ─── Type tags for primitive I/O ─────────────────────────────────────

class ArgType:
    GRID = "grid"
    MASK = "mask"
    INT = "int"
    COLOR = "color"
    OBJECTS = "objects"         # list of (mask, color)
    TUPLE = "tuple"
    BOOL = "bool"
    DICT = "dict"
    FUNC = "func"              # callable (for composition prims)
    LIST_FUNC = "list_func"    # list of callables
    SYMMETRY = "symmetry"


# ─── Primitive definition ────────────────────────────────────────────

@dataclass(frozen=True)
class DSLPrimitiveDef:
    token_id: int
    name: str
    category: str
    arity: int                    # number of inputs
    input_types: Tuple[str, ...]  # ArgType values
    output_type: str              # ArgType value
    description: str = ""


# ─── Primitive registry (name -> DSLPrimitiveDef) ────────────────────

def _build_registry() -> Dict[str, DSLPrimitiveDef]:
    _id = PRIMITIVE_OFFSET  # start at 14
    defs: List[DSLPrimitiveDef] = []

    def _add(name, cat, arity, in_types, out_type, desc=""):
        nonlocal _id
        defs.append(DSLPrimitiveDef(
            token_id=_id, name=name, category=cat,
            arity=arity, input_types=tuple(in_types),
            output_type=out_type, description=desc,
        ))
        _id += 1

    G = ArgType.GRID
    M = ArgType.MASK
    I = ArgType.INT
    C = ArgType.COLOR
    O = ArgType.OBJECTS
    T = ArgType.TUPLE
    D = ArgType.DICT
    F = ArgType.FUNC
    LF = ArgType.LIST_FUNC
    S = ArgType.SYMMETRY

    # ── SPATIAL (14-20) ──
    _add("rot90",      DSLCategory.SPATIAL, 1, [G], G, "Rotate 90° clockwise")
    _add("rot180",     DSLCategory.SPATIAL, 1, [G], G, "Rotate 180°")
    _add("rot270",     DSLCategory.SPATIAL, 1, [G], G, "Rotate 270° clockwise")
    _add("hmirror",    DSLCategory.SPATIAL, 1, [G], G, "Flip left-right")
    _add("vmirror",    DSLCategory.SPATIAL, 1, [G], G, "Flip top-bottom")
    _add("transpose",  DSLCategory.SPATIAL, 1, [G], G, "Transpose rows/cols")
    _add("shift",      DSLCategory.SPATIAL, 3, [G, I, I], G, "Shift grid by dx,dy (fill 0)")

    # ── COLOR (21-27) ──
    _add("fill",               DSLCategory.COLOR, 2, [G, C], G, "Fill entire grid with color")
    _add("recolor",            DSLCategory.COLOR, 3, [G, C, C], G, "Replace from_color with to_color")
    _add("flood_fill",         DSLCategory.COLOR, 4, [G, I, I, C], G, "Flood fill from (row,col)")
    _add("color_map",          DSLCategory.COLOR, 2, [G, D], G, "Remap colors via dict")
    _add("most_common_color",  DSLCategory.COLOR, 1, [G], C, "Most frequent color")
    _add("least_common_color", DSLCategory.COLOR, 1, [G], C, "Least frequent non-bg color")
    _add("count_color",        DSLCategory.COLOR, 2, [G, C], I, "Count cells of a color")

    # ── OBJECT (28-34) ──
    _add("find_objects",    DSLCategory.OBJECT, 1, [G], O, "Connected components list")
    _add("isolate_object",  DSLCategory.OBJECT, 2, [G, I], G, "Keep only object at index")
    _add("extract_largest", DSLCategory.OBJECT, 1, [G], G, "Keep only largest object")
    _add("extract_smallest",DSLCategory.OBJECT, 1, [G], G, "Keep only smallest object")
    _add("bounding_box",    DSLCategory.OBJECT, 2, [G, M], T, "Bounding box of mask")
    _add("move_object",     DSLCategory.OBJECT, 4, [G, M, I, I], G, "Move masked object by dr,dc")
    _add("count_objects",   DSLCategory.OBJECT, 1, [G], I, "Number of connected components")

    # ── GRID (35-42) ──
    _add("crop",     DSLCategory.GRID, 5, [G, I, I, I, I], G, "Crop subgrid r1,c1,r2,c2")
    _add("trim",     DSLCategory.GRID, 1, [G], G, "Remove all-zero borders")
    _add("pad",      DSLCategory.GRID, 5, [G, I, I, I, I], G, "Pad top,bot,left,right with 0")
    _add("concat_h", DSLCategory.GRID, 2, [G, G], G, "Horizontal concatenation")
    _add("concat_v", DSLCategory.GRID, 2, [G, G], G, "Vertical concatenation")
    _add("tile",     DSLCategory.GRID, 3, [G, I, I], G, "Tile grid NxM")
    _add("resize",   DSLCategory.GRID, 3, [G, I, I], G, "Nearest-neighbor resize")
    _add("overlay",  DSLCategory.GRID, 3, [G, G, M], G, "Overlay top on base where mask")

    # ── SET (43-46) ──
    _add("intersection", DSLCategory.SET, 2, [G, G], G, "Non-zero where both non-zero")
    _add("union",        DSLCategory.SET, 2, [G, G], G, "Non-zero from either")
    _add("difference",   DSLCategory.SET, 2, [G, G], G, "Non-zero in g1 not g2")
    _add("xor",          DSLCategory.SET, 2, [G, G], G, "Non-zero in exactly one")

    # ── LOGIC (47-49) ──
    _add("if_color",       DSLCategory.LOGIC, 2, [G, C], M, "Boolean mask where grid==color")
    _add("filter_by_size", DSLCategory.LOGIC, 3, [O, I, I], O, "Filter objects by pixel count")
    _add("select_by_color",DSLCategory.LOGIC, 2, [G, C], M, "Mask of specific color")

    # ── COMPOSITION (50-52) ──
    _add("sequence",               DSLCategory.COMPOSITION, 2, [LF, G], G, "Apply funcs in order")
    _add("iterate_until_fixpoint", DSLCategory.COMPOSITION, 2, [F, G], G, "Apply until stable")
    _add("apply_to_each",          DSLCategory.COMPOSITION, 3, [F, O, G], G, "Apply func per object")

    # ── PATTERN (53-55) ──
    _add("detect_period",   DSLCategory.PATTERN, 2, [G, I], I, "Repeating period along axis")
    _add("extend_pattern",  DSLCategory.PATTERN, 3, [G, I, I], G, "Extend pattern to target size")
    _add("symmetry_type",   DSLCategory.PATTERN, 1, [G], S, "Which symmetries grid has")

    registry: Dict[str, DSLPrimitiveDef] = {}
    for d in defs:
        registry[d.name] = d
    return registry


DSL_REGISTRY: Dict[str, DSLPrimitiveDef] = _build_registry()

# Reverse lookup: token_id -> DSLPrimitiveDef
TOKEN_TO_PRIMITIVE: Dict[int, DSLPrimitiveDef] = {
    d.token_id: d for d in DSL_REGISTRY.values()
}

# Convenience: name -> token_id
PRIMITIVE_NAME_TO_ID: Dict[str, int] = {
    d.name: d.token_id for d in DSL_REGISTRY.values()
}


# ─── Helper functions ────────────────────────────────────────────────

def color_to_token(color: int) -> int:
    """Map ARC color (0-9) to its token ID."""
    assert 0 <= color <= 9, f"Color must be 0-9, got {color}"
    return COLOR_OFFSET + color


def token_to_color(token_id: int) -> int:
    """Map token ID to ARC color (0-9)."""
    color = token_id - COLOR_OFFSET
    assert 0 <= color <= 9, f"Token {token_id} is not a color token"
    return color


def const_to_token(value: int) -> int:
    """Map constant integer (0-139) to its token ID."""
    assert 0 <= value <= 139, f"Constant must be 0-139, got {value}"
    return CONST_OFFSET + value


def token_to_const(token_id: int) -> int:
    """Map token ID back to constant integer."""
    value = token_id - CONST_OFFSET
    assert 0 <= value <= 139, f"Token {token_id} is not a constant token"
    return value


def is_primitive_token(token_id: int) -> bool:
    return token_id in TOKEN_TO_PRIMITIVE


def is_color_token(token_id: int) -> bool:
    return COLOR_OFFSET <= token_id < COLOR_OFFSET + 10


def is_structural_token(token_id: int) -> bool:
    return STRUCT_OFFSET <= token_id < CONST_OFFSET


def is_const_token(token_id: int) -> bool:
    return CONST_OFFSET <= token_id < VOCAB_SIZE


def get_primitives_by_category(category: str) -> List[DSLPrimitiveDef]:
    return [d for d in DSL_REGISTRY.values() if d.category == category]

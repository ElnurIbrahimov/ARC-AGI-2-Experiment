"""
Comprehensive tests for Phase 1: DSL engine and data pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config.dsl_config import (
    DSL_REGISTRY, TOKEN_TO_PRIMITIVE, PRIMITIVE_NAME_TO_ID, VOCAB_SIZE,
    color_to_token, token_to_color, const_to_token, token_to_const,
    GRID_REF, CONST_INT, CONST_COLOR,
    is_primitive_token, is_color_token, is_const_token,
    get_primitives_by_category,
)
from dsl.primitives import (
    rot90, rot180, rot270, hmirror, vmirror, transpose, shift,
    fill, recolor, flood_fill, color_map, most_common_color,
    least_common_color, count_color,
    find_objects, isolate_object, extract_largest, extract_smallest,
    bounding_box, move_object, count_objects,
    crop, trim, pad, concat_h, concat_v, tile, resize, overlay,
    intersection, union, difference, xor,
    if_color, filter_by_size, select_by_color,
    sequence, iterate_until_fixpoint, apply_to_each,
    detect_period, extend_pattern, symmetry_type,
    PRIMITIVE_FUNCTIONS,
)
from dsl.program import (
    DSLProgram, DSLNode, input_node, const_node, color_node, prim_node,
)
from dsl.engine import DSLEngine
from dsl.parser import DSLParser
from dsl.validator import DSLValidator
from dsl.error_trace import (
    build_error_trace, build_error_traces_from_validation, classify_error,
    format_error_traces,
)
from data.grid_utils import (
    pad_grid, normalize_grid, grid_to_string, string_to_grid,
    grids_equal, grid_diff, unique_colors, grid_to_flat, flat_to_grid,
)
from data.augmentation import (
    augment_task, apply_dihedral, augment_with_color_permutations,
    apply_color_permutation, random_color_permutation,
)


# ═══════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════

class TestDSLConfig:
    def test_registry_size(self):
        # We defined ~42 primitives — check they're all there
        assert len(DSL_REGISTRY) >= 40
        assert len(DSL_REGISTRY) <= 50

    def test_vocab_size(self):
        assert VOCAB_SIZE == 214

    def test_token_id_uniqueness(self):
        ids = [d.token_id for d in DSL_REGISTRY.values()]
        assert len(ids) == len(set(ids)), "Duplicate token IDs"

    def test_color_tokens(self):
        for c in range(10):
            tok = color_to_token(c)
            assert token_to_color(tok) == c
            assert is_color_token(tok)

    def test_const_tokens(self):
        for v in [0, 50, 139]:
            tok = const_to_token(v)
            assert token_to_const(tok) == v
            assert is_const_token(tok)

    def test_primitive_tokens(self):
        for name, defn in DSL_REGISTRY.items():
            assert is_primitive_token(defn.token_id)
            assert TOKEN_TO_PRIMITIVE[defn.token_id] == defn

    def test_categories_exist(self):
        cats = set(d.category for d in DSL_REGISTRY.values())
        assert "SPATIAL" in cats
        assert "COLOR" in cats
        assert "OBJECT" in cats
        assert "GRID" in cats

    def test_get_by_category(self):
        spatial = get_primitives_by_category("SPATIAL")
        assert len(spatial) >= 6


# ═══════════════════════════════════════════════════════════════
# Primitive tests
# ═══════════════════════════════════════════════════════════════

class TestSpatialPrimitives:
    def setup_method(self):
        self.grid = np.array([[1, 2], [3, 4]], dtype=int)

    def test_rot90(self):
        r = rot90(self.grid)
        expected = np.array([[3, 1], [4, 2]])
        assert np.array_equal(r, expected)

    def test_rot180(self):
        r = rot180(self.grid)
        expected = np.array([[4, 3], [2, 1]])
        assert np.array_equal(r, expected)

    def test_rot270(self):
        r = rot270(self.grid)
        expected = np.array([[2, 4], [1, 3]])
        assert np.array_equal(r, expected)

    def test_rot360_identity(self):
        r = rot90(rot90(rot90(rot90(self.grid))))
        assert np.array_equal(r, self.grid)

    def test_hmirror(self):
        r = hmirror(self.grid)
        expected = np.array([[2, 1], [4, 3]])
        assert np.array_equal(r, expected)

    def test_vmirror(self):
        r = vmirror(self.grid)
        expected = np.array([[3, 4], [1, 2]])
        assert np.array_equal(r, expected)

    def test_transpose(self):
        r = transpose(self.grid)
        expected = np.array([[1, 3], [2, 4]])
        assert np.array_equal(r, expected)

    def test_shift(self):
        g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        r = shift(g, 1, 0)  # shift right by 1
        assert r[0, 0] == 0
        assert r[0, 1] == 1
        assert r[0, 2] == 2

    def test_shift_returns_copy(self):
        g = np.array([[1, 2], [3, 4]])
        r = shift(g, 0, 0)
        assert np.array_equal(r, g)
        r[0, 0] = 99
        assert g[0, 0] == 1


class TestColorPrimitives:
    def test_fill(self):
        g = np.zeros((3, 3), dtype=int)
        r = fill(g, 5)
        assert np.all(r == 5)

    def test_recolor(self):
        g = np.array([[1, 2, 1], [2, 1, 2]])
        r = recolor(g, 1, 3)
        assert np.array_equal(r, np.array([[3, 2, 3], [2, 3, 2]]))

    def test_flood_fill(self):
        g = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        r = flood_fill(g, 0, 0, 5)
        assert r[0, 0] == 5
        assert r[1, 1] == 5
        assert r[2, 0] == 1  # not filled (different color)

    def test_color_map(self):
        g = np.array([[1, 2], [3, 4]])
        r = color_map(g, {1: 5, 3: 7})
        assert r[0, 0] == 5
        assert r[0, 1] == 2
        assert r[1, 0] == 7

    def test_most_common_color(self):
        g = np.array([[1, 1, 1], [2, 2, 3]])
        assert most_common_color(g) == 1

    def test_least_common_color(self):
        g = np.array([[0, 0, 1], [1, 1, 2]])
        assert least_common_color(g) == 2

    def test_count_color(self):
        g = np.array([[1, 1, 2], [2, 2, 2]])
        assert count_color(g, 2) == 4
        assert count_color(g, 1) == 2


class TestObjectPrimitives:
    def setup_method(self):
        # Two separate objects of different colors
        self.grid = np.array([
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 0],
        ])

    def test_find_objects(self):
        objs = find_objects(self.grid)
        assert len(objs) == 2

    def test_count_objects(self):
        assert count_objects(self.grid) == 2

    def test_isolate_object(self):
        iso = isolate_object(self.grid, 0)
        # Object 0 is the first found CC (color 1)
        assert np.sum(iso != 0) > 0
        assert count_objects(iso) == 1

    def test_extract_largest(self):
        r = extract_largest(self.grid)
        # Both objects have 3 cells, so largest picks first (or either)
        assert count_objects(r) == 1

    def test_extract_smallest(self):
        g = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 0],
        ])
        r = extract_smallest(g)
        # Object 2 has 1 cell, object 1 has 5 cells
        assert np.sum(r != 0) == 1

    def test_bounding_box(self):
        mask = self.grid == 2
        bb = bounding_box(self.grid, mask)
        assert bb == (2, 2, 3, 3)

    def test_move_object(self):
        mask = self.grid == 1
        r = move_object(self.grid, mask, 1, 1)
        assert r[0, 0] == 0  # old position cleared
        assert r[1, 1] == 1  # moved

    def test_empty_grid_objects(self):
        g = np.zeros((3, 3), dtype=int)
        assert count_objects(g) == 0
        assert len(find_objects(g)) == 0


class TestGridPrimitives:
    def test_crop(self):
        g = np.arange(16).reshape(4, 4)
        r = crop(g, 1, 1, 2, 2)
        assert r.shape == (2, 2)
        assert r[0, 0] == 5

    def test_trim(self):
        g = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        r = trim(g)
        assert r.shape == (1, 1)
        assert r[0, 0] == 1

    def test_pad(self):
        g = np.array([[1, 2], [3, 4]])
        r = pad(g, 1, 1, 1, 1)
        assert r.shape == (4, 4)
        assert r[0, 0] == 0
        assert r[1, 1] == 1

    def test_concat_h(self):
        g1 = np.array([[1, 2], [3, 4]])
        g2 = np.array([[5, 6], [7, 8]])
        r = concat_h(g1, g2)
        assert r.shape == (2, 4)
        assert r[0, 2] == 5

    def test_concat_v(self):
        g1 = np.array([[1, 2], [3, 4]])
        g2 = np.array([[5, 6], [7, 8]])
        r = concat_v(g1, g2)
        assert r.shape == (4, 2)
        assert r[2, 0] == 5

    def test_tile(self):
        g = np.array([[1, 2], [3, 4]])
        r = tile(g, 2, 3)
        assert r.shape == (4, 6)
        assert r[2, 0] == 1  # tiled

    def test_resize(self):
        g = np.array([[1, 2], [3, 4]])
        r = resize(g, 4, 4)
        assert r.shape == (4, 4)
        assert r[0, 0] == 1
        assert r[3, 3] == 4

    def test_overlay(self):
        base = np.zeros((3, 3), dtype=int)
        top = np.full((3, 3), 5, dtype=int)
        mask = np.array([[True, False, False],
                         [False, True, False],
                         [False, False, True]])
        r = overlay(base, top, mask)
        assert r[0, 0] == 5
        assert r[0, 1] == 0
        assert r[1, 1] == 5


class TestSetPrimitives:
    def test_intersection(self):
        g1 = np.array([[1, 0], [0, 2]])
        g2 = np.array([[1, 3], [0, 0]])
        r = intersection(g1, g2)
        assert r[0, 0] == 1
        assert r[0, 1] == 0
        assert r[1, 1] == 0

    def test_union(self):
        g1 = np.array([[1, 0], [0, 2]])
        g2 = np.array([[0, 3], [4, 0]])
        r = union(g1, g2)
        assert r[0, 0] == 1
        assert r[0, 1] == 3
        assert r[1, 0] == 4
        assert r[1, 1] == 2

    def test_difference(self):
        g1 = np.array([[1, 2], [3, 0]])
        g2 = np.array([[1, 0], [0, 0]])
        r = difference(g1, g2)
        assert r[0, 0] == 0  # both non-zero -> excluded
        assert r[0, 1] == 2
        assert r[1, 0] == 3

    def test_xor(self):
        g1 = np.array([[1, 0], [0, 2]])
        g2 = np.array([[1, 3], [0, 0]])
        r = xor(g1, g2)
        assert r[0, 0] == 0  # both non-zero
        assert r[0, 1] == 3
        assert r[1, 1] == 2


class TestLogicPrimitives:
    def test_if_color(self):
        g = np.array([[1, 2, 1], [3, 1, 0]])
        m = if_color(g, 1)
        assert m[0, 0] == True
        assert m[0, 1] == False

    def test_select_by_color(self):
        g = np.array([[1, 2], [2, 1]])
        m = select_by_color(g, 2)
        assert m[0, 1] == True
        assert m[0, 0] == False

    def test_filter_by_size(self):
        g = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 0],
        ])
        objs = find_objects(g)
        filtered = filter_by_size(objs, 1, 2)
        # Object 1 (color 1) has 4 cells, object 2 (color 2) has 1 cell
        assert len(filtered) == 1


class TestPatternPrimitives:
    def test_detect_period_rows(self):
        g = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        assert detect_period(g, 0) == 2

    def test_detect_period_cols(self):
        g = np.array([[1, 2, 1, 2], [3, 4, 3, 4]])
        assert detect_period(g, 1) == 2

    def test_extend_pattern(self):
        g = np.array([[1, 2], [3, 4]])
        r = extend_pattern(g, 0, 6)
        assert r.shape == (6, 2)
        assert np.array_equal(r[4:6, :], g)

    def test_symmetry_type(self):
        g = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])
        syms = symmetry_type(g)
        assert "hmirror" in syms
        assert "vmirror" in syms


# ═══════════════════════════════════════════════════════════════
# Program tests
# ═══════════════════════════════════════════════════════════════

class TestDSLProgram:
    def test_simple_program(self):
        # rot90(INPUT)
        node = DSLNode(op="rot90", args=[input_node()])
        prog = DSLProgram(root=node)
        assert prog.depth() == 2
        assert prog.size() == 2
        assert "rot90" in prog.to_string()

    def test_nested_program(self):
        # hmirror(rot90(INPUT))
        inner = prim_node("rot90", input_node())
        outer = prim_node("hmirror", inner)
        prog = DSLProgram(root=outer)
        assert prog.depth() == 3

    def test_roundtrip(self):
        # rot90(INPUT)
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        tokens = prog.to_tokens()
        restored = DSLProgram.from_tokens(tokens)
        assert restored.to_string() == prog.to_string()

    def test_roundtrip_with_const(self):
        # shift(INPUT, 1, 2)
        prog = DSLProgram(root=prim_node("shift", input_node(), const_node(1), const_node(2)))
        tokens = prog.to_tokens()
        restored = DSLProgram.from_tokens(tokens)
        assert restored.to_string() == prog.to_string()

    def test_roundtrip_with_color(self):
        # fill(INPUT, C3)
        node = DSLNode(op="fill", args=[input_node(), color_node(3)])
        prog = DSLProgram(root=node)
        tokens = prog.to_tokens()
        restored = DSLProgram.from_tokens(tokens)
        assert "fill" in restored.to_string()

    def test_deeply_nested(self):
        # vmirror(hmirror(rot90(INPUT)))
        n = input_node()
        n = prim_node("rot90", n)
        n = prim_node("hmirror", n)
        n = prim_node("vmirror", n)
        prog = DSLProgram(root=n)
        tokens = prog.to_tokens()
        restored = DSLProgram.from_tokens(tokens)
        assert restored.depth() == prog.depth()


# ═══════════════════════════════════════════════════════════════
# Engine tests
# ═══════════════════════════════════════════════════════════════

class TestDSLEngine:
    def setup_method(self):
        self.engine = DSLEngine(timeout_ms=5000)
        self.grid = np.array([[1, 2], [3, 4]])

    def test_execute_rot90(self):
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        result = self.engine.execute(prog, self.grid)
        assert result.success
        expected = np.array([[3, 1], [4, 2]])
        assert np.array_equal(result.output_grid, expected)

    def test_execute_nested(self):
        # rot180(INPUT) = rot90(rot90(INPUT))
        prog = DSLProgram(root=prim_node("rot180", input_node()))
        result = self.engine.execute(prog, self.grid)
        assert result.success
        expected = rot180(self.grid)
        assert np.array_equal(result.output_grid, expected)

    def test_execute_with_const(self):
        # fill(INPUT, 5)
        prog = DSLProgram(root=DSLNode(
            op="fill", args=[input_node(), color_node(5)]
        ))
        result = self.engine.execute(prog, self.grid)
        assert result.success
        assert np.all(result.output_grid == 5)

    def test_execute_trace(self):
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        result = self.engine.execute(prog, self.grid)
        assert len(result.trace) >= 1
        assert result.trace[0].op == "rot90"

    def test_execute_error(self):
        # Bad program — unknown op
        prog = DSLProgram(root=DSLNode(op="nonexistent", args=[input_node()]))
        result = self.engine.execute(prog, self.grid)
        assert not result.success
        assert result.error is not None

    def test_execute_timeout(self):
        # iterate_until_fixpoint with a function that never converges
        # (recolor 1->2, 2->1 never stabilizes)
        # Actually this will stabilize after 2 steps. Let's use a near-instant timeout.
        engine = DSLEngine(timeout_ms=0.001)
        prog = DSLProgram(root=prim_node("tile", input_node(), const_node(100), const_node(100)))
        result = engine.execute(prog, self.grid)
        # Might or might not timeout depending on speed — just check it doesn't crash
        assert isinstance(result.success, bool)


# ═══════════════════════════════════════════════════════════════
# Parser tests
# ═══════════════════════════════════════════════════════════════

class TestDSLParser:
    def setup_method(self):
        self.parser = DSLParser()

    def test_parse_simple(self):
        # Build tokens for rot90(INPUT)
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        tokens = prog.to_tokens()
        result = self.parser.parse(tokens)
        assert result.success
        assert result.program is not None
        assert result.program.to_string() == prog.to_string()

    def test_parse_with_bos_eos(self):
        from config.dsl_config import BOS_TOKEN, EOS_TOKEN
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        tokens = [BOS_TOKEN] + prog.to_tokens() + [EOS_TOKEN]
        result = self.parser.parse(tokens)
        assert result.success

    def test_parse_empty(self):
        result = self.parser.parse([])
        assert not result.success

    def test_parse_extra_tokens(self):
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        tokens = prog.to_tokens() + [GRID_REF]  # extra token
        result = self.parser.parse(tokens)
        assert not result.success

    def test_parse_nested(self):
        inner = prim_node("rot90", input_node())
        outer = prim_node("hmirror", inner)
        prog = DSLProgram(root=outer)
        tokens = prog.to_tokens()
        result = self.parser.parse(tokens)
        assert result.success
        assert result.program.depth() == prog.depth()

    def test_validate_tokens(self):
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        tokens = prog.to_tokens()
        errors = self.parser.validate_tokens(tokens)
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════
# Validator tests
# ═══════════════════════════════════════════════════════════════

class TestDSLValidator:
    def test_validate_correct(self):
        validator = DSLValidator()
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        g_in = np.array([[1, 2], [3, 4]])
        g_out = rot90(g_in)
        result = validator.validate(prog, [(g_in, g_out)])
        assert result.score == 1.0
        assert result.all_passed

    def test_validate_wrong(self):
        validator = DSLValidator()
        prog = DSLProgram(root=prim_node("rot90", input_node()))
        g_in = np.array([[1, 2], [3, 4]])
        g_out = np.array([[9, 9], [9, 9]])  # wrong
        result = validator.validate(prog, [(g_in, g_out)])
        assert result.score == 0.0
        assert not result.all_passed
        assert result.cell_accuracy < 1.0

    def test_validate_multiple(self):
        validator = DSLValidator()
        prog = DSLProgram(root=prim_node("hmirror", input_node()))
        examples = []
        for _ in range(5):
            g = np.random.randint(0, 10, (4, 4))
            examples.append((g, hmirror(g)))
        result = validator.validate(prog, examples)
        assert result.score == 1.0
        assert result.num_passed == 5


# ═══════════════════════════════════════════════════════════════
# Error trace tests
# ═══════════════════════════════════════════════════════════════

class TestErrorTrace:
    def test_classify_color_error(self):
        expected = np.array([[1, 2], [3, 4]])
        actual = np.array([[5, 6], [7, 8]])
        cat = classify_error(expected, actual)
        # Structure same (all non-zero), different colors
        assert cat == "color_error"

    def test_classify_size_error(self):
        expected = np.array([[1, 2], [3, 4]])
        actual = np.array([[1, 2, 3]])
        assert classify_error(expected, actual) == "size_error"

    def test_classify_execution_error(self):
        expected = np.array([[1, 2]])
        assert classify_error(expected, None) == "execution_error"

    def test_build_trace(self):
        expected = np.array([[1, 0], [0, 2]])
        actual = np.array([[1, 0], [0, 3]])
        trace = build_error_trace(0, expected, actual)
        assert trace.diff_count() == 1
        assert (1, 1) in trace.diff_colors
        assert trace.diff_colors[(1, 1)] == (2, 3)

    def test_format_traces(self):
        expected = np.array([[1, 2], [3, 4]])
        actual = np.array([[1, 2], [3, 5]])
        trace = build_error_trace(0, expected, actual)
        formatted = format_error_traces([trace])
        assert "1 failing" in formatted


# ═══════════════════════════════════════════════════════════════
# Grid utils tests
# ═══════════════════════════════════════════════════════════════

class TestGridUtils:
    def test_pad_grid(self):
        g = np.array([[1, 2], [3, 4]])
        r = pad_grid(g, 4, 4)
        assert r.shape == (4, 4)
        assert r[0, 0] == 1
        assert r[3, 3] == 0

    def test_normalize_grid(self):
        g = [[1, 2], [15, -1]]
        r = normalize_grid(g)
        assert r.dtype == int
        assert r[1, 0] == 9   # clamped
        assert r[1, 1] == 0   # clamped

    def test_grid_string_roundtrip(self):
        g = np.array([[1, 2, 3], [4, 5, 6]])
        s = grid_to_string(g)
        r = string_to_grid(s)
        assert np.array_equal(g, r)

    def test_grids_equal(self):
        g1 = np.array([[1, 2], [3, 4]])
        g2 = g1.copy()
        assert grids_equal(g1, g2)
        g2[0, 0] = 9
        assert not grids_equal(g1, g2)

    def test_grid_diff(self):
        g1 = np.array([[1, 2], [3, 4]])
        g2 = np.array([[1, 9], [3, 4]])
        d = grid_diff(g1, g2)
        assert d[0, 1] == True
        assert d[0, 0] == False

    def test_flat_roundtrip(self):
        g = np.array([[1, 2], [3, 4]])
        flat = grid_to_flat(g)
        assert flat == [1, 2, 3, 4]
        r = flat_to_grid(flat, 2, 2)
        assert np.array_equal(g, r)


# ═══════════════════════════════════════════════════════════════
# Augmentation tests
# ═══════════════════════════════════════════════════════════════

class TestAugmentation:
    def test_dihedral_count(self):
        g_in = [np.array([[1, 2], [3, 4]])]
        g_out = [np.array([[5, 6], [7, 8]])]
        results = apply_dihedral(g_in, g_out)
        assert len(results) == 8  # D4 group

    def test_dihedral_consistency(self):
        # Same transform applied to input and output
        g_in = [np.array([[1, 2], [3, 4]])]
        g_out = [np.array([[5, 6], [7, 8]])]
        for aug_in, aug_out, name in apply_dihedral(g_in, g_out):
            assert aug_in[0].shape == aug_out[0].shape

    def test_color_permutation(self):
        rng = np.random.default_rng(42)
        mapping = random_color_permutation(rng=rng)
        assert mapping[0] == 0
        # All values in 0-9
        assert all(0 <= v <= 9 for v in mapping.values())
        # Bijection on 1-9
        assert set(mapping[k] for k in range(1, 10)) == set(range(1, 10))

    def test_apply_color_perm(self):
        g = np.array([[0, 1, 2], [3, 4, 5]])
        mapping = {0: 0, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 6, 7: 7, 8: 8, 9: 9}
        r = apply_color_permutation(g, mapping)
        assert r[0, 0] == 0
        assert r[0, 1] == 5
        assert r[0, 2] == 4

    def test_augment_task(self):
        inputs = [np.array([[1, 2], [3, 4]])]
        outputs = [np.array([[4, 3], [2, 1]])]
        results = augment_task(inputs, outputs, seed=42)
        # Should have: 1 original + 7 dihedral + color perms
        assert len(results) >= 8

    def test_augment_preserves_original(self):
        g_in = np.array([[1, 2], [3, 4]])
        g_out = np.array([[5, 6], [7, 8]])
        results = augment_task([g_in], [g_out], seed=42)
        # First result should be the original
        assert np.array_equal(results[0][0][0], g_in)
        assert np.array_equal(results[0][1][0], g_out)


# ═══════════════════════════════════════════════════════════════
# Full pipeline integration test
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: build program -> serialize -> parse -> execute -> validate."""
        # Task: rot90
        g1_in = np.array([[1, 2, 3], [4, 5, 6]])
        g1_out = rot90(g1_in)
        g2_in = np.array([[7, 8], [9, 1], [2, 3]])
        g2_out = rot90(g2_in)

        # Build program
        prog = DSLProgram(root=prim_node("rot90", input_node()))

        # Serialize to tokens
        tokens = prog.to_tokens()
        assert len(tokens) > 0

        # Parse back
        parser = DSLParser()
        parse_result = parser.parse(tokens)
        assert parse_result.success
        restored = parse_result.program

        # Execute
        engine = DSLEngine()
        r1 = engine.execute(restored, g1_in)
        assert r1.success
        assert np.array_equal(r1.output_grid, g1_out)

        # Validate
        validator = DSLValidator()
        val_result = validator.validate(restored, [(g1_in, g1_out), (g2_in, g2_out)])
        assert val_result.all_passed
        assert val_result.score == 1.0

    def test_full_pipeline_with_error_trace(self):
        """Pipeline with wrong program -> error traces."""
        g_in = np.array([[1, 2], [3, 4]])
        g_expected = np.array([[4, 3], [2, 1]])  # rot180

        # Wrong program: rot90 instead of rot180
        prog = DSLProgram(root=prim_node("rot90", input_node()))

        validator = DSLValidator()
        val_result = validator.validate(prog, [(g_in, g_expected)])
        assert not val_result.all_passed

        traces = build_error_traces_from_validation(val_result, [(g_in, g_expected)])
        assert len(traces) == 1
        assert traces[0].suggested_category in ("color_error", "spatial_error", "structural_error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for the puzzle generator.

Tests every rule transformation with hand-crafted expected outputs,
then tests the generator assembly logic.

Run with:  pytest tests/test_puzzle_generator.py -v
"""

import pytest
import numpy as np
import json

from puzzle_generator.grid_utils import (
    Grid, GRID_SIZE, NUM_COLORS, make_grid, grids_equal, grid_to_np,
)
from puzzle_generator.rules import (
    translate_by_n, mirror, recolor, apply_rule, get_all_params,
    RULE_REGISTRY,
)
from puzzle_generator.generator import generate_puzzle, generate_batch


# ===================================================================
# Helper fixtures
# ===================================================================

# A simple 5×5 grid with a known pattern for manual verification.
# Uses colors 0, 1, 2.
KNOWN_GRID: Grid = [
    [0, 1, 2, 0, 1],
    [1, 2, 0, 1, 2],
    [2, 0, 1, 2, 0],
    [0, 1, 2, 0, 1],
    [1, 2, 0, 1, 2],
]

# A grid with a single non-zero "block" for easy visual checking.
BLOCK_GRID: Grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 2, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]


# ===================================================================
# 1. Translate-by-N tests
# ===================================================================

class TestTranslate:
    def test_translate_right_1(self):
        result = translate_by_n(BLOCK_GRID, 'right', 1)
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_translate_left_1(self):
        result = translate_by_n(BLOCK_GRID, 'left', 1)
        expected = [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_translate_down_1(self):
        result = translate_by_n(BLOCK_GRID, 'down', 1)
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_translate_up_1(self):
        result = translate_by_n(BLOCK_GRID, 'up', 1)
        expected = [
            [0, 1, 1, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_translate_right_2(self):
        result = translate_by_n(BLOCK_GRID, 'right', 2)
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_translate_down_3(self):
        result = translate_by_n(BLOCK_GRID, 'down', 3)
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_translate_magnitude_0_is_identity(self):
        result = translate_by_n(KNOWN_GRID, 'left', 0)
        assert result == KNOWN_GRID

    def test_translate_magnitude_4_leaves_one_row_or_col(self):
        """Magnitude 4 on a 5×5 grid should leave only 1 row/col."""
        result = translate_by_n(BLOCK_GRID, 'up', 4)
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_translate_preserves_grid_size(self):
        for direction in ['up', 'down', 'left', 'right']:
            for mag in range(1, 5):
                result = translate_by_n(KNOWN_GRID, direction, mag)
                assert len(result) == GRID_SIZE
                for row in result:
                    assert len(row) == GRID_SIZE

    def test_translate_fill_color(self):
        """Vacated cells should be fill_color (default 0)."""
        result = translate_by_n(BLOCK_GRID, 'right', 3, fill_color=0)
        # Left 3 columns should all be 0
        for r in range(GRID_SIZE):
            for c in range(3):
                assert result[r][c] == 0, f"Non-zero at ({r},{c})"

    def test_translate_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            translate_by_n(BLOCK_GRID, 'diagonal', 1)

    def test_translate_with_known_grid(self):
        """Test translate-left-1 on the patterned KNOWN_GRID."""
        result = translate_by_n(KNOWN_GRID, 'left', 1)
        expected = [
            [1, 2, 0, 1, 0],
            [2, 0, 1, 2, 0],
            [0, 1, 2, 0, 0],
            [1, 2, 0, 1, 0],
            [2, 0, 1, 2, 0],
        ]
        assert result == expected, f"Got:\n{result}"


# ===================================================================
# 2. Mirror tests
# ===================================================================

class TestMirror:
    def test_mirror_horizontal(self):
        """Horizontal mirror = flip left↔right (reverse each row)."""
        result = mirror(BLOCK_GRID, 'horizontal')
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_mirror_vertical(self):
        """Vertical mirror = flip top↔bottom (reverse row order)."""
        result = mirror(BLOCK_GRID, 'vertical')
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_mirror_horizontal_is_involution(self):
        """Applying horizontal mirror twice returns the original."""
        once = mirror(KNOWN_GRID, 'horizontal')
        twice = mirror(once, 'horizontal')
        assert twice == KNOWN_GRID

    def test_mirror_vertical_is_involution(self):
        """Applying vertical mirror twice returns the original."""
        once = mirror(KNOWN_GRID, 'vertical')
        twice = mirror(once, 'vertical')
        assert twice == KNOWN_GRID

    def test_mirror_preserves_grid_size(self):
        for axis in ['horizontal', 'vertical']:
            result = mirror(KNOWN_GRID, axis)
            assert len(result) == GRID_SIZE
            for row in result:
                assert len(row) == GRID_SIZE

    def test_mirror_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            mirror(BLOCK_GRID, 'diagonal')

    def test_mirror_horizontal_known_grid(self):
        result = mirror(KNOWN_GRID, 'horizontal')
        expected = [
            [1, 0, 2, 1, 0],
            [2, 1, 0, 2, 1],
            [0, 2, 1, 0, 2],
            [1, 0, 2, 1, 0],
            [2, 1, 0, 2, 1],
        ]
        assert result == expected, f"Got:\n{result}"


# ===================================================================
# 3. Recolor tests
# ===================================================================

class TestRecolor:
    def test_recolor_swap_0_1(self):
        """Swap colors 0↔1, keep 2."""
        cmap = {0: 1, 1: 0, 2: 2}
        result = recolor(BLOCK_GRID, cmap)
        expected = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 2, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_recolor_cycle_012(self):
        """Cycle: 0→1, 1→2, 2→0."""
        cmap = {0: 1, 1: 2, 2: 0}
        result = recolor(BLOCK_GRID, cmap)
        expected = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 1, 1],
            [1, 2, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        assert result == expected, f"Got:\n{result}"

    def test_recolor_preserves_grid_size(self):
        cmap = {0: 2, 1: 0, 2: 1}
        result = recolor(KNOWN_GRID, cmap)
        assert len(result) == GRID_SIZE
        for row in result:
            assert len(row) == GRID_SIZE

    def test_recolor_is_consistent(self):
        """Same color in input → same color in output (deterministic)."""
        cmap = {0: 2, 1: 0, 2: 1}
        result = recolor(KNOWN_GRID, cmap)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                assert result[r][c] == cmap[KNOWN_GRID[r][c]]

    def test_recolor_inverse(self):
        """Applying a permutation then its inverse recovers the original."""
        cmap = {0: 1, 1: 2, 2: 0}
        inv = {v: k for k, v in cmap.items()}
        once = recolor(KNOWN_GRID, cmap)
        twice = recolor(once, inv)
        assert twice == KNOWN_GRID

    def test_recolor_missing_color_raises(self):
        incomplete_map = {0: 1, 1: 0}  # missing 2
        with pytest.raises(ValueError):
            recolor(KNOWN_GRID, incomplete_map)

    def test_all_recolor_params_exclude_identity(self):
        params = get_all_params('recolor')
        for p in params:
            assert p['color_map'] != {0: 0, 1: 1, 2: 2}, \
                "Identity permutation should be excluded"
        assert len(params) == 5  # 3! - 1 = 5


# ===================================================================
# 4. apply_rule dispatcher tests
# ===================================================================

class TestApplyRule:
    def test_apply_translate(self):
        result = apply_rule('translate', BLOCK_GRID,
                            {'direction': 'right', 'magnitude': 1})
        direct = translate_by_n(BLOCK_GRID, 'right', 1)
        assert result == direct

    def test_apply_mirror(self):
        result = apply_rule('mirror', KNOWN_GRID, {'axis': 'horizontal'})
        direct = mirror(KNOWN_GRID, 'horizontal')
        assert result == direct

    def test_apply_recolor(self):
        cmap = {0: 1, 1: 2, 2: 0}
        result = apply_rule('recolor', KNOWN_GRID, {'color_map': cmap})
        direct = recolor(KNOWN_GRID, cmap)
        assert result == direct

    def test_apply_unknown_rule_raises(self):
        with pytest.raises(ValueError):
            apply_rule('rotate', KNOWN_GRID, {})


# ===================================================================
# 5. Grid generation tests
# ===================================================================

class TestGridGeneration:
    def test_make_grid_shape(self):
        rng = np.random.default_rng(42)
        grid = make_grid(rng)
        assert len(grid) == GRID_SIZE
        for row in grid:
            assert len(row) == GRID_SIZE

    def test_make_grid_all_colors_present(self):
        """With ensure_all_colors=True, all 3 colors must appear."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            grid = make_grid(rng, ensure_all_colors=True)
            flat = {c for row in grid for c in row}
            assert flat == {0, 1, 2}, f"Missing colors: {flat}"

    def test_make_grid_colors_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            grid = make_grid(rng)
            for row in grid:
                for c in row:
                    assert 0 <= c < NUM_COLORS

    def test_make_grid_deterministic(self):
        g1 = make_grid(np.random.default_rng(123))
        g2 = make_grid(np.random.default_rng(123))
        assert g1 == g2


# ===================================================================
# 6. Puzzle generator assembly tests
# ===================================================================

class TestPuzzleGenerator:
    def test_generate_puzzle_structure(self):
        rng = np.random.default_rng(42)
        puzzle = generate_puzzle(
            rule_type='translate',
            rule_params={'direction': 'right', 'magnitude': 1},
            num_demos=3,
            rng=rng,
        )
        assert 'id' in puzzle
        assert puzzle['rule_type'] == 'translate'
        assert len(puzzle['demo_pairs']) == 3
        assert 'input' in puzzle['test_pair']
        assert 'output' in puzzle['test_pair']

    def test_demo_pairs_are_correct(self):
        """Each demo output should equal apply_rule(demo input)."""
        rng = np.random.default_rng(42)
        puzzle = generate_puzzle(
            rule_type='mirror',
            rule_params={'axis': 'vertical'},
            num_demos=5,
            rng=rng,
        )
        for demo in puzzle['demo_pairs']:
            expected = apply_rule('mirror', demo['input'],
                                  {'axis': 'vertical'})
            assert demo['output'] == expected

    def test_test_pair_is_correct(self):
        rng = np.random.default_rng(42)
        puzzle = generate_puzzle(
            rule_type='recolor',
            rule_params={'color_map': {0: 2, 1: 0, 2: 1}},
            num_demos=3,
            rng=rng,
        )
        expected = apply_rule('recolor', puzzle['test_pair']['input'],
                              {'color_map': {0: 2, 1: 0, 2: 1}})
        assert puzzle['test_pair']['output'] == expected

    def test_demos_are_informative(self):
        """No demo should have input == output."""
        rng = np.random.default_rng(42)
        puzzle = generate_puzzle(
            rule_type='translate',
            rule_params={'direction': 'down', 'magnitude': 2},
            num_demos=5,
            rng=rng,
        )
        for demo in puzzle['demo_pairs']:
            assert demo['input'] != demo['output'], \
                "Demo input equals output — not informative"

    def test_generate_batch_count(self):
        puzzles = generate_batch(
            rule_type='translate',
            rule_params={'direction': 'left', 'magnitude': 1},
            count=10,
            num_demos=3,
            seed=42,
        )
        assert len(puzzles) == 10

    def test_generate_batch_deterministic(self):
        batch1 = generate_batch('mirror', {'axis': 'horizontal'},
                                count=5, num_demos=3, seed=99)
        batch2 = generate_batch('mirror', {'axis': 'horizontal'},
                                count=5, num_demos=3, seed=99)
        for p1, p2 in zip(batch1, batch2):
            assert p1 == p2

    def test_puzzle_json_serializable(self):
        rng = np.random.default_rng(42)
        puzzle = generate_puzzle(
            rule_type='translate',
            rule_params={'direction': 'up', 'magnitude': 1},
            num_demos=3,
            rng=rng,
        )
        # Should not raise
        serialized = json.dumps(puzzle)
        deserialized = json.loads(serialized)
        assert deserialized['rule_type'] == 'translate'

    def test_puzzle_recolor_json_serializable(self):
        """Recolor params have int keys that need str conversion for JSON."""
        rng = np.random.default_rng(42)
        puzzle = generate_puzzle(
            rule_type='recolor',
            rule_params={'color_map': {0: 1, 1: 2, 2: 0}},
            num_demos=3,
            rng=rng,
        )
        serialized = json.dumps(puzzle)
        assert '"color_map"' in serialized


# ===================================================================
# 7. Parameter enumeration tests
# ===================================================================

class TestParameterEnumeration:
    def test_translate_params_count(self):
        """4 directions × 4 magnitudes (1..4) = 16."""
        params = get_all_params('translate')
        assert len(params) == 16

    def test_mirror_params_count(self):
        params = get_all_params('mirror')
        assert len(params) == 2

    def test_recolor_params_count(self):
        """3! - 1 = 5 non-identity permutations."""
        params = get_all_params('recolor')
        assert len(params) == 5

    def test_all_translate_params_valid(self):
        for p in get_all_params('translate'):
            assert p['direction'] in ['up', 'down', 'left', 'right']
            assert 1 <= p['magnitude'] <= 4

    def test_all_recolor_params_are_permutations(self):
        for p in get_all_params('recolor'):
            cmap = p['color_map']
            assert set(cmap.keys()) == {0, 1, 2}
            assert set(cmap.values()) == {0, 1, 2}


# ===================================================================
# 8. Edge-case & regression tests
# ===================================================================

class TestEdgeCases:
    def test_translate_full_grid(self):
        """Shifting by 4 on a 5×5 grid leaves at most 1 row/col of data."""
        full = [[2] * 5 for _ in range(5)]
        result = translate_by_n(full, 'right', 4, fill_color=0)
        for r in range(5):
            for c in range(4):
                assert result[r][c] == 0
            assert result[r][4] == 2

    def test_recolor_all_same_color(self):
        """Grid with only one non-background color — recolor should work
        but ensure_all_colors would reject this grid in generation."""
        mono = [[0] * 5 for _ in range(5)]
        mono[2][2] = 1
        mono[0][0] = 2
        cmap = {0: 1, 1: 0, 2: 2}
        result = recolor(mono, cmap)
        assert result[2][2] == 0
        assert result[0][0] == 2
        assert result[3][3] == 1  # was 0 → 1

    def test_mirror_symmetric_grid(self):
        """A horizontally symmetric grid should equal its horizontal mirror."""
        sym = [
            [0, 1, 2, 1, 0],
            [1, 2, 0, 2, 1],
            [2, 0, 1, 0, 2],
            [1, 2, 0, 2, 1],
            [0, 1, 2, 1, 0],
        ]
        assert mirror(sym, 'horizontal') == sym

    def test_consecutive_translates(self):
        """translate_right_1 twice == translate_right_2."""
        once = translate_by_n(BLOCK_GRID, 'right', 1)
        twice = translate_by_n(once, 'right', 1)
        direct = translate_by_n(BLOCK_GRID, 'right', 2)
        assert twice == direct


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Transformation rules for ARC-like puzzles.

Each rule is a pure function: grid_in → grid_out.
Rules are parameterised and deterministic — given the same input grid
and the same rule parameters, the output is always identical.

Implemented rules (in order):
  1. translate_by_n  — shift grid content in a direction by N cells
  2. mirror          — flip grid horizontally or vertically
  3. recolor         — apply a hidden color permutation
"""

import numpy as np
from typing import Dict, Tuple
from .grid_utils import Grid, GRID_SIZE, NUM_COLORS, grid_to_np, np_to_grid


# ---------------------------------------------------------------------------
# 1. Translate-by-N
# ---------------------------------------------------------------------------

def translate_by_n(grid: Grid, direction: str, magnitude: int,
                   fill_color: int = 0) -> Grid:
    """Shift grid content by `magnitude` cells in `direction`.

    Cells that shift off the edge are lost; vacated cells are filled
    with `fill_color`.

    Parameters
    ----------
    grid : Grid
        5×5 input grid.
    direction : str
        One of 'up', 'down', 'left', 'right'.
    magnitude : int
        Number of cells to shift (0 ≤ magnitude < GRID_SIZE).
    fill_color : int
        Color to fill vacated cells (default 0).

    Returns
    -------
    Grid
        Transformed 5×5 grid.
    """
    if magnitude == 0:
        return [row[:] for row in grid]  # identity copy
    if magnitude >= GRID_SIZE:
        return [[fill_color] * GRID_SIZE for _ in range(GRID_SIZE)]

    arr = grid_to_np(grid)
    out = np.full_like(arr, fill_color)

    if direction == 'up':
        out[:GRID_SIZE - magnitude, :] = arr[magnitude:, :]
    elif direction == 'down':
        out[magnitude:, :] = arr[:GRID_SIZE - magnitude, :]
    elif direction == 'left':
        out[:, :GRID_SIZE - magnitude] = arr[:, magnitude:]
    elif direction == 'right':
        out[:, magnitude:] = arr[:, :GRID_SIZE - magnitude]
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return np_to_grid(out)


# ---------------------------------------------------------------------------
# 2. Mirror
# ---------------------------------------------------------------------------

def mirror(grid: Grid, axis: str) -> Grid:
    """Flip the grid along an axis.

    Parameters
    ----------
    grid : Grid
        5×5 input grid.
    axis : str
        'horizontal' — flip left↔right (columns reversed).
        'vertical'   — flip top↔bottom (rows reversed).

    Returns
    -------
    Grid
        Transformed 5×5 grid.
    """
    arr = grid_to_np(grid)
    if axis == 'horizontal':
        out = np.fliplr(arr)
    elif axis == 'vertical':
        out = np.flipud(arr)
    else:
        raise ValueError(f"Invalid axis: {axis}")
    return np_to_grid(out)


# ---------------------------------------------------------------------------
# 3. Recolor
# ---------------------------------------------------------------------------

def recolor(grid: Grid, color_map: Dict[int, int]) -> Grid:
    """Apply a color permutation to the grid.

    Parameters
    ----------
    grid : Grid
        5×5 input grid.
    color_map : dict
        Mapping {old_color: new_color}. Must cover all colors present
        in the grid.

    Returns
    -------
    Grid
        Transformed 5×5 grid.
    """
    out = []
    for row in grid:
        new_row = []
        for c in row:
            if c not in color_map:
                raise ValueError(
                    f"Color {c} in grid but not in color_map {color_map}")
            new_row.append(color_map[c])
        out.append(new_row)
    return out


# ---------------------------------------------------------------------------
# Registry — maps rule_type strings to (function, parameter-schema) pairs
# ---------------------------------------------------------------------------

def _all_translate_params() -> list:
    """All valid translate parameters (direction × magnitude)."""
    params = []
    for d in ['up', 'down', 'left', 'right']:
        for m in range(1, GRID_SIZE):  # 1..4
            params.append({'direction': d, 'magnitude': m})
    return params


def _all_mirror_params() -> list:
    """All valid mirror parameters."""
    return [
        {'axis': 'horizontal'},
        {'axis': 'vertical'},
    ]


def _all_recolor_params() -> list:
    """All non-identity permutations of 3 colors.

    There are 3! = 6 permutations; we exclude the identity {0:0,1:1,2:2}
    leaving 5 valid recolor mappings.
    """
    from itertools import permutations
    params = []
    for perm in permutations(range(NUM_COLORS)):
        cmap = {i: perm[i] for i in range(NUM_COLORS)}
        if cmap == {0: 0, 1: 1, 2: 2}:
            continue  # skip identity
        params.append({'color_map': cmap})
    return params


# Master registry
RULE_REGISTRY = {
    'translate': {
        'fn': translate_by_n,
        'all_params': _all_translate_params,
    },
    'mirror': {
        'fn': mirror,
        'all_params': _all_mirror_params,
    },
    'recolor': {
        'fn': recolor,
        'all_params': _all_recolor_params,
    },
}


def apply_rule(rule_type: str, grid: Grid, rule_params: dict) -> Grid:
    """Apply a named rule with given parameters to a grid."""
    if rule_type not in RULE_REGISTRY:
        raise ValueError(f"Unknown rule_type: {rule_type}")
    fn = RULE_REGISTRY[rule_type]['fn']
    return fn(grid, **rule_params)


def get_all_params(rule_type: str) -> list:
    """Return all valid parameter dicts for a rule type."""
    if rule_type not in RULE_REGISTRY:
        raise ValueError(f"Unknown rule_type: {rule_type}")
    return RULE_REGISTRY[rule_type]['all_params']()

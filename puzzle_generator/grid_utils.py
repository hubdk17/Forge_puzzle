"""
Core grid utilities for 5×5 ARC-like puzzle generation.

All grids are represented as list-of-lists of ints, shape (5, 5).
Colors are integers in {0, 1, 2} (exactly 3 colors).
"""

import numpy as np
from typing import List

Grid = List[List[int]]

GRID_SIZE = 5
NUM_COLORS = 3
COLORS = list(range(NUM_COLORS))  # [0, 1, 2]


def make_grid(rng: np.random.Generator, ensure_all_colors: bool = True) -> Grid:
    """Generate a random 5×5 grid with exactly 3 colors.

    If ensure_all_colors is True, the grid is guaranteed to contain
    all 3 colors (at least one cell of each). This prevents degenerate
    cases where a color mapping is unobservable.
    """
    while True:
        grid = rng.integers(0, NUM_COLORS, size=(GRID_SIZE, GRID_SIZE)).tolist()
        if not ensure_all_colors:
            return grid
        flat = [c for row in grid for c in row]
        if len(set(flat)) == NUM_COLORS:
            return grid


def grid_to_np(grid: Grid) -> np.ndarray:
    """Convert list-of-lists grid to numpy array."""
    return np.array(grid, dtype=np.int64)


def np_to_grid(arr: np.ndarray) -> Grid:
    """Convert numpy array to list-of-lists grid."""
    return arr.tolist()


def grids_equal(a: Grid, b: Grid) -> bool:
    """Check exact equality of two grids."""
    return a == b

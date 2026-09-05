"""
Puzzle generator: assembles complete ARC-like puzzle instances.

Each puzzle contains:
  - id             : unique string identifier
  - rule_type      : 'translate' | 'mirror' | 'recolor'
  - rule_params    : dict of parameters for the rule
  - demo_pairs     : list of (input_grid, output_grid) demonstration pairs
  - test_pair      : dict with 'input' and 'output' grids
"""

import hashlib
import json
import numpy as np
from typing import List, Dict, Any, Optional

from .grid_utils import Grid, make_grid, grids_equal
from .rules import apply_rule, get_all_params, RULE_REGISTRY


def _make_id(rule_type: str, rule_params: dict, index: int) -> str:
    """Create a deterministic, human-readable puzzle ID."""
    # Compact param string for readability
    param_str = json.dumps(rule_params, sort_keys=True, separators=(',', ':'))
    raw = f"{rule_type}|{param_str}|{index}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{rule_type}_{short_hash}_{index}"


def _is_informative_demo(input_grid: Grid, output_grid: Grid) -> bool:
    """Check that a demonstration pair is non-trivial.

    A demo is informative if input ≠ output (the transformation actually
    changes something). For recolor, we also want at least 2 colors present
    so the mapping is partially observable.
    """
    return not grids_equal(input_grid, output_grid)


def generate_puzzle(
    rule_type: str,
    rule_params: dict,
    num_demos: int,
    rng: np.random.Generator,
    index: int = 0,
    max_attempts: int = 100,
) -> Dict[str, Any]:
    """Generate a single puzzle instance.

    Parameters
    ----------
    rule_type : str
        One of 'translate', 'mirror', 'recolor'.
    rule_params : dict
        Parameters for the rule (e.g. {'direction': 'up', 'magnitude': 2}).
    num_demos : int
        Number of demonstration pairs to include.
    rng : np.random.Generator
        Random number generator (for reproducibility).
    index : int
        Index for ID generation.
    max_attempts : int
        Max tries to find informative grids.

    Returns
    -------
    dict
        Complete puzzle instance.
    """
    puzzle_id = _make_id(rule_type, rule_params, index)

    demo_pairs = []
    attempts = 0
    while len(demo_pairs) < num_demos and attempts < max_attempts:
        inp = make_grid(rng, ensure_all_colors=True)
        out = apply_rule(rule_type, inp, rule_params)
        if _is_informative_demo(inp, out):
            demo_pairs.append({'input': inp, 'output': out})
        attempts += 1

    if len(demo_pairs) < num_demos:
        raise RuntimeError(
            f"Could not generate {num_demos} informative demos for "
            f"{rule_type} {rule_params} after {max_attempts} attempts"
        )

    # Test pair — must also be informative and use a fresh grid
    test_inp = None
    test_out = None
    for _ in range(max_attempts):
        candidate = make_grid(rng, ensure_all_colors=True)
        candidate_out = apply_rule(rule_type, candidate, rule_params)
        if _is_informative_demo(candidate, candidate_out):
            test_inp = candidate
            test_out = candidate_out
            break
    if test_inp is None:
        raise RuntimeError(
            f"Could not generate an informative test pair for "
            f"{rule_type} {rule_params}"
        )

    # Serialisable rule_params (convert int keys to str for JSON compat)
    serial_params = {}
    for k, v in rule_params.items():
        if isinstance(v, dict):
            serial_params[k] = {str(kk): int(vv) for kk, vv in v.items()}
        else:
            serial_params[k] = v if not isinstance(v, np.integer) else int(v)

    return {
        'id': puzzle_id,
        'rule_type': rule_type,
        'rule_params': serial_params,
        'demo_pairs': demo_pairs,
        'test_pair': {
            'input': test_inp,
            'output': test_out,
        },
    }


def generate_batch(
    rule_type: str,
    rule_params: dict,
    count: int,
    num_demos: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate `count` puzzle instances for a given rule+params.

    Each instance uses a deterministic sub-seed for reproducibility.
    """
    puzzles = []
    for i in range(count):
        rng = np.random.default_rng(seed + i)
        puzzle = generate_puzzle(
            rule_type=rule_type,
            rule_params=rule_params,
            num_demos=num_demos,
            rng=rng,
            index=i,
        )
        puzzles.append(puzzle)
    return puzzles

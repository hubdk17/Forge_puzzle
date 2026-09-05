"""
Shared data loading utilities for both context and optimization models.

Handles JSONL puzzle loading, grid encoding, and batching.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

GRID_SIZE = 5
NUM_COLORS = 3
GRID_CELLS = GRID_SIZE * GRID_SIZE  # 25
GRID_FEATURES = GRID_CELLS * NUM_COLORS  # 75 (one-hot)


def encode_grid(grid: List[List[int]]) -> torch.Tensor:
    """One-hot encode a 5x5 grid → flat tensor of shape (75,).

    Each cell is one-hot encoded over 3 colors, then flattened.
    """
    t = torch.zeros(GRID_CELLS, NUM_COLORS)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            t[r * GRID_SIZE + c, grid[r][c]] = 1.0
    return t.flatten()  # (75,)


def decode_grid(logits: torch.Tensor) -> List[List[int]]:
    """Convert (25, 3) logits → 5x5 grid of predicted colors."""
    if logits.dim() == 1:
        logits = logits.view(GRID_CELLS, NUM_COLORS)
    preds = logits.argmax(dim=-1)  # (25,)
    grid = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            row.append(preds[r * GRID_SIZE + c].item())
        grid.append(row)
    return grid


def encode_demo_pair(demo: Dict) -> torch.Tensor:
    """Encode a demo pair (input, output) → tensor of shape (150,).

    Concatenates one-hot encoded input and output grids.
    """
    inp_enc = encode_grid(demo['input'])
    out_enc = encode_grid(demo['output'])
    return torch.cat([inp_enc, out_enc])  # (150,)


def load_puzzles(filepath: str) -> List[Dict[str, Any]]:
    """Load puzzles from a JSONL file."""
    puzzles = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                puzzles.append(json.loads(line))
    return puzzles


def puzzle_to_tensors(puzzle: Dict, num_demos: Optional[int] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a puzzle dict to tensors.

    Parameters
    ----------
    puzzle : dict
        Puzzle instance from JSONL.
    num_demos : int or None
        If set, use only the first `num_demos` demonstration pairs.
        If None, use all available demos.

    Returns
    -------
    demo_pairs : Tensor of shape (D, 150)
        Encoded demonstration pairs.
    test_input : Tensor of shape (75,)
        Encoded test input grid.
    test_target : Tensor of shape (25,) int64
        Target color index per cell.
    """
    demos = puzzle['demo_pairs']
    if num_demos is not None:
        demos = demos[:num_demos]

    demo_tensors = torch.stack([encode_demo_pair(d) for d in demos])  # (D, 150)

    test_input = encode_grid(puzzle['test_pair']['input'])  # (75,)

    # Target: flat list of color indices for cross-entropy
    target_grid = puzzle['test_pair']['output']
    target = torch.zeros(GRID_CELLS, dtype=torch.long)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            target[r * GRID_SIZE + c] = target_grid[r][c]

    return demo_tensors, test_input, target


class PuzzleDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for puzzle instances.

    Precomputes all tensors on initialization for speed.
    """

    def __init__(self, filepath: str, num_demos: Optional[int] = None):
        self.puzzles = load_puzzles(filepath)
        self.num_demos = num_demos

        # Precompute tensors
        self.data = []
        for p in self.puzzles:
            demos, test_in, target = puzzle_to_tensors(p, num_demos)
            self.data.append((demos, test_in, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_puzzle(self, idx):
        """Return the raw puzzle dict."""
        return self.puzzles[idx]


def collate_puzzles(batch):
    """Custom collate for variable-length demo sequences.

    Returns:
        demo_pairs: list of tensors, each (D_i, 150)
        test_inputs: Tensor (B, 75)
        targets: Tensor (B, 25)
    """
    demo_pairs = [item[0] for item in batch]
    test_inputs = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    return demo_pairs, test_inputs, targets

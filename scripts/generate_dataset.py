"""
Dataset generation with train/validation/test/novelty splits.

Split strategy (explicitly documented):

TRANSLATE (16 total param combos: 4 directions x 4 magnitudes 1-4)
  Train params:   {up,down,left,right} x {1, 2}     = 8 combos
  Novelty params: {up,down,left,right} x {3, 4}     = 8 combos
  Rationale:      magnitudes 3 and 4 are NEVER seen during training.
                  The model must generalise to unseen shift distances.

MIRROR (2 total param combos: horizontal, vertical)
  Train params:   {horizontal, vertical}             = 2 combos
  Novelty params: none (only 2 params; both needed for training)
  Rationale:      mirror has too few params for meaningful novelty split.

RECOLOR (5 non-identity permutations of 3 colours)
  Train params:   {0->1,1->0,2->2}, {0->1,1->2,2->0}, {0->2,1->1,2->0}  = 3 combos
  Novelty params: {0->0,1->2,2->1}, {0->2,1->0,2->1}                     = 2 combos
  Rationale:      2 colour permutations are NEVER seen during training.
                  The model must infer an unseen mapping from demos.

SPLIT RATIOS (within train-eligible params):
  Train:       70%
  Validation:  15%
  Test:        15%

LEAKAGE PREVENTION:
  - Novelty params are defined at the parameter level, not the instance level.
  - No novelty-param instance ever appears in train/val/test.
  - Train/val/test splits are instance-level (same params, different grids).
  - Each split uses non-overlapping random seeds.

Run: python -m scripts.generate_dataset [--num_per_param 350] [--num_demos 5] [--seed 42]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puzzle_generator.generator import generate_batch
from puzzle_generator.rules import get_all_params


# -----------------------------------------------------------------------
# Split definitions: which params are train vs novelty
# -----------------------------------------------------------------------

def _get_train_novelty_params():
    """Return (train_params, novelty_params) for each rule type.

    Returns dict: rule_type -> {'train': [...], 'novelty': [...]}
    """
    splits = {}

    # --- TRANSLATE ---
    all_translate = get_all_params('translate')
    train_translate = [p for p in all_translate
                       if p['magnitude'] in (1, 2)]
    novelty_translate = [p for p in all_translate
                         if p['magnitude'] in (3, 4)]
    splits['translate'] = {'train': train_translate, 'novelty': novelty_translate}

    # --- MIRROR ---
    all_mirror = get_all_params('mirror')
    splits['mirror'] = {'train': all_mirror, 'novelty': []}

    # --- RECOLOR ---
    all_recolor = get_all_params('recolor')
    # Sort for deterministic ordering
    sorted_recolor = sorted(all_recolor,
                            key=lambda p: str(sorted(p['color_map'].items())))

    # First 3 → train, last 2 → novelty
    train_recolor = sorted_recolor[:3]
    novelty_recolor = sorted_recolor[3:]
    splits['recolor'] = {'train': train_recolor, 'novelty': novelty_recolor}

    return splits


def _split_instances(puzzles, train_frac=0.70, val_frac=0.15, seed=0):
    """Split a list of puzzles into train/val/test by instance.

    Does NOT modify input. Returns (train, val, test) lists.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(puzzles))

    n_train = int(len(puzzles) * train_frac)
    n_val = int(len(puzzles) * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [puzzles[i] for i in train_idx]
    val = [puzzles[i] for i in val_idx]
    test = [puzzles[i] for i in test_idx]

    return train, val, test


def _save_jsonl(puzzles, filepath):
    """Save puzzle list as JSONL."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for p in puzzles:
            f.write(json.dumps(p, separators=(',', ':')) + '\n')
    return len(puzzles)


def generate_dataset(num_per_param=350, num_demos=5, seed=42, data_dir='data'):
    """Generate the full dataset with all splits.

    Parameters
    ----------
    num_per_param : int
        Number of puzzle instances per (rule_type, param) combo.
        With 8 train translate params * 350 = 2800 translate train instances.
    num_demos : int
        Max number of demonstration pairs per puzzle.
    seed : int
        Base random seed.
    data_dir : str
        Output directory.
    """
    splits_def = _get_train_novelty_params()

    all_train = []
    all_val = []
    all_test = []
    all_novelty = []

    stats = {}

    for rule_type, param_split in splits_def.items():
        train_params = param_split['train']
        novelty_params = param_split['novelty']

        # Generate train-eligible instances
        rule_train_pool = []
        for pi, params in enumerate(train_params):
            # Use deterministic seed: base_seed + rule_offset + param_index * 100000
            rule_offset = {'translate': 0, 'mirror': 1_000_000,
                           'recolor': 2_000_000}[rule_type]
            batch_seed = seed + rule_offset + pi * 100_000
            batch = generate_batch(
                rule_type=rule_type,
                rule_params=params,
                count=num_per_param,
                num_demos=num_demos,
                seed=batch_seed,
            )
            rule_train_pool.extend(batch)

        # Split into train/val/test
        split_seed = seed + {'translate': 10, 'mirror': 20, 'recolor': 30}[rule_type]
        train, val, test = _split_instances(rule_train_pool, seed=split_seed)

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

        # Generate novelty instances
        rule_novelty = []
        for pi, params in enumerate(novelty_params):
            rule_offset = {'translate': 3_000_000, 'mirror': 4_000_000,
                           'recolor': 5_000_000}[rule_type]
            batch_seed = seed + rule_offset + pi * 100_000
            batch = generate_batch(
                rule_type=rule_type,
                rule_params=params,
                count=num_per_param,
                num_demos=num_demos,
                seed=batch_seed,
            )
            rule_novelty.extend(batch)

        all_novelty.extend(rule_novelty)

        stats[rule_type] = {
            'train_params': len(train_params),
            'novelty_params': len(novelty_params),
            'train_instances': len(train),
            'val_instances': len(val),
            'test_instances': len(test),
            'novelty_instances': len(rule_novelty),
        }

    # Save splits
    counts = {}
    counts['train'] = _save_jsonl(all_train, os.path.join(data_dir, 'train.jsonl'))
    counts['validation'] = _save_jsonl(all_val, os.path.join(data_dir, 'validation.jsonl'))
    counts['test'] = _save_jsonl(all_test, os.path.join(data_dir, 'test.jsonl'))
    counts['novelty'] = _save_jsonl(all_novelty, os.path.join(data_dir, 'novelty.jsonl'))

    # Save split metadata
    metadata = {
        'seed': seed,
        'num_per_param': num_per_param,
        'num_demos': num_demos,
        'split_ratios': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
        'per_rule_stats': stats,
        'total_counts': counts,
        'novelty_definition': {
            'translate': 'magnitudes 3 and 4 (train uses only 1 and 2)',
            'mirror': 'no novelty params (only 2 total params)',
            'recolor': 'last 2 of 5 non-identity permutations held out',
        },
        'train_params': {
            rt: [str(p) for p in ps['train']]
            for rt, ps in splits_def.items()
        },
        'novelty_params': {
            rt: [str(p) for p in ps['novelty']]
            for rt, ps in splits_def.items()
        },
    }
    meta_path = os.path.join(data_dir, 'split_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate puzzle dataset')
    parser.add_argument('--num_per_param', type=int, default=350,
                        help='Instances per (rule, param) combo (default: 350)')
    parser.add_argument('--num_demos', type=int, default=5,
                        help='Demo pairs per puzzle (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Output directory (default: data)')
    args = parser.parse_args()

    print(f"Generating dataset: {args.num_per_param} instances/param, "
          f"seed={args.seed}")

    metadata = generate_dataset(
        num_per_param=args.num_per_param,
        num_demos=args.num_demos,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    print(f"\nDataset saved to {args.data_dir}/")
    print(f"  train.jsonl:      {metadata['total_counts']['train']} instances")
    print(f"  validation.jsonl: {metadata['total_counts']['validation']} instances")
    print(f"  test.jsonl:       {metadata['total_counts']['test']} instances")
    print(f"  novelty.jsonl:    {metadata['total_counts']['novelty']} instances")
    print(f"\nPer-rule breakdown:")
    for rt, st in metadata['per_rule_stats'].items():
        print(f"  {rt}:")
        print(f"    train params:     {st['train_params']}")
        print(f"    novelty params:   {st['novelty_params']}")
        print(f"    train instances:  {st['train_instances']}")
        print(f"    val instances:    {st['val_instances']}")
        print(f"    test instances:   {st['test_instances']}")
        print(f"    novelty instances:{st['novelty_instances']}")
    print(f"\nNovelty definition:")
    for rt, defn in metadata['novelty_definition'].items():
        print(f"  {rt}: {defn}")


if __name__ == '__main__':
    main()

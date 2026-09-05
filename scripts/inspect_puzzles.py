"""
Phase 2: Generate a small set of puzzles and print them for manual inspection.

Run: python -m scripts.inspect_puzzles
"""

import json
import numpy as np
from puzzle_generator.generator import generate_puzzle
from puzzle_generator.rules import apply_rule

COLORS = {0: '.', 1: '#', 2: '@'}


def print_grid(grid, indent=4):
    """Pretty-print a 5×5 grid using ASCII symbols."""
    prefix = ' ' * indent
    for row in grid:
        print(prefix + ' '.join(COLORS.get(c, '?') for c in row))


def print_puzzle(puzzle, verify=True):
    """Print a puzzle with all its demos and test, optionally verify correctness."""
    print(f"\n{'='*60}")
    print(f"ID:     {puzzle['id']}")
    print(f"Rule:   {puzzle['rule_type']}")
    print(f"Params: {json.dumps(puzzle['rule_params'])}")
    print(f"{'='*60}")

    for i, demo in enumerate(puzzle['demo_pairs']):
        print(f"\n  Demo {i+1}:")
        print(f"    Input:              Output:")
        for r in range(5):
            inp_row = ' '.join(COLORS.get(c, '?') for c in demo['input'][r])
            out_row = ' '.join(COLORS.get(c, '?') for c in demo['output'][r])
            print(f"    {inp_row}    ->    {out_row}")

        if verify:
            # Reconstruct params for apply_rule (recolor needs int keys)
            params = dict(puzzle['rule_params'])
            if 'color_map' in params:
                params['color_map'] = {int(k): v for k, v in params['color_map'].items()}
            expected = apply_rule(puzzle['rule_type'], demo['input'], params)
            ok = demo['output'] == expected
            print(f"    Verified: {'OK CORRECT' if ok else 'XX MISMATCH'}")

    print(f"\n  Test:")
    print(f"    Input:              Expected Output:")
    tp = puzzle['test_pair']
    for r in range(5):
        inp_row = ' '.join(COLORS.get(c, '?') for c in tp['input'][r])
        out_row = ' '.join(COLORS.get(c, '?') for c in tp['output'][r])
        print(f"    {inp_row}    ->    {out_row}")

    if verify:
        params = dict(puzzle['rule_params'])
        if 'color_map' in params:
            params['color_map'] = {int(k): v for k, v in params['color_map'].items()}
        expected = apply_rule(puzzle['rule_type'], tp['input'], params)
        ok = tp['output'] == expected
        print(f"    Verified: {'OK CORRECT' if ok else 'XX MISMATCH'}")


def main():
    print("Phase 2: Manual Puzzle Inspection")
    print("Legend:  . = color 0,  # = color 1,  @ = color 2\n")

    test_cases = [
        ('translate', {'direction': 'right', 'magnitude': 1}),
        ('translate', {'direction': 'up', 'magnitude': 2}),
        ('translate', {'direction': 'left', 'magnitude': 3}),
        ('mirror', {'axis': 'horizontal'}),
        ('mirror', {'axis': 'vertical'}),
        ('recolor', {'color_map': {0: 1, 1: 0, 2: 2}}),
        ('recolor', {'color_map': {0: 2, 1: 0, 2: 1}}),
    ]

    all_correct = True
    for rule_type, params in test_cases:
        rng = np.random.default_rng(42)
        puzzle = generate_puzzle(
            rule_type=rule_type,
            rule_params=params,
            num_demos=3,
            rng=rng,
        )
        print_puzzle(puzzle, verify=True)

        # Check correctness
        for demo in puzzle['demo_pairs']:
            p = dict(params)
            if 'color_map' in p:
                p['color_map'] = {int(k): v for k, v in p['color_map'].items()}
            if demo['output'] != apply_rule(rule_type, demo['input'], p):
                all_correct = False
        p = dict(params)
        if 'color_map' in p:
            p['color_map'] = {int(k): v for k, v in p['color_map'].items()}
        if puzzle['test_pair']['output'] != apply_rule(rule_type, puzzle['test_pair']['input'], p):
            all_correct = False

    print(f"\n{'='*60}")
    if all_correct:
        print("ALL PUZZLES VERIFIED CORRECT [OK]")
    else:
        print("SOME PUZZLES HAVE ERRORS [FAIL]")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

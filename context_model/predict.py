"""
Context model prediction script.

Accepts a puzzle JSON and returns prediction with state inspection info.

Usage:
    python -m context_model.predict --puzzle puzzle.json
    python -m context_model.predict --puzzle_jsonl data/test.jsonl --limit 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    encode_grid, encode_demo_pair, decode_grid,
    puzzle_to_tensors, GRID_CELLS, NUM_COLORS
)
from context_model.model import ContextModel


def load_model(checkpoint_path, device=None):
    """Load a trained context model from checkpoint."""
    if device is None:
        device = torch.device('cpu')

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']

    model = ContextModel(
        state_dim=args['state_dim'],
        hidden_dim=args['hidden_dim'],
        num_gru_layers=args.get('num_gru_layers', 1),
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, args


def predict_puzzle(model, puzzle, num_demos=None, device=None):
    """Run prediction on a single puzzle with full state inspection.

    Parameters
    ----------
    model : ContextModel
        Trained model (must be in eval mode).
    puzzle : dict
        Puzzle instance.
    num_demos : int or None
        Number of demos to use. None = all.
    device : torch.device

    Returns
    -------
    dict with prediction, state_snapshot, confidence, latency_ms,
    and per-demo state info.
    """
    if device is None:
        device = next(model.parameters()).device

    demos, test_input, target = puzzle_to_tensors(puzzle, num_demos)
    demos = demos.to(device)
    test_input = test_input.to(device)

    # --- INFERENCE: No backward(), no optimizer, no parameter updates ---
    with torch.no_grad():
        t0 = time.perf_counter()
        logits, states = model.forward_with_state_trace(demos, test_input)
        latency_ms = (time.perf_counter() - t0) * 1000.0

    # Decode prediction
    prediction = decode_grid(logits)

    # Confidence: average of max softmax probability per cell
    probs = torch.softmax(logits, dim=-1)  # (25, 3)
    max_probs = probs.max(dim=-1).values  # (25,)
    confidence = max_probs.mean().item()

    # State snapshots
    state_snapshots = []
    for i, s in enumerate(states):
        state_snapshots.append({
            'demo_index': i,
            'state_norm': s.norm().item(),
            'state_dimension': s.shape[0],
            'state_vector': s.tolist(),
        })

    # State delta info
    state_deltas = []
    for i in range(1, len(states)):
        delta = (states[i] - states[i-1]).norm().item()
        state_deltas.append({
            'from_demo': i - 1,
            'to_demo': i,
            'delta_norm': delta,
        })

    result = {
        'prediction': prediction,
        'state_snapshot': states[-1].tolist() if states else [],
        'confidence': round(confidence, 4),
        'latency_ms': round(latency_ms, 2),
        'state_dimension': model.state_dim,
        'num_demos_used': demos.shape[0],
        'per_demo_states': state_snapshots,
        'state_deltas': state_deltas,
        'final_state_norm': states[-1].norm().item() if states else 0.0,
    }

    # Check correctness if test output is available
    if 'output' in puzzle.get('test_pair', {}):
        ground_truth = puzzle['test_pair']['output']
        result['ground_truth'] = ground_truth
        result['correct'] = (prediction == ground_truth)

    return result


def main():
    parser = argparse.ArgumentParser(description='Context model prediction')
    parser.add_argument('--checkpoint', type=str,
                        default='context_model/checkpoint.pt')
    parser.add_argument('--puzzle', type=str, default=None,
                        help='Path to single puzzle JSON file')
    parser.add_argument('--puzzle_jsonl', type=str, default=None,
                        help='Path to JSONL file with puzzles')
    parser.add_argument('--limit', type=int, default=5,
                        help='Max puzzles to process from JSONL')
    parser.add_argument('--num_demos', type=int, default=None,
                        help='Number of demos to use (None=all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: stdout)')
    args = parser.parse_args()

    device = torch.device('cpu')
    model, model_args = load_model(args.checkpoint, device)
    print(f"Loaded model: state_dim={model_args['state_dim']}, "
          f"hidden_dim={model_args['hidden_dim']}", file=sys.stderr)

    results = []

    if args.puzzle:
        with open(args.puzzle) as f:
            puzzle = json.load(f)
        result = predict_puzzle(model, puzzle, args.num_demos, device)
        results.append(result)

    elif args.puzzle_jsonl:
        with open(args.puzzle_jsonl) as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                puzzle = json.loads(line.strip())
                result = predict_puzzle(model, puzzle, args.num_demos, device)
                result['puzzle_id'] = puzzle.get('id', f'puzzle_{i}')
                results.append(result)

    # Output
    output_str = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_str)

    # Summary to stderr
    if results:
        n_correct = sum(1 for r in results if r.get('correct', False))
        print(f"\nSummary: {n_correct}/{len(results)} correct "
              f"({n_correct/len(results)*100:.1f}%)", file=sys.stderr)
        avg_latency = sum(r['latency_ms'] for r in results) / len(results)
        print(f"Avg latency: {avg_latency:.2f}ms", file=sys.stderr)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Avg confidence: {avg_confidence:.4f}", file=sys.stderr)


if __name__ == '__main__':
    main()

"""
Optimization model prediction script.

At inference:
  demo -> forward -> loss -> backward -> parameter update -> repeat K times
  then: test input -> prediction

Usage:
    python -m optimization_model.predict --puzzle puzzle.json --K 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    decode_grid, puzzle_to_tensors, GRID_CELLS, NUM_COLORS
)
from optimization_model.model import OptimizationModel


def load_model(checkpoint_path, device=None):
    """Load a trained optimization model from checkpoint."""
    if device is None:
        device = torch.device('cpu')

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']

    model = OptimizationModel(
        hidden_dim=args['hidden_dim'],
        num_layers=args.get('num_layers', 3),
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    return model, args


def predict_puzzle(model, puzzle, K=5, inner_lr=0.01,
                   num_demos=None, device=None):
    """Run inference-time adaptation and prediction.

    Parameters
    ----------
    model : OptimizationModel
    puzzle : dict
    K : int
        Number of gradient steps at inference.
    inner_lr : float
    num_demos : int or None
    device : torch.device

    Returns
    -------
    dict with prediction, loss_curve, confidence, latency_ms, etc.
    """
    if device is None:
        device = next(model.parameters()).device

    demos, test_input, target = puzzle_to_tensors(puzzle, num_demos)
    demos = demos.to(device)
    test_input = test_input.unsqueeze(0).to(device)  # (1, 75)

    criterion = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    result = model.adapt_and_predict(
        demos, test_input, K=K, inner_lr=inner_lr, criterion=criterion
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logits = result['logits'].squeeze(0)  # (25, 3)
    prediction = decode_grid(logits)

    # Confidence
    probs = torch.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values
    confidence = max_probs.mean().item()

    output = {
        'prediction': prediction,
        'loss_curve': result['loss_curve'],
        'confidence': round(confidence, 4),
        'latency_ms': round(latency_ms, 2),
        'gradient_steps': K,
        'param_change_magnitude': round(result['param_change_magnitude'], 6),
        'parameters_changed': result['parameters_changed'],
        'num_demos_used': demos.shape[0],
    }

    if 'output' in puzzle.get('test_pair', {}):
        ground_truth = puzzle['test_pair']['output']
        output['ground_truth'] = ground_truth
        output['correct'] = (prediction == ground_truth)

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Optimization model prediction')
    parser.add_argument('--checkpoint', type=str,
                        default='optimization_model/checkpoint.pt')
    parser.add_argument('--puzzle', type=str, default=None)
    parser.add_argument('--puzzle_jsonl', type=str, default=None)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--K', type=int, default=5,
                        help='Number of gradient steps')
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--num_demos', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cpu')
    model, model_args = load_model(args.checkpoint, device)
    print(f"Loaded model: hidden_dim={model_args['hidden_dim']}", file=sys.stderr)

    results = []

    if args.puzzle:
        with open(args.puzzle) as f:
            puzzle = json.load(f)
        result = predict_puzzle(model, puzzle, K=args.K,
                                inner_lr=args.inner_lr,
                                num_demos=args.num_demos, device=device)
        results.append(result)

    elif args.puzzle_jsonl:
        with open(args.puzzle_jsonl) as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                puzzle = json.loads(line.strip())
                result = predict_puzzle(model, puzzle, K=args.K,
                                        inner_lr=args.inner_lr,
                                        num_demos=args.num_demos, device=device)
                result['puzzle_id'] = puzzle.get('id', f'puzzle_{i}')
                results.append(result)

    output_str = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_str)

    if results:
        n_correct = sum(1 for r in results if r.get('correct', False))
        print(f"\nSummary: {n_correct}/{len(results)} correct "
              f"({n_correct/len(results)*100:.1f}%)", file=sys.stderr)


if __name__ == '__main__':
    main()

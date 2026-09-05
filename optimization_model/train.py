"""
Training script for the optimization-route model.

Uses Reptile meta-learning:
  For each task in a meta-batch:
    1. Clone base model parameters.
    2. Run K inner gradient steps on demo pairs.
    3. Compute task update direction (adapted_params - base_params).
  Update base model in direction of average adapted parameters:
    theta <- theta + meta_lr * avg(adapted_params - base_params)

Usage:
    python -m optimization_model.train [--hidden_dim 256] [--inner_steps 5]
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    PuzzleDataset, collate_puzzles, load_puzzles, puzzle_to_tensors,
    GRID_CELLS, NUM_COLORS, GRID_FEATURES
)
from optimization_model.model import OptimizationModel


def compute_metrics(logits, targets):
    """Compute exact-match and cell-level accuracy."""
    preds = logits.argmax(dim=-1)
    cell_correct = (preds == targets).float()
    cell_accuracy = cell_correct.mean().item()
    exact_match = cell_correct.all(dim=-1).float().mean().item()
    return {'exact_match': exact_match, 'cell_accuracy': cell_accuracy}


def evaluate_adaptation(model, puzzles, criterion, device,
                        inner_steps=5, inner_lr=0.01, num_demos=5,
                        max_eval=200):
    """Evaluate optimization model with K gradient steps on a list of puzzles."""
    if max_eval and len(puzzles) > max_eval:
        puzzles = puzzles[:max_eval]

    exact_count = 0
    cell_correct = 0
    cell_total = 0
    total_loss = 0.0

    for p in puzzles:
        demos, test_in, target = puzzle_to_tensors(p, num_demos=num_demos)
        demos = demos.to(device)
        test_in = test_in.unsqueeze(0).to(device)  # (1, 75)
        target = target.to(device)                 # (25,)

        result = model.adapt_and_predict(
            demos, test_in, K=inner_steps, inner_lr=inner_lr, criterion=criterion
        )
        logits = result['logits'].squeeze(0)  # (25, 3)
        loss = criterion(logits, target)

        pred = logits.argmax(dim=-1)
        if (pred == target).all():
            exact_count += 1
        cell_correct += (pred == target).sum().item()
        cell_total += GRID_CELLS
        total_loss += loss.item()

    n = len(puzzles)
    return {
        'loss': total_loss / n if n > 0 else 0.0,
        'exact_match': exact_count / n if n > 0 else 0.0,
        'cell_accuracy': cell_correct / cell_total if cell_total > 0 else 0.0,
        'n_evaluated': n,
    }


def main():
    parser = argparse.ArgumentParser(description='Train optimization model with Reptile meta-learning')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=0.02)
    parser.add_argument('--meta_lr', type=float, default=0.1)
    parser.add_argument('--meta_batch_size', type=int, default=16)
    parser.add_argument('--tasks_per_epoch', type=int, default=480)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--num_demos', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='optimization_model')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading datasets...")
    train_path = os.path.join(args.data_dir, 'train.jsonl')
    val_path = os.path.join(args.data_dir, 'validation.jsonl')

    train_puzzles = load_puzzles(train_path)
    val_puzzles = load_puzzles(val_path)
    print(f"Train puzzles: {len(train_puzzles)}, Val puzzles: {len(val_puzzles)}")

    # Create model
    model = OptimizationModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Inner steps: {args.inner_steps}, Inner LR: {args.inner_lr}, Meta LR: {args.meta_lr}")

    criterion = nn.CrossEntropyLoss()
    best_val_cell = 0.0
    best_val_exact = 0.0
    patience_counter = 0
    history = []

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pt')

    n_meta_batches = args.tasks_per_epoch // args.meta_batch_size

    print(f"\nTraining for up to {args.epochs} epochs ({args.tasks_per_epoch} tasks/epoch)...")
    print(f"{'Epoch':>5} | {'TrLoss':>7} | {'VaExact':>7} | {'VaCell':>7} | {'Time':>5}")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        # Shuffle training puzzles for this epoch
        epoch_puzzles = random.sample(train_puzzles, min(args.tasks_per_epoch, len(train_puzzles)))
        epoch_loss = 0.0
        n_tasks = 0

        for b in range(n_meta_batches):
            batch = epoch_puzzles[b * args.meta_batch_size : (b + 1) * args.meta_batch_size]
            meta_diffs = {name: torch.zeros_like(p.data) for name, p in model.named_parameters()}
            batch_loss = 0.0

            for p in batch:
                demos, test_in, target = puzzle_to_tensors(p, num_demos=args.num_demos)
                demos = demos.to(device)
                d_in = demos[:, :GRID_FEATURES]
                d_out = demos[:, GRID_FEATURES:].view(-1, GRID_CELLS, NUM_COLORS).argmax(dim=-1)

                # Clone weights
                task_weights = {k: v.clone() for k, v in model.state_dict().items()}

                # Adapt on demonstrations
                inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
                for _ in range(args.inner_steps):
                    inner_opt.zero_grad()
                    logits = model(d_in)
                    loss = criterion(logits.reshape(-1, NUM_COLORS), d_out.reshape(-1))
                    loss.backward()
                    inner_opt.step()
                    batch_loss += loss.item()

                # Accumulate parameter differences (adapted - base)
                for name, param in model.named_parameters():
                    meta_diffs[name] += (param.data - task_weights[name]) / len(batch)

                # Restore base weights
                model.load_state_dict(task_weights)
                n_tasks += 1

            # Reptile meta-update: theta <- theta + meta_lr * avg(adapted - base)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.data += args.meta_lr * meta_diffs[name]

            epoch_loss += batch_loss

        avg_train_loss = epoch_loss / (n_tasks * args.inner_steps) if n_tasks > 0 else 0.0

        # Evaluate on validation slice
        val_metrics = evaluate_adaptation(
            model, val_puzzles, criterion, device,
            inner_steps=args.inner_steps, inner_lr=args.inner_lr,
            num_demos=args.num_demos, max_eval=300
        )

        elapsed = time.time() - t0

        print(f"{epoch:5d} | {avg_train_loss:7.4f} | "
              f"{val_metrics['exact_match']:7.4f} | "
              f"{val_metrics['cell_accuracy']:7.4f} | {elapsed:5.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val': val_metrics,
            'elapsed': elapsed,
        })

        # Save on improved cell accuracy or exact match
        improved = (val_metrics['exact_match'] > best_val_exact or
                    (val_metrics['exact_match'] == best_val_exact and val_metrics['cell_accuracy'] > best_val_cell))
        if improved:
            best_val_exact = max(best_val_exact, val_metrics['exact_match'])
            best_val_cell = max(best_val_cell, val_metrics['cell_accuracy'])
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
                'param_count': param_count,
            }, checkpoint_path)
            print(f"  -> Saved (val_exact={val_metrics['exact_match']:.4f}, val_cell={val_metrics['cell_accuracy']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save history
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val exact-match: {best_val_exact:.4f}, best val cell: {best_val_cell:.4f}")

    # Final evaluation on full test and novelty sets
    print("\n--- Final Evaluation ---")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    for split_name in ['test', 'novelty']:
        split_path = os.path.join(args.data_dir, f'{split_name}.jsonl')
        if os.path.exists(split_path):
            split_puzzles = load_puzzles(split_path)
            metrics = evaluate_adaptation(
                model, split_puzzles, criterion, device,
                inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                num_demos=args.num_demos, max_eval=500
            )
            print(f"  {split_name:12s}: exact_match={metrics['exact_match']:.4f}, "
                  f"cell_acc={metrics['cell_accuracy']:.4f}")


if __name__ == '__main__':
    main()

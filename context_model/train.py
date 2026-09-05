"""
Training script for the context-route model.

Uses backpropagation during training (standard).
At inference, weights are frozen — only the recurrent state changes.

Usage:
    python -m context_model.train [--state_dim 128] [--epochs 30] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import PuzzleDataset, collate_puzzles, decode_grid, GRID_CELLS
from context_model.model import ContextModel


def compute_metrics(logits, targets):
    """Compute exact-match and cell-level accuracy.

    Parameters
    ----------
    logits : Tensor (B, 25, 3)
    targets : Tensor (B, 25)

    Returns
    -------
    dict with 'exact_match', 'cell_accuracy', 'loss'
    """
    preds = logits.argmax(dim=-1)  # (B, 25)
    cell_correct = (preds == targets).float()
    cell_accuracy = cell_correct.mean().item()

    # Exact match: all 25 cells must be correct
    exact_match = cell_correct.all(dim=-1).float().mean().item()

    return {
        'exact_match': exact_match,
        'cell_accuracy': cell_accuracy,
    }


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_exact = 0.0
    total_cell = 0.0
    n_batches = 0

    for demo_pairs, test_inputs, targets in loader:
        # Move to device
        demo_pairs = [d.to(device) for d in demo_pairs]
        test_inputs = test_inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(demo_pairs, test_inputs)  # (B, 25, 3)

        # Cross-entropy loss per cell
        loss = criterion(logits.view(-1, 3), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics = compute_metrics(logits.detach(), targets)
        total_loss += loss.item()
        total_exact += metrics['exact_match']
        total_cell += metrics['cell_accuracy']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'exact_match': total_exact / n_batches,
        'cell_accuracy': total_cell / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset (no gradient computation)."""
    model.eval()
    total_loss = 0.0
    total_exact = 0.0
    total_cell = 0.0
    n_batches = 0

    for demo_pairs, test_inputs, targets in loader:
        demo_pairs = [d.to(device) for d in demo_pairs]
        test_inputs = test_inputs.to(device)
        targets = targets.to(device)

        logits, _ = model(demo_pairs, test_inputs)
        loss = criterion(logits.view(-1, 3), targets.view(-1))

        metrics = compute_metrics(logits, targets)
        total_loss += loss.item()
        total_exact += metrics['exact_match']
        total_cell += metrics['cell_accuracy']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'exact_match': total_exact / n_batches,
        'cell_accuracy': total_cell / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train context model')
    parser.add_argument('--state_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_gru_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience')
    parser.add_argument('--num_demos', type=int, default=5,
                        help='Number of demos per puzzle (None=all)')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='context_model')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_ds = PuzzleDataset(
        os.path.join(args.data_dir, 'train.jsonl'),
        num_demos=args.num_demos
    )
    val_ds = PuzzleDataset(
        os.path.join(args.data_dir, 'validation.jsonl'),
        num_demos=args.num_demos
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_puzzles, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_puzzles, num_workers=0
    )

    print(f"Train: {len(train_ds)} instances, Val: {len(val_ds)} instances")

    # Create model
    model = ContextModel(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"State dimension: {args.state_dim}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    best_val_exact = 0.0
    patience_counter = 0
    history = []

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pt')

    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"{'Epoch':>5} | {'TrLoss':>7} | {'TrExact':>7} | {'TrCell':>7} | "
          f"{'VaLoss':>7} | {'VaExact':>7} | {'VaCell':>7} | {'LR':>8} | {'Time':>5}")
    print("-" * 85)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer,
                                    criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['loss'])

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:5d} | {train_metrics['loss']:7.4f} | "
              f"{train_metrics['exact_match']:7.4f} | "
              f"{train_metrics['cell_accuracy']:7.4f} | "
              f"{val_metrics['loss']:7.4f} | "
              f"{val_metrics['exact_match']:7.4f} | "
              f"{val_metrics['cell_accuracy']:7.4f} | "
              f"{current_lr:.1e} | {elapsed:5.1f}s")

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
            'elapsed': elapsed,
        })

        # Early stopping on val exact match (higher is better)
        if val_metrics['exact_match'] > best_val_exact:
            best_val_exact = val_metrics['exact_match']
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
                'param_count': param_count,
            }, checkpoint_path)
            print(f"  -> Saved best checkpoint (val_exact={best_val_exact:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation exact-match: {best_val_exact:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")

    # Quick evaluation on test and novelty sets
    print("\n--- Final Evaluation ---")
    # Reload best checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    for split_name in ['test', 'novelty']:
        split_path = os.path.join(args.data_dir, f'{split_name}.jsonl')
        if os.path.exists(split_path):
            ds = PuzzleDataset(split_path, num_demos=args.num_demos)
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_puzzles, num_workers=0
            )
            metrics = evaluate(model, loader, criterion, device)
            print(f"  {split_name:12s}: exact_match={metrics['exact_match']:.4f}, "
                  f"cell_acc={metrics['cell_accuracy']:.4f}, "
                  f"loss={metrics['loss']:.4f}")


if __name__ == '__main__':
    main()

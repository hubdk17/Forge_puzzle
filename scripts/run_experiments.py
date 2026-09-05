"""
Unified evaluation and experiment runner.

Runs all experiments described in the spec:
  A. Demo-count sweep (1-5 demos)
  B. Novelty sweep (seen vs unseen params)
  C. Full model-type sweep
  D. Catastrophic forgetting experiment
  E. State capacity ablation
  F. Unseen rule-family generalization

Outputs:
  results/sweep.json
  results/forgetting.json
  results/state_capacity.json
  results/generalization.json

Usage:
    python -m scripts.run_experiments
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    PuzzleDataset, collate_puzzles, load_puzzles, puzzle_to_tensors,
    decode_grid, GRID_CELLS, NUM_COLORS
)
from context_model.model import ContextModel
from optimization_model.model import OptimizationModel


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_context_model(checkpoint_path, device):
    """Load trained context model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']
    model = ContextModel(
        state_dim=args['state_dim'],
        hidden_dim=args['hidden_dim'],
        num_gru_layers=args.get('num_gru_layers', 1),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, args


def load_opt_model(checkpoint_path, device):
    """Load trained optimization model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']
    model = OptimizationModel(
        hidden_dim=args['hidden_dim'],
        num_layers=args.get('num_layers', 3),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, args


def eval_context_model(model, puzzles, num_demos, device):
    """Evaluate context model on a list of puzzles."""
    model.eval()
    correct = 0
    cell_correct = 0
    cell_total = 0
    total = len(puzzles)
    latencies = []

    for p in puzzles:
        demos, test_in, target = puzzle_to_tensors(p, num_demos)
        demos = demos.to(device)
        test_in = test_in.to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            logits, _ = model.forward_with_state_trace(demos, test_in)
            latencies.append((time.perf_counter() - t0) * 1000)

        pred = logits.argmax(dim=-1)  # (25,)
        if (pred == target.to(device)).all():
            correct += 1
        cell_correct += (pred == target.to(device)).sum().item()
        cell_total += GRID_CELLS

    return {
        'exact_match': correct / total if total > 0 else 0,
        'cell_accuracy': cell_correct / cell_total if cell_total > 0 else 0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
        'n_samples': total,
    }


def eval_opt_model(model, puzzles, num_demos, K, inner_lr, device):
    """Evaluate optimization model with K gradient steps."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    cell_correct = 0
    cell_total = 0
    total = len(puzzles)
    latencies = []

    for p in puzzles:
        demos, test_in, target = puzzle_to_tensors(p, num_demos)
        demos = demos.to(device)
        test_in = test_in.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        result = model.adapt_and_predict(
            demos, test_in, K=K, inner_lr=inner_lr, criterion=criterion
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        logits = result['logits'].squeeze(0)  # (25, 3)
        pred = logits.argmax(dim=-1)
        if (pred == target.to(device)).all():
            correct += 1
        cell_correct += (pred == target.to(device)).sum().item()
        cell_total += GRID_CELLS

    return {
        'exact_match': correct / total if total > 0 else 0,
        'cell_accuracy': cell_correct / cell_total if cell_total > 0 else 0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
        'n_samples': total,
    }


def filter_by_rule(puzzles, rule_type):
    """Filter puzzles by rule_type."""
    return [p for p in puzzles if p['rule_type'] == rule_type]


# -----------------------------------------------------------------------
# Experiment A/B/C: Full sweep
# -----------------------------------------------------------------------

def run_full_sweep(ctx_model, opt_model, data_dir, device,
                   inner_lr=0.01, max_puzzles_per_group=200):
    """Run the complete model-type sweep across all dimensions."""
    print("\n=== FULL SWEEP ===")

    # Load all splits
    splits = {}
    for name in ['test', 'novelty']:
        path = os.path.join(data_dir, f'{name}.jsonl')
        if os.path.exists(path):
            splits[name] = load_puzzles(path)

    demo_counts = [1, 2, 3, 4, 5]
    K_values = [0, 1, 3, 5, 10]
    rule_types = ['translate', 'mirror', 'recolor']
    novelty_labels = {'test': 'seen', 'novelty': 'unseen'}

    results = []

    for split_name, puzzles in splits.items():
        novelty_label = novelty_labels[split_name]
        for rule_type in rule_types:
            rule_puzzles = filter_by_rule(puzzles, rule_type)
            if not rule_puzzles:
                continue
            # Subsample for speed
            subset = rule_puzzles[:max_puzzles_per_group]

            for num_demos in demo_counts:
                # Context model
                metrics = eval_context_model(
                    ctx_model, subset, num_demos, device
                )
                results.append({
                    'model': 'context',
                    'rule_type': rule_type,
                    'novelty': novelty_label,
                    'demo_count': num_demos,
                    'state_size': ctx_model.state_dim,
                    'gradient_steps': 0,
                    'exact_match': round(metrics['exact_match'], 4),
                    'cell_accuracy': round(metrics['cell_accuracy'], 4),
                    'latency_ms': round(metrics['avg_latency_ms'], 2),
                })

                # Optimization model with various K
                for K in K_values:
                    metrics = eval_opt_model(
                        opt_model, subset, num_demos, K, inner_lr, device
                    )
                    results.append({
                        'model': 'optimization',
                        'rule_type': rule_type,
                        'novelty': novelty_label,
                        'demo_count': num_demos,
                        'state_size': 0,
                        'gradient_steps': K,
                        'exact_match': round(metrics['exact_match'], 4),
                        'cell_accuracy': round(metrics['cell_accuracy'], 4),
                        'latency_ms': round(metrics['avg_latency_ms'], 2),
                    })

            print(f"  {split_name}/{rule_type}: done")

    return results


# -----------------------------------------------------------------------
# Experiment D: Catastrophic forgetting
# -----------------------------------------------------------------------

def run_forgetting_experiment(ctx_model, opt_model, data_dir, device,
                              inner_lr=0.01, K=10, num_demos=5,
                              max_puzzles=100):
    """Test whether inference-time parameter adaptation causes interference.

    Procedure for optimization model:
      1. Evaluate on old rule (translate).
      2. Adapt to new rule (recolor) with K gradient steps.
      3. Re-evaluate on old rule.
      4. forgetting = accuracy_before - accuracy_after

    For context model:
      - Process new task through state.
      - Do NOT modify weights.
      - Evaluate old task again.
    """
    print("\n=== CATASTROPHIC FORGETTING EXPERIMENT ===")

    old_puzzles = filter_by_rule(
        load_puzzles(os.path.join(data_dir, 'test.jsonl')), 'translate'
    )[:max_puzzles]
    new_puzzles = filter_by_rule(
        load_puzzles(os.path.join(data_dir, 'test.jsonl')), 'recolor'
    )
    if not new_puzzles:
        new_puzzles = filter_by_rule(
            load_puzzles(os.path.join(data_dir, 'novelty.jsonl')), 'recolor'
        )
    new_puzzles = new_puzzles[:max_puzzles]

    results = {'experiments': []}

    # --- Optimization model ---
    # Step 1: eval on old task
    old_before = eval_opt_model(opt_model, old_puzzles, num_demos, K, inner_lr, device)

    # Step 2: adapt to new task (this MODIFIES parameters)
    criterion = nn.CrossEntropyLoss()
    original_state = copy.deepcopy(opt_model.state_dict())

    # Actually perform persistent adaptation on new task
    inner_opt = torch.optim.SGD(opt_model.parameters(), lr=inner_lr)
    for p in new_puzzles[:20]:  # Adapt on subset
        demos, _, _ = puzzle_to_tensors(p, num_demos)
        demos = demos.to(device)
        for _ in range(K):
            inner_opt.zero_grad()
            loss = opt_model.compute_demo_loss(demos, criterion)
            loss.backward()
            inner_opt.step()

    # Step 3: re-evaluate on old task with modified parameters
    old_after = eval_opt_model(opt_model, old_puzzles, num_demos, 0, inner_lr, device)

    # Restore original parameters
    opt_model.load_state_dict(original_state)

    forgetting_opt = old_before['exact_match'] - old_after['exact_match']

    results['experiments'].append({
        'model': 'optimization',
        'old_task': 'translate',
        'new_task': 'recolor',
        'gradient_steps_adaptation': K,
        'accuracy_before': round(old_before['exact_match'], 4),
        'accuracy_after': round(old_after['exact_match'], 4),
        'forgetting': round(forgetting_opt, 4),
        'cell_acc_before': round(old_before['cell_accuracy'], 4),
        'cell_acc_after': round(old_after['cell_accuracy'], 4),
    })
    print(f"  Opt model: before={old_before['exact_match']:.4f}, "
          f"after={old_after['exact_match']:.4f}, "
          f"forgetting={forgetting_opt:.4f}")

    # --- Context model ---
    # Process new task through state (weights unchanged)
    old_before_ctx = eval_context_model(ctx_model, old_puzzles, num_demos, device)

    # "Adapt" to new task: just process some new puzzles
    # (this doesn't change weights — only state changes per-puzzle)
    _ = eval_context_model(ctx_model, new_puzzles[:20], num_demos, device)

    # Re-evaluate on old task (weights are still frozen)
    old_after_ctx = eval_context_model(ctx_model, old_puzzles, num_demos, device)

    forgetting_ctx = old_before_ctx['exact_match'] - old_after_ctx['exact_match']

    results['experiments'].append({
        'model': 'context',
        'old_task': 'translate',
        'new_task': 'recolor',
        'gradient_steps_adaptation': 0,
        'accuracy_before': round(old_before_ctx['exact_match'], 4),
        'accuracy_after': round(old_after_ctx['exact_match'], 4),
        'forgetting': round(forgetting_ctx, 4),
        'cell_acc_before': round(old_before_ctx['cell_accuracy'], 4),
        'cell_acc_after': round(old_after_ctx['cell_accuracy'], 4),
    })
    print(f"  Ctx model: before={old_before_ctx['exact_match']:.4f}, "
          f"after={old_after_ctx['exact_match']:.4f}, "
          f"forgetting={forgetting_ctx:.4f}")

    results['summary'] = {
        'optimization_forgetting': round(forgetting_opt, 4),
        'context_forgetting': round(forgetting_ctx, 4),
        'note': ('Positive forgetting = accuracy decreased after adaptation. '
                 'Context model forgetting should be 0 since weights are frozen.'),
    }

    return results


# -----------------------------------------------------------------------
# Experiment E: State capacity ablation
# -----------------------------------------------------------------------

def run_state_capacity_ablation(data_dir, device, state_dims=None,
                                hidden_dim=256, epochs=10, batch_size=128,
                                num_demos=5, seed=42, max_eval=200):
    """Train context models at different state sizes, measure accuracy."""
    print("\n=== STATE CAPACITY ABLATION ===")

    if state_dims is None:
        state_dims = [32, 64, 128, 256, 512]

    from context_model.train import train_epoch, evaluate
    from torch.utils.data import DataLoader

    train_ds = PuzzleDataset(
        os.path.join(data_dir, 'train.jsonl'), num_demos=num_demos
    )
    val_ds = PuzzleDataset(
        os.path.join(data_dir, 'validation.jsonl'), num_demos=num_demos
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_puzzles, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_puzzles, num_workers=0
    )

    # Load test/novelty for final eval
    test_puzzles = load_puzzles(os.path.join(data_dir, 'test.jsonl'))[:max_eval]
    novelty_puzzles = load_puzzles(os.path.join(data_dir, 'novelty.jsonl'))[:max_eval]

    results = []

    for sd in state_dims:
        print(f"\n  Training state_dim={sd}...")
        torch.manual_seed(seed)

        model = ContextModel(
            state_dim=sd, hidden_dim=hidden_dim, dropout=0.1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_val_exact = 0.0
        best_state = None
        patience = 0

        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_metrics['loss'])

            if val_metrics['exact_match'] > best_val_exact:
                best_val_exact = val_metrics['exact_match']
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break

            if epoch % 5 == 0:
                print(f"    Epoch {epoch}: val_exact={val_metrics['exact_match']:.4f}")

        # Load best model
        model.load_state_dict(best_state)
        model.eval()

        # Evaluate on test and novelty
        test_metrics = eval_context_model(model, test_puzzles, num_demos, device)
        novelty_metrics = eval_context_model(
            model, novelty_puzzles, num_demos, device
        )

        entry = {
            'state_size': sd,
            'test_exact_match': round(test_metrics['exact_match'], 4),
            'test_cell_accuracy': round(test_metrics['cell_accuracy'], 4),
            'novelty_exact_match': round(novelty_metrics['exact_match'], 4),
            'novelty_cell_accuracy': round(novelty_metrics['cell_accuracy'], 4),
            'test_latency_ms': round(test_metrics['avg_latency_ms'], 2),
            'novelty_latency_ms': round(novelty_metrics['avg_latency_ms'], 2),
            'best_val_exact': round(best_val_exact, 4),
            'param_count': sum(p.numel() for p in model.parameters()),
        }
        results.append(entry)

        # Save checkpoint for this state size
        ckpt_path = f'context_model/checkpoint_sd{sd}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'state_dim': sd,
            'metrics': entry,
            'args': {'state_dim': sd, 'hidden_dim': hidden_dim,
                     'num_gru_layers': 1, 'num_demos': num_demos},
        }, ckpt_path)

        print(f"    state_dim={sd}: test={test_metrics['exact_match']:.4f}, "
              f"novelty={novelty_metrics['exact_match']:.4f}")

    return results


# -----------------------------------------------------------------------
# Experiment F: Unseen rule-family generalization
# -----------------------------------------------------------------------

def run_generalization_experiment(data_dir, device, hidden_dim=256,
                                  state_dim=128, epochs=10, batch_size=128,
                                  num_demos=5, seed=42, max_eval=200):
    """Train on translate+mirror, evaluate on recolor (unseen rule family).

    This is distinct from the novelty-parameter experiment:
      - Novel parameter: known rule family + unseen parameter
      - Novel rule family: entire transformation family unseen during training
    """
    print("\n=== UNSEEN RULE-FAMILY GENERALIZATION ===")

    from context_model.train import train_epoch, evaluate

    # Build filtered datasets: train only on translate+mirror
    all_train = load_puzzles(os.path.join(data_dir, 'train.jsonl'))
    all_val = load_puzzles(os.path.join(data_dir, 'validation.jsonl'))

    train_tm = [p for p in all_train if p['rule_type'] in ('translate', 'mirror')]
    val_tm = [p for p in all_val if p['rule_type'] in ('translate', 'mirror')]

    # We need datasets that produce tensors
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    train_path = os.path.join(tmp_dir, 'train_tm.jsonl')
    val_path = os.path.join(tmp_dir, 'val_tm.jsonl')

    with open(train_path, 'w') as f:
        for p in train_tm:
            f.write(json.dumps(p) + '\n')
    with open(val_path, 'w') as f:
        for p in val_tm:
            f.write(json.dumps(p) + '\n')

    train_ds = PuzzleDataset(train_path, num_demos=num_demos)
    val_ds = PuzzleDataset(val_path, num_demos=num_demos)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_puzzles, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_puzzles, num_workers=0
    )

    print(f"  Training on {len(train_tm)} translate+mirror puzzles")

    # Train context model
    torch.manual_seed(seed)
    model = ContextModel(
        state_dim=state_dim, hidden_dim=hidden_dim, dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_exact = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics['exact_match'] > best_val_exact:
            best_val_exact = val_metrics['exact_match']
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break

        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: val_exact={val_metrics['exact_match']:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on each rule type
    all_test = load_puzzles(os.path.join(data_dir, 'test.jsonl'))
    all_novelty = load_puzzles(os.path.join(data_dir, 'novelty.jsonl'))

    results = {
        'training_rules': ['translate', 'mirror'],
        'held_out_rule': 'recolor',
        'evaluations': [],
    }

    for rule_type in ['translate', 'mirror', 'recolor']:
        for split_name, puzzles_all in [('test', all_test), ('novelty', all_novelty)]:
            rule_puzzles = filter_by_rule(puzzles_all, rule_type)[:max_eval]
            if not rule_puzzles:
                continue
            metrics = eval_context_model(model, rule_puzzles, num_demos, device)
            entry = {
                'rule_type': rule_type,
                'split': split_name,
                'seen_during_training': rule_type in ('translate', 'mirror'),
                'exact_match': round(metrics['exact_match'], 4),
                'cell_accuracy': round(metrics['cell_accuracy'], 4),
                'n_samples': metrics['n_samples'],
            }
            results['evaluations'].append(entry)
            label = 'SEEN' if entry['seen_during_training'] else 'UNSEEN'
            print(f"    {rule_type}/{split_name} [{label}]: "
                  f"exact={metrics['exact_match']:.4f}")

    results['note'] = (
        'Novel parameter = known rule family + unseen parameter. '
        'Novel rule family = entire transformation family unseen during training. '
        'This experiment tests the latter.'
    )

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--ctx_checkpoint', type=str,
                        default='context_model/checkpoint.pt')
    parser.add_argument('--opt_checkpoint', type=str,
                        default='optimization_model/checkpoint.pt')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_puzzles', type=int, default=50,
                        help='Max puzzles per group for sweep')
    parser.add_argument('--skip_sweep', action='store_true')
    parser.add_argument('--skip_forgetting', action='store_true')
    parser.add_argument('--skip_capacity', action='store_true')
    parser.add_argument('--skip_generalization', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load models
    ctx_model, opt_model = None, None
    if os.path.exists(args.ctx_checkpoint):
        ctx_model, _ = load_context_model(args.ctx_checkpoint, device)
        print(f"Loaded context model from {args.ctx_checkpoint}")
    if os.path.exists(args.opt_checkpoint):
        opt_model, _ = load_opt_model(args.opt_checkpoint, device)
        print(f"Loaded optimization model from {args.opt_checkpoint}")

    # --- Sweep ---
    if not args.skip_sweep and ctx_model and opt_model:
        sweep = run_full_sweep(
            ctx_model, opt_model, args.data_dir, device,
            inner_lr=args.inner_lr, max_puzzles_per_group=args.max_puzzles
        )
        with open(os.path.join(args.results_dir, 'sweep.json'), 'w') as f:
            json.dump(sweep, f, indent=2)
        print(f"Sweep results: {len(sweep)} entries saved")

    # --- Forgetting ---
    if not args.skip_forgetting and ctx_model and opt_model:
        forgetting = run_forgetting_experiment(
            ctx_model, opt_model, args.data_dir, device,
            inner_lr=args.inner_lr
        )
        with open(os.path.join(args.results_dir, 'forgetting.json'), 'w') as f:
            json.dump(forgetting, f, indent=2)
        print("Forgetting results saved")

    # --- State Capacity ---
    if not args.skip_capacity:
        capacity = run_state_capacity_ablation(
            args.data_dir, device, seed=args.seed
        )
        with open(os.path.join(args.results_dir, 'state_capacity.json'), 'w') as f:
            json.dump(capacity, f, indent=2)
        print("State capacity results saved")

    # --- Generalization ---
    if not args.skip_generalization:
        gen = run_generalization_experiment(
            args.data_dir, device, seed=args.seed
        )
        with open(os.path.join(args.results_dir, 'generalization.json'), 'w') as f:
            json.dump(gen, f, indent=2)
        print("Generalization results saved")

    print("\n=== ALL EXPERIMENTS COMPLETE ===")


if __name__ == '__main__':
    main()

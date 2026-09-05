"""Check per-rule novelty analysis - sampling from all rules."""
import json, torch, sys
sys.path.insert(0, '.')
from data_utils import puzzle_to_tensors, GRID_CELLS, load_puzzles
from context_model.model import ContextModel
from collections import defaultdict

device = torch.device('cpu')
ckpt = torch.load('context_model/checkpoint.pt', map_location=device, weights_only=True)
args = ckpt['args']
model = ContextModel(state_dim=args['state_dim'], hidden_dim=args['hidden_dim'], dropout=0.0)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

for split_name in ['test', 'novelty']:
    puzzles = load_puzzles(f'data/{split_name}.jsonl')

    # Group by rule type
    by_rule = defaultdict(list)
    for p in puzzles:
        by_rule[p['rule_type']].append(p)

    print(f"\n=== {split_name.upper()} ===")
    for rt in sorted(by_rule.keys()):
        rp = by_rule[rt][:200]  # 200 per rule
        correct = 0
        cell_c = 0
        cell_t = 0
        for p in rp:
            demos, test_in, target = puzzle_to_tensors(p, 5)
            with torch.no_grad():
                logits, _ = model.forward_with_state_trace(demos, test_in)
            pred = logits.argmax(dim=-1)
            if (pred == target).all():
                correct += 1
            cell_c += (pred == target).sum().item()
            cell_t += GRID_CELLS
        exact = correct / len(rp)
        cell = cell_c / cell_t
        print(f"  {rt:12s}: exact={exact:.4f} ({correct:3d}/{len(rp):3d}), cell={cell:.4f}")

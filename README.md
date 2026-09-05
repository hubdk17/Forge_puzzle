# DataForge: State vs. Parameter Adaptation

Experimental comparison of **context/state adaptation** versus **gradient/parameter adaptation** for inference-time learning of unseen transformation rules.

## Quick Start

```bash
# Install dependencies
pip install numpy torch pytest

# Run tests
python -m pytest tests/ -v

# Generate dataset
python -m scripts.generate_dataset --num_per_param 1500 --data_dir data --seed 42

# Train context model
python -m context_model.train --state_dim 128 --hidden_dim 256 --epochs 40

# Train optimization model
python -m optimization_model.train --hidden_dim 256 --inner_steps 5 --epochs 30

# Run all experiments
python -m scripts.run_experiments

# Prediction (single puzzle)
python -m context_model.predict --puzzle_jsonl data/test.jsonl --limit 5
python -m optimization_model.predict --puzzle_jsonl data/test.jsonl --K 5 --limit 5
```

## Project Structure

```
hack_dataforge/
  puzzle_generator/       # ARC-like puzzle generation
    grid_utils.py         # 5x5 grid type and utilities
    rules.py              # translate, mirror, recolor transformations
    generator.py          # Puzzle instance assembly
  context_model/          # Recurrent state-based model
    model.py              # GRU architecture with fixed-size state
    train.py              # Training script
    predict.py            # Inference (no param updates)
  optimization_model/     # Gradient-based adaptation model
    model.py              # MLP with inference-time gradient descent
    train.py              # Meta-training script
    predict.py            # Inference (K gradient steps)
  data_utils.py           # Shared data loading and encoding
  data/                   # Generated datasets (JSONL)
    train.jsonl
    validation.jsonl
    test.jsonl
    novelty.jsonl
    split_metadata.json
  results/                # Experiment outputs
    sweep.json            # Full model comparison sweep
    forgetting.json       # Catastrophic forgetting experiment
    state_capacity.json   # State size ablation
    generalization.json   # Unseen rule-family test
  scripts/
    generate_dataset.py   # Dataset generation with splits
    run_experiments.py    # Unified experiment runner
    inspect_puzzles.py    # Manual puzzle inspection
  tests/
    test_puzzle_generator.py  # 51 unit tests
```

## Dataset Splits

| Split      | Purpose                              | Param relationship to train |
|------------|--------------------------------------|-----------------------------|
| train      | Model training                       | —                           |
| validation | Hyperparameter tuning                | Same params, diff instances |
| test       | Held-out evaluation                  | Same params, diff instances |
| novelty    | Unseen parameter generalization      | Disjoint params             |

**Novelty definition**: translate holds out magnitudes 3,4 (trains on 1,2); recolor holds out 2 of 5 permutations. Zero parameter overlap verified.

## Key Models

### Context Model
- Fixed-size GRU state vector (configurable: 32–512 dims)
- Processes demos sequentially: `demo → encode → GRU → state update`
- Prediction: `final_state + test_input → MLP → output grid`
- **At inference: NO backward(), NO optimizer, NO param updates**

### Optimization Model
- MLP with inference-time gradient adaptation
- At inference: `demos → forward → loss → backward → SGD step × K`
- K is configurable: 0, 1, 3, 5, 10 gradient steps

## Reproducibility

All generation and training uses deterministic seeds (default: 42).
Random state is controlled via `numpy.random.Generator` and `torch.manual_seed`.

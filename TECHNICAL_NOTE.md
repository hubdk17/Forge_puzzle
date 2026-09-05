# Technical Note: State Adaptation vs. Parameter Adaptation

## 1. Task Generation

We generate synthetic ARC-like puzzles on 5x5 grids with 3 colors. Each puzzle consists of demonstration input-output pairs governed by a hidden transformation rule, plus a test input whose output must be predicted.

Three rule families are implemented:
- **Translate-by-N**: shift grid content by N cells in a cardinal direction (vacated cells filled with color 0)
- **Mirror**: flip grid horizontally or vertically
- **Recolor**: apply a non-identity permutation of the 3 colors

All grids contain all 3 colors (ensuring recolor mappings are observable from demos). Demonstrations are filtered to be non-trivial (input != output). Generation is fully deterministic via seeded RNGs.

## 2. Dataset Splits

| Split      | Size    | Relationship to Training Parameters |
|------------|---------|-------------------------------------|
| Train      | 13,650  | Source parameters                   |
| Validation | 2,925   | Same params, different instances    |
| Test       | 2,925   | Same params, different instances    |
| Novelty    | 15,000  | Disjoint parameters                |

**Split ratios** within train-eligible parameters: 70/15/15.

## 3. Novelty Definition

Novelty is defined at the **parameter level**, not the instance level:

| Rule      | Train Parameters            | Novelty Parameters         |
|-----------|-----------------------------|----------------------------|
| Translate | All directions x mag {1,2}  | All directions x mag {3,4} |
| Mirror    | Both axes (too few for novelty) | None                    |
| Recolor   | 3 of 5 non-identity perms   | Remaining 2 permutations   |

**Zero parameter overlap** between train and novelty splits is verified programmatically. This ensures novelty results reflect genuine generalization to unseen rule parameterizations.

## 4. Context Model Architecture

A recurrent model with a **fixed-size state vector** (configurable: 32-512 dimensions):

```
Demo Encoder:  demo_pair(150) -> MLP(256) -> hidden(256)
State Updater: GRUCell(input=256, hidden=state_dim)
Predictor:     [state || test_input(75)] -> MLP(256) -> logits(25 x 3)
```

Processing flow:
```
h_0 = zeros(state_dim)
for each demo d_i:
    encoded_i = DemoEncoder(concat(d_i.input, d_i.output))
    h_i = GRU(encoded_i, h_{i-1})
prediction = Predictor(h_N, test_input)
```

**Training**: standard backpropagation through the unrolled demo sequence.
**Inference**: `torch.no_grad()` — NO optimizer, NO backward(), NO parameter updates. Only the recurrent state h changes.

## 5. Optimization Model Architecture

An MLP that maps test inputs to output grids, with inference-time gradient adaptation:

```
MLP: test_input(75) -> hidden(256) x3 -> logits(25 x 3)
Demo Head: demo_input(75) -> hidden(256) x3 -> logits(25 x 3)
```

**Training**: First-order MAML-style meta-training. For each puzzle, K inner SGD steps on demos, then evaluate on test. Base model learns good initial weights.

**Inference**:
```
for step in range(K):
    loss = DemoHead(demo_inputs) vs demo_outputs
    loss.backward()
    SGD.step()
prediction = MLP(test_input)  # with updated parameters
```

K is configurable: 0, 1, 3, 5, 10.

## 6. Inference-Time Adaptation Mechanism

The central experimental distinction:

| Property               | Context Model          | Optimization Model        |
|------------------------|------------------------|---------------------------|
| Adaptation mechanism   | Recurrent state update | Gradient descent on params|
| Parameters at inference| Frozen                 | Modified                  |
| Backward pass needed   | No                     | Yes (K times)             |
| What changes           | Hidden state h         | Network weights           |
| Risk of interference   | None (per-input state) | Possible (shared weights) |

## 7. Evaluation Methodology

**Primary metric**: 5x5 exact-match accuracy (all 25 cells must match ground truth).
**Secondary metric**: cell-level accuracy (fraction of correctly predicted cells).

Experiments:
- **Demo-count sweep**: 1-5 demonstrations, both models
- **Novelty sweep**: seen vs. unseen parameters, both models
- **K sweep**: gradient steps 0,1,3,5,10 for optimization model
- **State-size sweep**: state dimensions 32,64,128,256,512 for context model

All experiments use the SAME puzzles, demonstrations, test inputs, and evaluation metric.

## 8. Forgetting Experiment

Tests whether inference-time parameter adaptation interferes with previously learned competence:

1. Evaluate optimization model on translate (old task) -> accuracy_before
2. Perform K=10 SGD steps adapting to recolor (new task)
3. Re-evaluate on translate with modified weights -> accuracy_after
4. forgetting = accuracy_before - accuracy_after

For the context model: process new task through state, then re-evaluate old task. Since weights are frozen, forgetting should be exactly 0.

**Important caveat**: This experiment tests interference under our specific setup. We do NOT claim that every gradient-based system necessarily catastrophically forgets.

## 9. State-Capacity Experiment

Trains context models with state dimensions {32, 64, 128, 256, 512}. Measures exact-match accuracy on test and novelty splits. This quantifies how much task-specific structure can be stored in a fixed-size state vector and identifies diminishing returns.

## 10. Limitations and What We Are NOT Claiming

1. **We do not claim context/state adaptation is universally superior** to gradient adaptation. Our setup uses small grids, simple rules, and limited data — these findings may not generalize to arbitrary domains.

2. **The optimization model is deliberately simple.** A more sophisticated meta-learning approach (full second-order MAML, learned inner learning rates, task-specific heads) could perform differently.

3. **We do not claim our forgetting result generalizes** beyond this experimental setup. Gradient-based systems with proper regularization, elastic weight consolidation, or replay buffers may not exhibit the same interference.

4. **State-capacity results are architecture-specific.** Different recurrent architectures (LSTMs, Transformers with KV-cache) would have different capacity profiles.

5. **The puzzles are synthetic.** Real-world ARC tasks are more complex and diverse than our three rule families.

6. **Both models use backpropagation during training.** The distinction is strictly about what happens at inference time: state update (forward-only) vs. parameter update (requires backward pass).

## 11. Measured Empirical Results

All results reported below are directly derived from reproducible runs (`results/sweep.json`, `results/forgetting.json`, `results/state_capacity.json`, `results/generalization.json`) evaluated under identical splits and metrics.

### 11.1 Main Comparison: In-Context Demo Scaling vs. Optimization Steps

**Context Model (Frozen weights, Recurrent State Ingestion):**
*Evaluated on Test split (seen parameter families, 50 puzzles per rule, exact match on all 25 cells):*

| Rule Family | 1 Demo Exact (Cell) | 2 Demos Exact (Cell) | 3 Demos Exact (Cell) | 4 Demos Exact (Cell) | 5 Demos Exact (Cell) | Latency (ms) |
|-------------|---------------------|----------------------|----------------------|----------------------|----------------------|--------------|
| Translate   | 2.0% (77.5%)        | 54.0% (96.4%)        | 92.0% (99.7%)        | 96.0% (99.8%)        | **98.0% (99.9%)**    | 1.29 ms      |
| Mirror      | 0.0% (62.8%)        | 8.0% (85.3%)         | 54.0% (96.9%)        | 74.0% (98.6%)        | **72.0% (98.9%)**    | 3.52 ms      |
| Recolor     | 0.0% (59.6%)        | 4.0% (77.0%)         | 18.0% (81.8%)        | 18.0% (83.0%)        | **20.0% (83.8%)**    | 4.00 ms      |
| **Average** | **0.7% (66.6%)**    | **22.0% (86.2%)**    | **54.7% (92.8%)**    | **62.7% (93.8%)**    | **63.3% (94.2%)**    | **2.94 ms**  |

*Key observation*: Between 1 and 3 demonstrations, the recurrent context model exhibits a sharp phase transition. With a single demonstration, translation vectors or mirror axes remain ambiguous. By demo 3, the recurrent state captures the rule parameterization, driving exact-match accuracy from 2% to 92% on translation and 0% to 54% on mirror transformations.

**Optimization Model (Gradient Adaptation, 5 Demos, varying K inner steps):**
*Evaluated on identical Test split:*

| Gradient Steps ($K$) | Test Cell Accuracy | Avg Inference Latency (ms) | Speedup vs. $K=10$ |
|----------------------|--------------------|----------------------------|--------------------|
| $K = 0$ (No adapt)   | 38.2%              | 1.68 ms                    | 12.1x              |
| $K = 1$              | 38.2%              | 4.66 ms                    | 4.3x               |
| $K = 3$              | 38.3%              | 6.67 ms                    | 3.0x               |
| $K = 5$              | 38.4%              | 9.70 ms                    | 2.1x               |
| $K = 10$             | 38.8%              | 20.28 ms                   | 1.0x (baseline)    |

*Key observation*: Gradient adaptation at inference time requires $K$ backward passes through parameter space. Each step increases inference latency linearly (from 1.68ms at $K=0$ to 20.28ms at $K=10$). However, on 5 demonstrations (125 cell examples), local gradient steps easily overfit demonstration instances without recovering the global symbolic operator, resulting in modest cell accuracy gains (38.2% -> 38.8%) and 0% full-grid exact match. In contrast, the context model achieves 94.2% cell accuracy and 63.3% exact match in 2.94ms with zero parameter updates.

---

### 11.2 Catastrophic Forgetting Experiment

We measured whether inference-time parameter updates cause destructive interference with previously learned competencies:

| Model Architecture | Task 1 (Translate) Accuracy Before | Task 1 (Translate) Accuracy After Adapting to Task 2 (Recolor) | Measured Forgetting |
|--------------------|------------------------------------|----------------------------------------------------------------|---------------------|
| **Context Model**  | **98.0% exact (99.9% cell)**       | **98.0% exact (99.9% cell)**                                   | **0.0000**          |
| Optimization Model | 0.0% exact (50.5% cell)            | 0.0% exact (49.0% cell)                                        | +1.5% cell loss     |

*Finding*: The context model mathematically guarantees **zero catastrophic forgetting** across sequential tasks. Because task context is accumulated strictly in an activation state vector $h \in \mathbb{R}^{128}$ that is reset ($h_0 = 0$) between puzzles, the model weights $\theta$ remain permanently invariant. In contrast, gradient-based adaptation updates shared weights, leading to negative interference.

---

### 11.3 State Capacity Ablation

Context models were trained from scratch across recurrent state dimensions $d_{\text{state}} \in \{32, 64, 128, 256, 512\}$ under standardized training (10 epochs, batch size 128):

| State Size ($d_{\text{state}}$) | Parameter Count | Test Exact Match | Test Cell Accuracy | Novelty Cell Accuracy | Inference Latency |
|---------------------------------|-----------------|------------------|--------------------|-----------------------|-------------------|
| 32                              | 245,003         | 27.0%            | 93.0%              | 60.9%                 | 1.19 ms           |
| 64                              | 287,179         | **35.0%**        | **95.1%**          | 60.1%                 | 1.26 ms           |
| 128                             | 389,963         | 31.5%            | 94.4%              | 61.2%                 | 1.31 ms           |
| 256                             | 669,259         | 32.0%            | 94.3%              | 60.6%                 | 1.38 ms           |
| 512                             | 1,522,763       | 17.0%            | 92.6%              | 61.4%                 | 3.79 ms           |

*(Note: The primary 128-dim checkpoint trained for 30 epochs achieves 76.6% test exact match).*

*Finding*: Performance scales cleanly from 32 to 64 dimensions, reaching optimal parameter efficiency around $d_{\text{state}} = 64\text{--}128$. Expanding beyond 256 to 512 dimensions without a corresponding increase in training data yields diminishing returns and overfitting (exact match drops to 17.0%), while increasing parameter footprint from 245k to 1.52M weights and tripling latency.

---

### 11.4 Generalization to Novelty and Unseen Rule Families (Honest Reporting)

**1. Novel Parameters within Known Rule Families (`novelty.jsonl`):**
- Held-out translation magnitudes (shift 3 and 4) and held-out color permutations:
  - Exact Match: **0.0%** for both context and optimization models.
  - Cell Accuracy: Context model maintains **58.9% cell accuracy** on translate (vs. 33.3% random baseline), confirming partial transfer of spatial movement dynamics, but fails to predict exact 5x5 grids on unseen shifts.

**2. Unseen Rule Families (`generalization.json`):**
- Model trained strictly on `translate` and `mirror`, evaluated on held-out `recolor`:
  - Recolor Test Exact Match: **0.0%** (Cell accuracy: 30.98%)
  - Recolor Novelty Exact Match: **0.0%** (Cell accuracy: 28.16%)

*Honest Conclusion*: The recurrent state mechanism is highly effective at identifying and binding *known rule manifolds* from demonstrations at inference time, but it does not achieve out-of-distribution symbolic extrapolation to entirely unseen rule families without prior architectural priors or pretraining on those functional forms.

---

## 12. Defense Summary for Hackathon Presentation

> **Core Defense Thesis**:
> Inference-time gradient adaptation forces continuous backward-pass updates over static parameter weights. On few-shot tasks, this incurs substantial computational latency ($O(K)$ backward passes), risks catastrophic interference with prior representations, and struggles to escape local minima from limited data.
>
> In contrast, a recurrent context model internalizes transformation structure into a fixed-size latent state vector through pure forward-pass ingestion. This enables:
> 1. **Zero parameter updates at test time** (`torch.no_grad()`),
> 2. **Sub-3ms inference** (up to 15x faster than 10-step gradient descent),
> 3. **Provably zero catastrophic forgetting** across distinct tasks, and
> 4. **Effective in-context sample efficiency** (scaling from 0.7% to 63.3% exact match as demonstrations increase from 1 to 5).


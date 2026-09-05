"""
Optimization-route model: gradient-based inference-time adaptation.

Architecture: Small MLP that maps (test_input) -> output_grid.
At inference, the model adapts to a new task by performing K gradient
steps on the demonstration pairs, then uses the updated parameters
to predict the test output.

This is deliberately simple — it exists to provide a clean, legitimate
comparison with the context model, NOT to be artificially sabotaged.

Training phase:
  - The model learns a good initialization via meta-learning (Reptile).
  - For each task in a meta-batch, K inner gradient steps are taken on
    demo pairs, and the meta-parameters are updated towards the adapted weights.

Inference phase:
  - Demo pairs are processed via forward + backward passes.
  - Parameters are updated K times via SGD.
  - The updated model predicts the test output.
  - Original base parameters are restored (leaving base model intact).
"""

import copy
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import GRID_FEATURES, GRID_CELLS, NUM_COLORS

TEST_INPUT_DIM = GRID_FEATURES        # 75
DEMO_PAIR_DIM = GRID_FEATURES * 2     # 150
OUTPUT_DIM = GRID_CELLS * NUM_COLORS  # 75


class OptimizationModel(nn.Module):
    """MLP model that adapts via gradient descent at inference time.

    The model takes a test input grid and predicts the output grid.
    Task-specific knowledge comes from gradient updates on demo pairs.

    Parameters
    ----------
    hidden_dim : int
        Hidden layer dimension.
    num_layers : int
        Number of hidden layers (default 3).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build single unified MLP for both demos and test prediction
        layers = []
        in_dim = TEST_INPUT_DIM  # 75

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, OUTPUT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Predict output grid from input grid.

        Parameters
        ----------
        x : Tensor (B, 75)

        Returns
        -------
        logits : Tensor (B, 25, 3)
        """
        raw = self.mlp(x)  # (B, 75)
        return raw.view(-1, GRID_CELLS, NUM_COLORS)  # (B, 25, 3)

    def compute_demo_loss(self, demo_pairs, criterion):
        """Compute loss on demonstration pairs for adaptation.

        Parameters
        ----------
        demo_pairs : Tensor (D, 150)
            Concatenated (input, output) for each demo.
        criterion : loss function

        Returns
        -------
        loss : scalar tensor
        """
        demo_inputs = demo_pairs[:, :GRID_FEATURES]  # (D, 75)
        demo_targets_onehot = demo_pairs[:, GRID_FEATURES:]  # (D, 75)

        # Convert one-hot targets to class indices
        demo_targets = demo_targets_onehot.view(-1, GRID_CELLS, NUM_COLORS).argmax(dim=-1)  # (D, 25)

        # Predict using the shared MLP
        logits = self.forward(demo_inputs)  # (D, 25, 3)
        loss = criterion(logits.reshape(-1, NUM_COLORS), demo_targets.reshape(-1))
        return loss

    def adapt_and_predict(self, demo_pairs, test_input, K=5,
                          inner_lr=0.01, criterion=None):
        """Inference-time adaptation via gradient descent.

        1. Clone model parameters (in-memory tensor clone).
        2. Perform K gradient steps on the demo pairs.
        3. Predict test output with updated parameters.
        4. Restore original parameters.

        Parameters
        ----------
        demo_pairs : Tensor (D, 150)
        test_input : Tensor (1, 75)
        K : int
            Number of gradient steps.
        inner_lr : float
            Learning rate for inner adaptation.
        criterion : loss function (default: CrossEntropyLoss)

        Returns
        -------
        dict with logits, loss_curve, param_change_magnitude
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Save original parameters efficiently
        original_state = {k: v.clone() for k, v in self.state_dict().items()}

        loss_curve = []

        # Record initial parameter norm
        initial_param_norm = sum(
            p.data.norm().item() ** 2 for p in self.parameters()
        ) ** 0.5

        # Perform K gradient steps on demonstrations
        if K > 0:
            self.train()
            inner_optimizer = torch.optim.SGD(self.parameters(), lr=inner_lr)

            for step in range(K):
                inner_optimizer.zero_grad()
                loss = self.compute_demo_loss(demo_pairs, criterion)
                loss.backward()
                inner_optimizer.step()
                loss_curve.append(loss.item())

        # Predict with adapted parameters
        self.eval()
        with torch.no_grad():
            logits = self.forward(test_input)  # (1, 25, 3)

        # Compute parameter change magnitude
        final_param_norm = sum(
            p.data.norm().item() ** 2 for p in self.parameters()
        ) ** 0.5

        param_change = 0.0
        current_state = self.state_dict()
        for key in original_state:
            diff = (current_state[key].float() - original_state[key].float())
            param_change += diff.norm().item() ** 2
        param_change = param_change ** 0.5

        # Restore original base parameters
        self.load_state_dict(original_state)

        return {
            'logits': logits,
            'loss_curve': loss_curve,
            'param_change_magnitude': param_change,
            'initial_param_norm': initial_param_norm,
            'final_param_norm': final_param_norm,
            'parameters_changed': K > 0,
            'gradient_steps': K,
        }

"""
Context-route model: fixed-size recurrent state for inference-time adaptation.

Architecture
------------
1. Demo Encoder (MLP):
   demo_pair (150) → hidden → hidden_dim

2. State Updater (GRU cell):
   GRUCell(input=hidden_dim, hidden=state_dim)
   Processes demos sequentially: h0 → h1 → h2 → ... → hN

3. Predictor (MLP):
   concat(hN, test_input_encoded) → hidden → 25×3 logits

Key property: at inference, ONLY the recurrent state h changes.
Model parameters (weights) are frozen. No optimizer, no backward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import GRID_FEATURES, GRID_CELLS, NUM_COLORS

# Input dimensions
DEMO_PAIR_DIM = GRID_FEATURES * 2  # 150 (input grid + output grid)
TEST_INPUT_DIM = GRID_FEATURES      # 75
OUTPUT_DIM = GRID_CELLS * NUM_COLORS  # 75 (25 cells × 3 colors)


class ContextModel(nn.Module):
    """Recurrent context model with fixed-size state vector.

    Parameters
    ----------
    state_dim : int
        Dimension of the recurrent state vector (e.g. 32, 64, 128, 256, 512).
    hidden_dim : int
        Hidden dimension for encoder and predictor MLPs.
    num_gru_layers : int
        Number of stacked GRU layers (default 1).
    dropout : float
        Dropout rate for encoder/predictor (default 0.1).
    """

    def __init__(self, state_dim=128, hidden_dim=256, num_gru_layers=1,
                 dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        # --- Demo Encoder ---
        # Maps a concatenated (input_grid, output_grid) pair → hidden repr
        self.demo_encoder = nn.Sequential(
            nn.Linear(DEMO_PAIR_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- State Updater (GRU) ---
        # Processes encoded demos sequentially, updating a fixed-size state
        self.gru = nn.GRUCell(hidden_dim, state_dim)

        # For multi-layer GRU, we stack GRUCells
        if num_gru_layers > 1:
            self.extra_gru_layers = nn.ModuleList([
                nn.GRUCell(state_dim, state_dim)
                for _ in range(num_gru_layers - 1)
            ])
        else:
            self.extra_gru_layers = None

        # --- Predictor ---
        # Maps (final_state, test_input) → output grid logits
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + TEST_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

    def init_state(self, batch_size=1, device=None):
        """Initialize the recurrent state to zeros."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.state_dim, device=device)

    def process_demo(self, state, demo_encoded):
        """Process one encoded demo and return the updated state.

        Parameters
        ----------
        state : Tensor (B, state_dim)
            Current recurrent state.
        demo_encoded : Tensor (B, hidden_dim)
            Output of demo_encoder for one demo pair.

        Returns
        -------
        new_state : Tensor (B, state_dim)
            Updated state after processing this demo.
        """
        new_state = self.gru(demo_encoded, state)
        if self.extra_gru_layers is not None:
            for layer in self.extra_gru_layers:
                new_state = layer(new_state, new_state)
        return new_state

    def encode_demo(self, demo_pair):
        """Encode a demo pair tensor through the demo encoder MLP.

        Parameters
        ----------
        demo_pair : Tensor (..., 150)

        Returns
        -------
        encoded : Tensor (..., hidden_dim)
        """
        return self.demo_encoder(demo_pair)

    def predict(self, state, test_input):
        """Predict output grid from final state and test input.

        Parameters
        ----------
        state : Tensor (B, state_dim)
            Final recurrent state after processing all demos.
        test_input : Tensor (B, 75)
            One-hot encoded test input grid.

        Returns
        -------
        logits : Tensor (B, 25, 3)
            Per-cell color logits.
        """
        combined = torch.cat([state, test_input], dim=-1)
        raw = self.predictor(combined)  # (B, 75)
        return raw.view(-1, GRID_CELLS, NUM_COLORS)  # (B, 25, 3)

    def forward(self, demo_pairs_list, test_inputs):
        """Full forward pass: process demos sequentially, then predict.

        Parameters
        ----------
        demo_pairs_list : list of Tensor
            Each element has shape (D_i, 150) where D_i is the number
            of demos for puzzle i. Length of list = batch size B.
        test_inputs : Tensor (B, 75)
            Encoded test input grids.

        Returns
        -------
        logits : Tensor (B, 25, 3)
            Per-cell color logits.
        final_states : Tensor (B, state_dim)
            Final recurrent states (useful for inspection).
        """
        batch_size = len(demo_pairs_list)
        device = test_inputs.device

        state = self.init_state(batch_size, device)

        # Find max demo count for sequential processing
        max_demos = max(d.shape[0] for d in demo_pairs_list)

        for step in range(max_demos):
            # Build batch for this step: only include puzzles that have
            # a demo at this step index
            mask = []
            step_demos = []
            for i, demos in enumerate(demo_pairs_list):
                if step < demos.shape[0]:
                    mask.append(i)
                    step_demos.append(demos[step])

            if not step_demos:
                break

            step_batch = torch.stack(step_demos)  # (K, 150)
            step_encoded = self.encode_demo(step_batch)  # (K, hidden_dim)

            # Update only the states that have demos at this step
            mask_t = torch.tensor(mask, device=device)
            selected_states = state[mask_t]  # (K, state_dim)
            new_states = self.process_demo(selected_states, step_encoded)
            state = state.clone()
            state[mask_t] = new_states

        logits = self.predict(state, test_inputs)
        return logits, state

    def forward_with_state_trace(self, demo_pairs, test_input):
        """Single-instance forward pass that returns state after each demo.

        For inspection / educational component.

        Parameters
        ----------
        demo_pairs : Tensor (D, 150)
            Encoded demo pairs for one puzzle.
        test_input : Tensor (75,)
            Encoded test input.

        Returns
        -------
        logits : Tensor (25, 3)
        states : list of Tensor, each (state_dim,)
            State after each demo (length D).
        """
        device = demo_pairs.device
        state = self.init_state(1, device).squeeze(0)  # (state_dim,)
        states = []

        for i in range(demo_pairs.shape[0]):
            demo_enc = self.encode_demo(demo_pairs[i:i+1])  # (1, hidden_dim)
            state = self.process_demo(
                state.unsqueeze(0), demo_enc
            ).squeeze(0)  # (state_dim,)
            states.append(state.detach().clone())

        logits = self.predict(
            state.unsqueeze(0), test_input.unsqueeze(0)
        ).squeeze(0)  # (25, 3)

        return logits, states

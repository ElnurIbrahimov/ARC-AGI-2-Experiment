"""
BroadMind Adapter for ARC-AGI-2
================================

Wraps BroadMind v0.77 (Elastic Inference) for ARC DSL program execution.

BroadMind is a neural program executor with:
- ElasticSolver: wisdom-guided execution with Mixture of Recursions
- WisdomBank: stored procedural knowledge for task families
- Halter: adaptive compute — stops when execution converges
- Elastic width/depth for hardware-adaptive inference

This adapter bridges the dimensional gap:
  Our 7B model:  d_model=4096, 50 DSL operations
  BroadMind:     d_model=192, n_variables=3, num_operations=13

Key mappings:
  - ARC grid states (4096-dim embeddings) -> BroadMind state space (192-dim)
  - 50 DSL primitives -> BroadMind operation indices (13 ops via learned projection)
  - ARC task families -> BroadMind wisdom bank (4 families, 5 wisdom slots)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Try importing BroadMind from source
try:
    sys.path.insert(0, r'C:\Users\asus\Desktop\BroadMind')
    from BroadMind_v077_elastic import (
        BroadMindV077,
        Config as BroadMindConfig,
        ElasticSolver,
        WisdomBank,
        N_OPS as BROADMIND_N_OPS,
    )
    BROADMIND_AVAILABLE = True
except ImportError:
    BROADMIND_AVAILABLE = False
    BROADMIND_N_OPS = 13  # 12 ops + PAD

    class BroadMindConfig:
        """Stub config matching BroadMind v0.77 defaults."""
        n_variables = 3
        n_task_families = 4
        max_program_length = 4
        max_eval_length = 16
        value_range = 10
        comparison_scale = 30.0
        d_model = 192
        d_latent = 96
        d_wisdom = 48
        n_wisdom_slots = 5
        halt_threshold = 0.5
        distill_batch_size = 32
        batch_size = 256
        n_iterations_phase1 = 2500
        n_iterations_phase2 = 1500
        n_iterations_phase2b = 800
        n_iterations_phase3 = 1500
        n_iterations_phase4 = 1500
        n_iterations_phase5 = 500
        noise_std = 0.05
        lr = 1e-3
        lr_fine = 1e-4
        grad_clip = 1.0
        dropout = 0.1
        weight_decay = 1e-4
        max_recursion_depth = 4
        recursion_enc_dim = 24
        compute_cost_weight = 0.005
        width_multipliers = [0.25, 0.50, 0.75, 1.0]
        self_distill_weight = 2.0
        n_iterations_phase6 = 2500
        n_iterations_phase7 = 1000
        n_iterations_phase8 = 300
        lr_elastic = 2e-5
        latency_target_ms = 10.0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class BroadMindV077(nn.Module):
        """Stub BroadMind when source is not available."""
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.wisdom_bank = nn.Module()
            self.wisdom_bank.wisdom_codes = nn.Parameter(
                torch.randn(config.n_wisdom_slots, config.d_wisdom) * 0.1
            )
            self._pred_layer = nn.Linear(config.n_variables, config.n_variables)

        def get_wisdom(self, programs, initial_states):
            batch_size = initial_states.shape[0]
            wisdom = torch.zeros(batch_size, self.config.d_wisdom,
                                 device=initial_states.device)
            attention = torch.ones(batch_size, self.config.n_wisdom_slots,
                                   device=initial_states.device) / self.config.n_wisdom_slots
            return wisdom, attention

        def forward_all_steps(self, programs, initial_states, wisdom=None,
                              training_noise_std=0.0, width_mult=1.0, depth_mask=None):
            batch_size = programs.shape[0]
            n_steps = programs.shape[1]
            if wisdom is None:
                wisdom, _ = self.get_wisdom(programs, initial_states)
            preds = initial_states.unsqueeze(1).expand(-1, n_steps, -1)
            halt_logits = torch.zeros(batch_size, n_steps, 1, device=initial_states.device)
            return preds, halt_logits, wisdom, 0.0

        def forward_adaptive(self, programs, initial_states, **kwargs):
            batch_size = programs.shape[0]
            pred = initial_states
            steps_used = torch.ones(batch_size, device=initial_states.device)
            return pred, steps_used, 0.0


# ─── Output container ───────────────────────────────────────────────

@dataclass
class BroadMindResult:
    """Output from BroadMind program execution."""
    predictions: torch.Tensor       # (batch, n_steps, broadmind_n_vars) BroadMind predictions
    compute_cost: float             # total compute cost from MoR routing
    steps_used: int                 # number of program steps executed
    halt_confidence: float          # mean halting confidence
    wisdom: torch.Tensor            # (batch, d_wisdom) matched wisdom


# ─── Adapter ────────────────────────────────────────────────────────

class BroadMindAdapter(nn.Module):
    """
    Wraps BroadMind v0.77 for ARC-AGI-2 DSL program execution.

    Handles dimensional mapping between our 7B model and BroadMind:
    - Grid state projection: 4096 -> 192 (with reverse projection 192 -> 4096)
    - Operation mapping: 50 DSL ops -> BroadMind's 13 op indices
    - Wisdom extraction: get task-specific wisdom from BroadMind's bank

    Uses BroadMind's adaptive halting to decide when program execution is done,
    and elastic width/depth for compute-efficient inference.

    Parameter budget: ~2-4M (projection layers + op mapper).
    """

    def __init__(
        self,
        d_model: int = 4096,
        broadmind_d_model: int = 192,
        num_dsl_ops: int = 50,
        num_broadmind_ops: int = None,
        broadmind_config: object = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model:           Hidden dim of our 7B backbone.
            broadmind_d_model: BroadMind's internal d_model (192).
            num_dsl_ops:       Number of ARC DSL primitives (50).
            num_broadmind_ops: BroadMind's op count (auto-detected from config).
            broadmind_config:  BroadMind Config object. None uses defaults.
            dropout:           Dropout rate for projection layers.
        """
        super().__init__()
        self.d_model = d_model
        self.broadmind_d_model = broadmind_d_model
        self.num_dsl_ops = num_dsl_ops

        # Initialize BroadMind config
        self.bm_config = broadmind_config or BroadMindConfig()
        self.num_broadmind_ops = num_broadmind_ops or BROADMIND_N_OPS

        # Core BroadMind module
        self.broadmind = BroadMindV077(self.bm_config)

        # Grid state projection: backbone hidden -> BroadMind state space
        # Two-stage: 4096 -> 512 -> 192 (gradual compression)
        intermediate_dim = max(broadmind_d_model * 2, 512)
        self.state_projector = nn.Sequential(
            nn.Linear(d_model, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, broadmind_d_model),
            nn.LayerNorm(broadmind_d_model),
        )

        # Reverse projection: BroadMind output -> backbone space
        # Used when feeding BroadMind results back to the backbone
        self.state_unprojector = nn.Sequential(
            nn.Linear(broadmind_d_model, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Operation mapping: 50 DSL ops -> BroadMind's op space
        # Learned embedding for each DSL op, then project to BroadMind op logits
        self.dsl_op_embedding = nn.Embedding(num_dsl_ops, 64)
        self.op_mapper = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, self.num_broadmind_ops),
        )

        # Wisdom projection: adapt BroadMind wisdom to our backbone's dim
        self.wisdom_projector = nn.Sequential(
            nn.Linear(self.bm_config.d_wisdom, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

        # State adapter: our backbone produces d_model-dim embeddings for grid states,
        # but BroadMind's solver expects n_variables-dim (3) state vectors.
        # This is a learned compression of the grid embedding to BroadMind's state space.
        self.grid_to_state = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.bm_config.n_variables),
        )

        # Reverse: BroadMind state predictions back to d_model
        self.state_to_grid = nn.Sequential(
            nn.Linear(self.bm_config.n_variables, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

    def encode_grid_state(self, grid_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project from backbone hidden_dim to BroadMind's d_model.

        Args:
            grid_embedding: (batch, d_model) from our 7B model.

        Returns:
            (batch, broadmind_d_model) projected state.
        """
        return self.state_projector(grid_embedding)

    def encode_operations(self, op_ids: torch.Tensor) -> torch.Tensor:
        """
        Map ARC DSL operation IDs to BroadMind operation indices.

        Uses soft mapping (logits) during training, hard argmax at eval.

        Args:
            op_ids: (batch, seq_len) DSL primitive token IDs (0-indexed into the 50 ops).

        Returns:
            (batch, seq_len) BroadMind operation indices.
        """
        # Embed and project
        op_emb = self.dsl_op_embedding(op_ids)           # (batch, seq_len, 64)
        op_logits = self.op_mapper(op_emb)               # (batch, seq_len, num_broadmind_ops)

        if self.training:
            # Gumbel-softmax for differentiable discrete selection
            op_soft = F.gumbel_softmax(op_logits, tau=1.0, hard=True)
            # Convert one-hot back to indices
            op_indices = op_soft.argmax(dim=-1)
        else:
            op_indices = op_logits.argmax(dim=-1)

        return op_indices

    def execute_program(
        self,
        grid_embedding: torch.Tensor,
        op_sequence: torch.Tensor,
        wisdom: Optional[torch.Tensor] = None,
        width_mult: float = 1.0,
        depth_mask: Optional[torch.Tensor] = None,
        use_adaptive_halt: bool = False,
    ) -> BroadMindResult:
        """
        Execute a DSL program using BroadMind's elastic solver.

        Args:
            grid_embedding: (batch, d_model) backbone hidden state for the grid.
            op_sequence:    (batch, seq_len) DSL operation IDs (0-indexed, our 50 ops).
            wisdom:         (batch, d_wisdom) pre-computed wisdom. None = auto-match.
            width_mult:     Elastic width (0.25, 0.5, 0.75, 1.0).
            depth_mask:     BoolTensor for elastic depth.
            use_adaptive_halt: If True, use adaptive halting instead of running all steps.

        Returns:
            BroadMindResult with predictions, compute cost, and wisdom.
        """
        # Map operations to BroadMind space
        bm_ops = self.encode_operations(op_sequence)

        # Compress grid embedding to BroadMind's state vector
        initial_states = self.grid_to_state(grid_embedding)

        # Get wisdom if not provided
        if wisdom is None:
            wisdom, _ = self.broadmind.get_wisdom(bm_ops, initial_states)

        if use_adaptive_halt:
            # Adaptive halting: BroadMind decides when execution is done
            pred, steps_used, compute_cost = self.broadmind.forward_adaptive(
                bm_ops, initial_states,
                width_mult=width_mult,
                depth_mask=depth_mask,
            )
            # pred is (batch, n_variables) — final state only
            n_steps = int(steps_used.max().item())
            halt_conf = 1.0  # adaptive means it halted with confidence
            # Reshape to match (batch, 1, n_variables) for consistency
            predictions = pred.unsqueeze(1)
        else:
            # Run all steps
            preds, halt_logits, wisdom, compute_cost = self.broadmind.forward_all_steps(
                bm_ops, initial_states, wisdom=wisdom,
                width_mult=width_mult,
                depth_mask=depth_mask,
            )
            predictions = preds
            n_steps = op_sequence.shape[1]
            # Mean halt probability across steps
            halt_conf = torch.sigmoid(halt_logits).mean().item()

        return BroadMindResult(
            predictions=predictions,
            compute_cost=float(compute_cost) if torch.is_tensor(compute_cost) else compute_cost,
            steps_used=n_steps,
            halt_confidence=halt_conf,
            wisdom=wisdom,
        )

    def get_wisdom(
        self,
        grid_embedding: torch.Tensor,
        op_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get task-specific wisdom from BroadMind's wisdom bank.

        Projects the result to our backbone's d_model for integration
        with the main model.

        Args:
            grid_embedding: (batch, d_model) backbone hidden state.
            op_sequence:    (batch, seq_len) DSL operation IDs.

        Returns:
            (batch, d_model) wisdom projected to backbone space.
        """
        # Map to BroadMind space
        bm_ops = self.encode_operations(op_sequence)
        initial_states = self.grid_to_state(grid_embedding)

        # Get wisdom
        wisdom, attention = self.broadmind.get_wisdom(bm_ops, initial_states)

        # Project to backbone space
        return self.wisdom_projector(wisdom)

    def decode_predictions(self, bm_predictions: torch.Tensor) -> torch.Tensor:
        """
        Project BroadMind state predictions back to backbone space.

        Args:
            bm_predictions: (batch, n_steps, n_variables) or (batch, n_variables)
                            BroadMind predictions.

        Returns:
            (batch, [n_steps,] d_model) projected to backbone hidden dim.
        """
        if bm_predictions.dim() == 3:
            batch, n_steps, n_vars = bm_predictions.shape
            flat = bm_predictions.reshape(batch * n_steps, n_vars)
            projected = self.state_to_grid(flat)
            return projected.reshape(batch, n_steps, self.d_model)
        else:
            return self.state_to_grid(bm_predictions)

    def get_diagnostics(self) -> Dict:
        """Return adapter diagnostics."""
        n_adapter_params = sum(p.numel() for p in self.parameters())
        n_broadmind_params = sum(p.numel() for p in self.broadmind.parameters())
        return {
            "adapter_total_params": n_adapter_params,
            "adapter_total_params_human": f"{n_adapter_params / 1e6:.2f}M",
            "broadmind_params": n_broadmind_params,
            "broadmind_available": BROADMIND_AVAILABLE,
            "d_model": self.d_model,
            "broadmind_d_model": self.broadmind_d_model,
            "num_dsl_ops": self.num_dsl_ops,
            "num_broadmind_ops": self.num_broadmind_ops,
        }

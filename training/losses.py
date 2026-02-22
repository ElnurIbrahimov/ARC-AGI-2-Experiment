"""
Combined loss function for ARC-AGI-2 training.

Components:
1. DSL token cross-entropy (with label smoothing, PAD ignored)
2. Grid reconstruction MSE (auxiliary, Stage 2+)
3. MoE load-balancing auxiliary loss (Switch Transformer formulation)
4. Causeway structural regularization (NOTEARS acyclicity + L1 sparsity, Stage 3)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARCLoss(nn.Module):
    """
    Combined loss for ARC-AGI-2 training.

    Components:
    1. dsl_token_loss: Cross-entropy on DSL token prediction
       - Ignore PAD tokens (id=0)
       - Label smoothing 0.1

    2. grid_loss: Grid reconstruction auxiliary loss
       - MSE between predicted grid reconstruction and target grid
       - Only active in Stage 2+ (grid_weight > 0)
       - Helps model learn grid structure

    3. moe_aux_loss: Load balancing for MoE routing
       - Importance loss: variance of expert importance scores
       - Load balance loss: dot product of routing fractions and expert fractions
       - Standard Switch Transformer formulation

    4. causeway_structural_loss: From Causeway regularization
       - DAG acyclicity: tr(e^A) - d (NOTEARS)
       - Sparsity: L1 on adjacency weights
       - Only active in Stage 3 (causeway_weight > 0)

    Stage-specific weights:
    - Stage 1 (pretrain):      dsl=1.0, grid=0.0, moe=0.01, causeway=0.0
    - Stage 2 (finetune):      dsl=1.0, grid=0.1, moe=0.01, causeway=0.0
    - Stage 3 (integration):   dsl=0.5, grid=0.1, moe=0.01, causeway=0.05
    """

    def __init__(
        self,
        dsl_weight: float = 1.0,
        grid_weight: float = 0.0,
        moe_weight: float = 0.01,
        causeway_weight: float = 0.0,
        label_smoothing: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.dsl_weight = dsl_weight
        self.grid_weight = grid_weight
        self.moe_weight = moe_weight
        self.causeway_weight = causeway_weight
        self.label_smoothing = label_smoothing
        self.pad_token_id = pad_token_id

    @classmethod
    def for_stage(cls, stage: int) -> "ARCLoss":
        """Factory method: create loss with appropriate weights for training stage."""
        if stage == 1:
            return cls(
                dsl_weight=1.0, grid_weight=0.0,
                moe_weight=0.01, causeway_weight=0.0,
            )
        elif stage == 2:
            return cls(
                dsl_weight=1.0, grid_weight=0.1,
                moe_weight=0.01, causeway_weight=0.0,
            )
        elif stage == 3:
            return cls(
                dsl_weight=0.5, grid_weight=0.1,
                moe_weight=0.01, causeway_weight=0.05,
            )
        else:
            raise ValueError(f"Unknown training stage: {stage}. Expected 1, 2, or 3.")

    # ------------------------------------------------------------------
    # Individual loss components
    # ------------------------------------------------------------------

    def _dsl_token_loss(
        self, logits: torch.Tensor, target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy with label smoothing, ignoring PAD positions.

        Args:
            logits: (B, T, V) raw logits from model
            target_tokens: (B, T) ground-truth token IDs
        Returns:
            scalar loss tensor (0.0 if all tokens are PAD)
        """
        B, T, V = logits.shape

        # Mask: True where target is NOT pad
        non_pad_mask = target_tokens.reshape(B * T) != self.pad_token_id
        if non_pad_mask.sum() == 0:
            # All tokens are PAD -- nothing to learn from
            return torch.tensor(0.0, device=logits.device)

        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            target_tokens.reshape(B * T),
            ignore_index=self.pad_token_id,
            label_smoothing=self.label_smoothing,
        )
        return loss

    def _grid_loss(
        self,
        grid_reconstruction: Optional[torch.Tensor],
        target_grid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        MSE between predicted and target grids.

        Both tensors can be any shape as long as they match -- typically
        (B, H, W) with integer color values cast to float, or (B, H, W, C)
        if using a one-hot / embedding representation.

        Returns 0 if either input is None.
        """
        if grid_reconstruction is None or target_grid is None:
            return torch.tensor(0.0, device=self._current_device())

        # Ensure float for MSE
        pred = grid_reconstruction.float()
        target = target_grid.float()
        return F.mse_loss(pred, target)

    @staticmethod
    def _moe_aux_loss(aux_loss: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Pass through the MoE auxiliary loss computed inside the MoE layer.

        The model's MoE routing layer already computes the Switch Transformer
        load-balance loss (N * sum(f_i * P_i) where f_i = fraction of tokens
        routed to expert i, P_i = mean routing probability for expert i).
        We just forward it here so weighting happens in one place.
        """
        if aux_loss is None:
            return torch.tensor(0.0, device=device)
        return aux_loss

    @staticmethod
    def _causeway_structural_loss(
        causeway_reg_losses: Optional[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Causeway DAG regularization.

        Expected keys in causeway_reg_losses:
          - 'acyclicity': tr(e^{A circ A}) - d   (NOTEARS constraint)
          - 'sparsity':   ||A||_1                 (L1 on adjacency)

        Returns 0 if dict is None or empty.
        """
        if causeway_reg_losses is None:
            return torch.tensor(0.0, device=device)

        acyclicity = causeway_reg_losses.get(
            "acyclicity", torch.tensor(0.0, device=device),
        )
        sparsity = causeway_reg_losses.get(
            "sparsity", torch.tensor(0.0, device=device),
        )
        return acyclicity + sparsity

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        causeway_reg_losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components and return weighted total.

        Args:
            model_output: {
                'logits': (B, T, V) DSL token logits,
                'aux_loss': scalar MoE auxiliary loss (optional),
                'grid_reconstruction': optional (B, H, W) predicted grid
            }
            targets: {
                'target_tokens': (B, T) target DSL token IDs,
                'target_grid': optional (B, H, W) target grid
            }
            causeway_reg_losses: optional {
                'acyclicity': scalar,
                'sparsity': scalar
            }

        Returns: {
            'total': weighted sum of all components,
            'dsl_token': CE loss value,
            'grid': grid MSE value,
            'moe_aux': MoE balance value,
            'causeway_structural': causeway reg value
        }
        """
        device = model_output["logits"].device

        # 1. DSL token cross-entropy
        dsl_token_loss = self._dsl_token_loss(
            model_output["logits"],
            targets["target_tokens"],
        )

        # 2. Grid reconstruction MSE
        grid_loss = self._grid_loss(
            model_output.get("grid_reconstruction"),
            targets.get("target_grid"),
        )

        # 3. MoE auxiliary load-balance loss
        moe_aux_loss = self._moe_aux_loss(
            model_output.get("aux_loss"),
            device,
        )

        # 4. Causeway structural regularization
        causeway_structural_loss = self._causeway_structural_loss(
            causeway_reg_losses,
            device,
        )

        # Weighted total
        total = (
            self.dsl_weight * dsl_token_loss
            + self.grid_weight * grid_loss
            + self.moe_weight * moe_aux_loss
            + self.causeway_weight * causeway_structural_loss
        )

        return {
            "total": total,
            "dsl_token": dsl_token_loss,
            "grid": grid_loss,
            "moe_aux": moe_aux_loss,
            "causeway_structural": causeway_structural_loss,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_device(self) -> torch.device:
        """Best-effort device detection for zero-tensors."""
        # nn.Module with no parameters defaults to CPU, which is fine
        # for the 0.0 sentinel tensors. During forward() the actual
        # device comes from model_output['logits'].
        return torch.device("cpu")

    def extra_repr(self) -> str:
        return (
            f"dsl_weight={self.dsl_weight}, grid_weight={self.grid_weight}, "
            f"moe_weight={self.moe_weight}, causeway_weight={self.causeway_weight}, "
            f"label_smoothing={self.label_smoothing}, pad_token_id={self.pad_token_id}"
        )

"""
Vector-Quantized Contrastive Predictive Coding (VQ-CPC) pretext task.

Architecture overview:
  raw sequence
      │
  [Backbone encoder]  ← shared, trained by pipeline
      │ z (continuous latents)
  [VQ Codebook]       ← owned by this task
      │ z_q (quantized latents)
  [AR model (GRU)]    ← owned by this task
      │ c (context vector)
  [Predictors]        ← one per future step, owned by this task
      │ predicted z_q
  InfoNCE loss against negatives sampled from the batch

The VQ bottleneck encourages discrete, structured representations and
provides an implicit clustering objective via the codebook commitment loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from typing import Any

from .base import MultiPassTask, PretextBatch


class VectorQuantizer(nn.Module):
    """
    Straight-through estimator VQ layer.

    Forward pass:
      1. Find nearest codebook entry for each latent vector.
      2. Return quantized vectors (gradients flow through via straight-through).
      3. Return commitment loss: ||z - sg[z_q]||² + β||sg[z] - z_q||²
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, T, D) continuous encoder outputs

        Returns:
            z_q:              (B, T, D) quantized latents (straight-through gradient)
            commitment_loss:  scalar
            encoding_indices: (B, T) codebook indices (for logging/analysis)
        """
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*T, D)

        # Distances to codebook entries: ||z - e||² = ||z||² - 2z·e + ||e||²
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )  # (B*T, num_embeddings)

        encoding_indices = distances.argmin(1)  # (B*T,)
        z_q_flat = self.codebook(encoding_indices)  # (B*T, D)
        z_q = z_q_flat.reshape(B, T, D)

        # Commitment loss
        commitment_loss = (
            self.commitment_cost * F.mse_loss(z_q.detach(), z)    # encoder commit
            + F.mse_loss(z_q, z.detach())                          # codebook update
        )

        # Straight-through: copy z_q value but route gradients through z
        z_q_st = z + (z_q - z).detach()

        return z_q_st, commitment_loss, encoding_indices.reshape(B, T)

    def perplexity(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """Codebook utilization metric. Perfect uniform use = num_embeddings."""
        one_hot = F.one_hot(encoding_indices.reshape(-1), self.num_embeddings).float()
        avg_probs = one_hot.mean(0)
        return torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))


class VQCPC(MultiPassTask):
    """
    VQ-CPC pretext task.

    The backbone encoder is passed in from outside (owned by the pipeline).
    This task owns: VQ codebook, AR context model, per-step predictors.

    Args:
        encoder_dim:     Output dimensionality of the backbone encoder.
        context_dim:     Hidden size of the GRU context model.
        num_embeddings:  Codebook size (number of discrete codes).
        pred_steps:      Number of future timesteps to predict.
        n_negatives:     Number of negative samples per positive in InfoNCE.
        commitment_cost: Weight of the commitment loss term in VQ.
        vq_loss_weight:  Weight of the VQ commitment loss relative to CPC loss.
        ar_layers:       Number of GRU layers.
    """

    def __init__(
        self,
        encoder_dim: int,
        context_dim: int,
        num_embeddings: int = 512,
        pred_steps: int = 4,
        n_negatives: int = 16,
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        ar_layers: int = 1,
    ):
        super().__init__()
        self.pred_steps = pred_steps
        self.n_negatives = n_negatives
        self.vq_loss_weight = vq_loss_weight

        self.vq = VectorQuantizer(num_embeddings, encoder_dim, commitment_cost)

        self.ar_model = nn.GRU(
            input_size=encoder_dim,
            hidden_size=context_dim,
            num_layers=ar_layers,
            batch_first=True,
        )

        # One linear predictor per future step (predict in quantized space)
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, encoder_dim) for _ in range(pred_steps)
        ])

    # ------------------------------------------------------------------
    # PretextTask interface
    # ------------------------------------------------------------------

    def transform(self, raw_batch: Any) -> PretextBatch:
        """
        Expects raw_batch to be either:
          - (sequences, labels)  where sequences is (B, T, features)
          - just sequences        (B, T, features)
        No augmentation is applied; the VQ bottleneck provides the
        implicit regularization that drives representation learning.
        """
        if isinstance(raw_batch, (tuple, list)):
            sequences, _, metadata = raw_batch  # Expect (data, label, metadata)
        else:
            sequences = raw_batch
            metadata = {}

        # Handle (B, C, T, F) -> (B, T, F) if C=1
        if sequences.ndim == 4:
             sequences = sequences.squeeze(1)

        return PretextBatch(inputs=sequences, targets=None, metadata=metadata)

    def compute_loss_with_model(
        self, encoder: nn.Module, batch: PretextBatch
    ) -> torch.Tensor:
        """
        Full VQ-CPC forward pass.

        Returns the combined loss: InfoNCE + vq_loss_weight * commitment_loss
        Also stores diagnostic metrics in batch.metadata for logging.
        """
        sequences = batch.inputs  # (B, T, feature_dim)
        device = sequences.device
        B, T, _ = sequences.shape

        # ── 1. Encode every timestep ──────────────────────────────────────
        # Flatten time into batch dim, encode, reshape back
        flat = sequences.reshape(B * T, -1)        # (B*T, feature_dim)
        z_flat = encoder(flat)                      # (B*T, encoder_dim)
        z = z_flat.reshape(B, T, -1)               # (B, T, encoder_dim)

        # ── 2. Vector quantize ───────────────────────────────────────────
        z_q, commitment_loss, indices = self.vq(z) # z_q: (B, T, encoder_dim)

        # ── 3. AR context model ──────────────────────────────────────────
        context_len = T - self.pred_steps
        if context_len <= 0:
            raise ValueError(
                f"Sequence length {T} must be > pred_steps ({self.pred_steps}). "
                "Use a longer window or fewer prediction steps."
            )
        c, _ = self.ar_model(z_q[:, :context_len, :])  # (B, context_len, context_dim)

        # Use final context step for prediction
        # (extend to use all context positions if needed for more signal)
        c_t = c[:, -1, :]  # (B, context_dim)

        # ── 4. InfoNCE loss over pred_steps ──────────────────────────────
        cpc_loss = torch.tensor(0.0, device=device)
        correct = 0
        total = 0

        # Precompute negatives: sample from quantized latents across batch
        z_q_pool = z_q.reshape(B * T, -1)  # (B*T, encoder_dim)

        for k, predictor in enumerate(self.predictors, start=1):
            target_idx = context_len + k - 1
            if target_idx >= T:
                break

            z_future = z_q[:, target_idx, :]      # (B, encoder_dim) — positive
            predicted = predictor(c_t)             # (B, encoder_dim)

            # Negatives: B * n_negatives random latents from the pool
            neg_flat_idx = torch.randint(0, B * T, (B * self.n_negatives,), device=device)
            z_neg = z_q_pool[neg_flat_idx].reshape(B, self.n_negatives, -1)  # (B, N, D)

            # Scores
            pos_score = (predicted * z_future).sum(-1, keepdim=True)           # (B, 1)
            neg_scores = torch.bmm(z_neg, predicted.unsqueeze(-1)).squeeze(-1) # (B, N)
            
            # --- Cross-User Negative Sampling ---
            # If 'user' metadata is available, mask out negatives from the same user
            # We need to map z_neg back to their original batch index to check user identity
            
            # This is tricky with random sampling from z_q_pool.
            # Alternative: Use Batch Negatives (score against all other batch elements at current step)
            # This allows easy user masking.
            
            # Let's switch to Batch Negatives for this step (consistent with our previous implementation)
            # Logits: (B, B) where (i, j) is score of predicting z_j from c_i
            logits = torch.matmul(predicted, z_future.T) # (B, D) @ (D, B) -> (B, B)
            
            if 'user' in batch.metadata and isinstance(batch.metadata['user'], (list, tuple)):
                users = [str(u) for u in batch.metadata['user']]
                unique_users = sorted(list(set(users)))
                user_to_idx = {u: i for i, u in enumerate(unique_users)}
                user_indices = torch.tensor([user_to_idx[u] for u in users], device=device)
                
                # Mask: True if users are different (valid negative) or i==j (positive)
                mask_diff_users = (user_indices.unsqueeze(1) != user_indices.unsqueeze(0))
                mask_diff_users.fill_diagonal_(True)
                
                logits = logits.masked_fill(~mask_diff_users, float('-inf'))
                
            labels = torch.arange(B, device=device)
            cpc_loss = cpc_loss + F.cross_entropy(logits, labels)

            # Accuracy tracking
            correct += (logits.argmax(1) == labels).sum().item()
            total += B

        cpc_loss = cpc_loss / self.pred_steps

        # ── 5. Perplexity for logging ─────────────────────────────────────
        perplexity = self.vq.perplexity(indices)

        total_loss = cpc_loss + self.vq_loss_weight * commitment_loss

        # Stash metrics — training loop can pull these for logging
        batch.metadata.update({
            "cpc_loss": cpc_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "vq_perplexity": perplexity.item(),
            "cpc_f1": f1_score(labels.cpu().numpy(), logits.argmax(1).cpu().numpy(), average='macro')
        })

        return total_loss

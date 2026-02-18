import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import PretextTask

class StridedConvEncoder(nn.Module):
    """
    Encoder that maps raw CSI (B, 1, T, F) to Latents (B, D, T').
    Uses strided convolutions to downsample in time.
    """
    def __init__(self, in_features, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, out_dim, kernel_size=5, stride=2, padding=2)
        # T=500 -> 250 -> 125 -> 63
        
    def forward(self, x):
        # x: (B, 1, T, F) -> (B, F, T)
        # Note: CSIDataset usually returns (B, 1, T, F) or (B, T, F) depending on collate/transforms
        # Let's assume input is (B, 1, T, F). We need to verify dimensionality.
        if x.ndim == 4:
            x = x.squeeze(1) # (B, T, F)
        x = x.permute(0, 2, 1) # (B, F, T) for Conv1d
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x # (B, D, T')

class VectorQuantizer(nn.Module):
    """
    Standard VQ layer.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B, D, T) -> (B, T, D)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encoding_indices

class VQCPC(PretextTask):
    """
    Vector Quantized Contrastive Predictive Coding.
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        
        self.feature_size = getattr(config, 'feature_size', 232)
        self.hidden_dim = getattr(config, 'model_dim', 128)
        self.vq_embeddings = 512
        
        self.encoder = StridedConvEncoder(self.feature_size, self.hidden_dim, self.hidden_dim)
        self.vq = VectorQuantizer(self.vq_embeddings, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=2, batch_first=True)
        self.W_k = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(5)])

    def get_encoder(self):
        return self.encoder

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def validation_step(self, batch, batch_idx):
        return {}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if isinstance(batch, (list, tuple)):
            data = batch[0]
            metadata = batch[2] if len(batch) >= 3 else None
        elif isinstance(batch, dict):
            data = batch['data']
            metadata = batch.get('metadata', None)
        else:
            data = batch
            metadata = None
            
        # 1. Encode
        z = self.encoder(data) # (B, D, T')
        
        # 2. VQ
        vq_loss, z_q, perplexity, _ = self.vq(z)
        
        # 3. Autoregressor (Context C)
        # z_q: (B, D, T') -> (B, T', D) for GRU
        z_q_t = z_q.permute(0, 2, 1)
        c, _ = self.gru(z_q_t) # (B, T', D)
        
        # 4. CPC Loss (InfoNCE)
        # Predict z_{t+k} using c_t
        total_nce_loss = 0
        steps = len(self.W_k)
        seq_len = c.size(1)
        
        for k in range(1, steps + 1):
            if k >= seq_len: break
            
            c_curr = c[:, :-k, :] # (B, L, D)
            z_future = z_q_t[:, k:, :] # (B, L, D)
            
            pred = self.W_k[k-1](c_curr) # (B, L, D)
            
            # Simple dot product logits: (B, L, D) * (B, L, D) -> (B, L)
            # This implementation of InfoNCE is simplified (positive samples only logic usually requires negatives).
            # True CPC uses defaults to batch negatives.
            
            # Logits: (B, L, B) -> For each timestep in batch, score against all other samples?
            # Or simplified: score against random negatives?
            # Let's use Batch Negatives:
            # For each t, we have B samples. 1 positive, B-1 negatives.
            
            # Reshape for matmul
            # pred: (B*L, D)
            # z_future: (B*L, D)
            # logits = pred @ z_future.T (B*L, B*L) -> Too big
            
            # Let's do it per timestep to save memory or sub-sample
            # Or just use the standard implementation:
            
            # Simplified: NCE over batch at specific time steps
            # (B, D) @ (B, D).T -> (B, B)
            
            # We average over all valid timesteps
            # Actually, let's just pick a random timestep to optimize speed
            t = torch.randint(0, seq_len - k, (1,)).item()
            c_t = c[:, t, :] # (B, D)
            pred_t = self.W_k[k-1](c_t) # (B, D)
            z_next = z_q_t[:, t+k, :] # (B, D)
            
            # Logits: (B, B) where (i, j) is score of predicting z_j from c_i
            logits = torch.matmul(pred_t, z_next.T) # (B, D) @ (D, B) -> (B, B)
            
            # Cross-User Negative Sampling
            # If metadata is available and has 'user', we mask out negatives from the same user
            if metadata and isinstance(metadata, (list, tuple)) and len(metadata) > 0 and 'user' in metadata[0]:
                users = [str(m['user']) for m in metadata]
                batch_size = len(users)
                
                # Create mask: (B, B) where mask[i,j] is True if we should keep the pair (i,j)
                # We keep (i, j) if i == j (positive) OR user[i] != user[j] (valid negative)
                # We discard (i, j) if i != j AND user[i] == user[j] (same-user negative)
                
                # Using python set/dict for string comparison and torch for broadcasting
                # This removes numpy dependency
                unique_users = sorted(list(set(users)))
                user_to_idx = {u: i for i, u in enumerate(unique_users)}
                user_indices = torch.tensor([user_to_idx[u] for u in users], device=logits.device)
                
                # mask_diff_users: True where users distinct
                mask_diff_users = (user_indices.unsqueeze(1) != user_indices.unsqueeze(0))
                
                # Ensure diagonal is True (positives)
                mask_diff_users.fill_diagonal_(True)
                
                # Apply mask: Set invalid pairs to -inf so Softmax ignores them
                logits = logits.masked_fill(~mask_diff_users, float('-inf'))
            
            labels = torch.arange(logits.size(0), device=logits.device)
            total_nce_loss += F.cross_entropy(logits, labels)
            
        loss = total_nce_loss / steps + vq_loss
        
        return {
            'loss': loss,
            'nce_loss': total_nce_loss / steps,
            'vq_loss': vq_loss,
            'perplexity': perplexity
        }

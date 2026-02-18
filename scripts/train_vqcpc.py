#!/usr/bin/env python3
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from load.dataloader import get_loaders
from pretext_tasks.vqcpc import VQCPC

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def cross_user_negative_mask(metadata, device):
    """
    Creates a mask (B, B) where mask[i, j] is True if sample j is a valid negative for sample i.
    Valid negative: user[i] != user[j].
    Also includes diagonal as True (positive).
    """
    if not metadata or not isinstance(metadata, (list, tuple)) or 'user' not in metadata[0]:
        return None
        
    users = [str(m['user']) for m in metadata]
    
    # Map users to integers for efficient comparison in Torch
    unique_users = sorted(list(set(users)))
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    user_indices = torch.tensor([user_to_idx[u] for u in users], device=device)
    
    # True where users are different
    # (B, 1) != (1, B) -> (B, B)
    mask_diff_users = (user_indices.unsqueeze(1) != user_indices.unsqueeze(0))
    
    mask_diff_users.fill_diagonal_(True) # Always keep self as positive
    
    return mask_diff_users

def main():
    parser = argparse.ArgumentParser(description="Bespoke VQ-CPC Training Script")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../data/csi-bench-dataset/csi-bench-dataset", help="Path to dataset")
    parser.add_argument("--task", type=str, default="MotionSourceRecognition", help="Dataset task name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_key", type=str, default="CSI_amps")
    parser.add_argument("--file_format", type=str, default="h5")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pin memory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results_vqcpc")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    
    # VQ-CPC specific args
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--feature_size", type=int, default=232)
    parser.add_argument("--win_len", type=int, default=500)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Save dirs
    save_path = os.path.join(args.save_dir, args.task)
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))
    
    # Data
    print(f"Loading data for task {args.task}...")
    datasets = get_loaders(
        root=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_key=args.data_key,
        file_format=args.file_format,
        pin_memory=(device != "cpu" and not args.no_pin_memory),
    )
    
    if isinstance(datasets, dict) and "loaders" in datasets:
        loaders = datasets["loaders"]
    else:
        loaders = datasets

    train_loader = loaders["train"]
    
    # Config wrapper for VQCPC compatibility
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    config = Config(**vars(args))
    
    # Model
    print("Initializing VQ-CPC...")
    model = VQCPC(config, device)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (data, label, metadata)
            if isinstance(batch, (list, tuple)):
                data = batch[0]
                metadata = batch[2] if len(batch) >= 3 else None
            elif isinstance(batch, dict):
                data = batch['data']
                metadata = batch.get('metadata', None)
            else:
                data = batch
                metadata = None
                
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # --- Bespoke Training Step Logic ---
            
            # 1. Encode
            z = model.encoder(data) # (B, D, T')
            
            # 2. VQ
            vq_loss, z_q, perplexity, _ = model.vq(z)
            
            # 3. Autoregressor
            z_q_t = z_q.permute(0, 2, 1)
            c, _ = model.gru(z_q_t) # (B, T', D)
            
            # 4. CPC Loss (InfoNCE) with Cross-User Negatives
            total_nce_loss = 0
            steps = len(model.W_k)
            seq_len = c.size(1)
            
            valid_steps = 0
            for k in range(1, steps + 1):
                if k >= seq_len: break
                valid_steps += 1
                
                # Pick a random timestep t
                t = torch.randint(0, seq_len - k, (1,)).item()
                c_t = c[:, t, :] # (B, D)
                pred_t = model.W_k[k-1](c_t) # (B, D)
                z_next = z_q_t[:, t+k, :] # (B, D)
                
                # Logits: (B, B)
                logits = torch.matmul(pred_t, z_next.T)
                
                # Apply Cross-User Mask
                mask = cross_user_negative_mask(metadata, device)
                if mask is not None:
                    logits = logits.masked_fill(~mask, float('-inf'))
                
                labels = torch.arange(logits.size(0), device=device)
                total_nce_loss += F.cross_entropy(logits, labels)
            
            if valid_steps > 0:
                nce_loss = total_nce_loss / valid_steps
            else:
                nce_loss = torch.tensor(0.0, device=device)
                
            loss = nce_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            # Logging
            step_metrics = {
                'loss': loss.item(),
                'nce_loss': nce_loss.item(),
                'vq_loss': vq_loss.item(),
                'perplexity': perplexity.item()
            }
            
            if batch_idx % args.log_interval == 0:
                for k, v in step_metrics.items():
                    writer.add_scalar(k, v, epoch * len(train_loader) + batch_idx)
            
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
            
        # Save Checkpoint
        save_file = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_file)
        
        # Save Encoder
        encoder_path = os.path.join(save_path, "encoder_latest.pt")
        try:
            torch.save(model.encoder.state_dict(), encoder_path)
        except Exception as e:
            print(f"Error saving encoder: {e}")
            
    print(f"Training complete. Results saved to {save_path}")
    writer.close()

if __name__ == "__main__":
    main()

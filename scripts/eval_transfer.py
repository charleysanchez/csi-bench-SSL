"""
Transfer Learning Evaluation Script
===================================

This script evaluates the generalizability of pretrained models on unseen domains 
(e.g., new users, devices, or environments). It supports zero-shot evaluation 
and fine-tuning on a small subset of the target domain.

Usage:
    python scripts/eval_transfer.py \
        --model_path results_vqcpc/best.pt \
        --data_dir ../data/csi-bench-dataset/csi-bench-dataset \
        --task MotionSourceRecognition \
        --target_filter "{'user': ['5']}" \
        --finetune_split 0.1 \
        --epochs 20
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load.dataset import CSIDataset
import model.encoders as model_encoders
import model.models as model_models
from model.encoders import CSIEncoder, CSIConvEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer Learning Evaluation")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Root dataset directory")
    parser.add_argument("--task", type=str, default="MotionSourceRecognition", help="Task name")
    parser.add_argument("--target_filter", type=str, required=True, 
                        help="Filter dict for target domain (e.g. \"{'user': ['5']}\")")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained .pt checkpoint")
    parser.add_argument("--architecture", type=str, default=None, 
                        help="Model architecture class name (e.g. 'CSIEncoder', 'ResNet18Classifier'). "
                             "If None, uses 'encoder' arg to choose default MLP/Conv encoder.")
    parser.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "conv"], help="Encoder type (deprecated if --architecture used)")
    
    parser.add_argument("--model_dim", type=int, default=128, help="Model/Encoder output dim")
    parser.add_argument("--feature_size", type=int, default=232, help="Input feature size")
    parser.add_argument("--win_len", type=int, default=500, help="Window length")
    
    # Fine-tuning
    parser.add_argument("--finetune_split", type=float, default=0.0, 
                        help="Fraction of target data used for fine-tuning (0.0 = zero-shot)")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Fine-tuning learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder/backbone weights")
    
    return parser.parse_args()

class FrameAverageClassifier(nn.Module):
    """
    Wraps a frame-level encoder to create a sequence classifier.
    
    Logic:
    1. Input: (B, T, F) sequence.
    2. Flatten -> (B*T, F).
    3. Encode -> (B*T, D) using the frame-level encoder.
    4. Reshape -> (B, T, D).
    5. Mean Pool -> (B, D) aggregating over time.
    6. Classify -> (B, NumClasses).
    
    Use this for encoders that don't handle time internally (e.g. CSIEncoder).
    Do NOT use this for sequence models (e.g. LSTM, ResNet) that take (B, T, F) directly.
    """
    def __init__(self, encoder, model_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(model_dim, num_classes)
        
    def forward(self, x):
        # x: (B, 1, T, F) or (B, T, F)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # Now x is (B, T, F)
        B, T, F = x.shape
        
        # Flatten time to encode processing each frame
        x_flat = x.view(B * T, F)
        
        # Encode
        features_flat = self.encoder(x_flat) # (B*T, D)
        
        # Reshape back
        features = features_flat.view(B, T, -1) # (B, T, D)
        
        # Mean Pooling over time
        pooled = features.mean(dim=1) # (B, D)
        
        return self.classifier(pooled)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(args, num_classes, device):
    """
    Instantiate the model based on arguments.
    Returns: (model, is_wrapper)
    """
    # 1. Check if specific architecture requested
    if args.architecture:
        # Try finding in encoders (Frame-Level)
        if hasattr(model_encoders, args.architecture):
            print(f"Loading frame-level encoder: {args.architecture} (wrapping in FrameAverageClassifier)")
            cls = getattr(model_encoders, args.architecture)
            # Assuming standard encoder init: feature_size, model_dim
            try:
                encoder = cls(feature_size=args.feature_size, model_dim=args.model_dim)
            except TypeError:
                # Fallback for Conv encoder which might accept channels etc
                encoder = cls(feature_size=args.feature_size, model_dim=args.model_dim)
            
            # Wrap in FrameAverageClassifier to handle time
            model = FrameAverageClassifier(encoder, args.model_dim, num_classes)
            return model.to(device), True
            
        # Try finding in models (Sequence-Level Classifiers)
        elif hasattr(model_models, args.architecture):
            print(f"Loading sequence classifier: {args.architecture}")
            cls = getattr(model_models, args.architecture)
            # models usually take: win_len, feature_size, num_classes
            try:
                model = cls(win_len=args.win_len, feature_size=args.feature_size, num_classes=num_classes)
            except Exception as e:
                print(f"Error instantiating {args.architecture}: {e}")
                print("Trying alternative init layout...")
                model = cls(feature_size=args.feature_size, num_classes=num_classes)
            
            return model.to(device), False
        else:
            raise ValueError(f"Architecture {args.architecture} not found in model.encoders or model.models")
            
    # 2. explicit legacy choices (Default to Frame Encoders)
    else:
        if args.encoder == "mlp":
            encoder = CSIEncoder(feature_size=args.feature_size, model_dim=args.model_dim)
        else:
            encoder = CSIConvEncoder(feature_size=args.feature_size, model_dim=args.model_dim)
        
        model = FrameAverageClassifier(encoder, args.model_dim, num_classes)
        return model.to(device), True

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        target_filter = json.loads(args.target_filter)
        print(f"Target Domain Filter: {target_filter}")
    except Exception as e:
        raise ValueError(f"Invalid target_filter format. Use dictionary string. Error: {e}")

    try:
        dataset = CSIDataset(
            root=args.data_dir,
            task=args.task,
            split="all_available",
            filter_dict=target_filter,
            debug=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("Error: No samples found for the specified target filter!")
        sys.exit(1)
        
    num_classes = dataset.num_classes
    print(f"Dataset Size: {len(dataset)}, Num Classes: {num_classes}")
    
    # Split into Fine-tune / Test
    total_size = len(dataset)
    if args.finetune_split > 0:
        train_size = int(total_size * args.finetune_split)
        test_size = total_size - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])
        
        print(f"Splitting data: {train_size} training (fine-tuning), {test_size} testing")
    else:
        print("Zero-shot mode: Using all data for testing.")
        train_set = None
        test_set = dataset

    # Loaders
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_loader = None
    if train_set:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Initialize Model
    model, is_wrapper = get_model(args, num_classes, device)
        
    # Load Pretrained Weights
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        state_dict = None
        # Check if loading full checkpoint or just encoder state dict
        if isinstance(checkpoint, dict):
            if 'encoder_state' in checkpoint:
                 state_dict = checkpoint['encoder_state']
            elif 'encoder_state_dict' in checkpoint:
                state_dict = checkpoint['encoder_state_dict']
            elif 'model_state_dict' in checkpoint:
                # Try to filter out 'encoder.' prefix if existing
                full_sd = checkpoint['model_state_dict']
                state_dict = {}
                for k, v in full_sd.items():
                    if k.startswith('encoder.'):
                        state_dict[k.replace('encoder.', '')] = v
                    elif not k.startswith('task.'): 
                        state_dict[k] = v
            elif 'encoder' in checkpoint:
                 state_dict = checkpoint['encoder']
        else:
             # checkpoint might be the state dict itself
             state_dict = checkpoint
             
        if state_dict:
            if is_wrapper:
                # Load into model.encoder
                msg = model.encoder.load_state_dict(state_dict, strict=False)
            else:
                # Load into full model
                msg = model.load_state_dict(state_dict, strict=False)
            print(f"Model load status: {msg}")
        else:
            print("Warning: Could not identify state dict in checkpoint. Using random init.")
    else:
        print(f"Warning: Model path {args.model_path} does not exist. Initializing random weights.")

    
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        if is_wrapper:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            # Try to freeze everything except the last layer
            # Heuristic: Find last linear layer
            for name, param in model.named_parameters():
                param.requires_grad = False
            
            # Unfreeze common head names
            found_head = False
            for head_name in ['fc', 'classifier', 'head']:
                if hasattr(model, head_name):
                    print(f"Unfreezing head: {head_name}")
                    for param in getattr(model, head_name).parameters():
                        param.requires_grad = True
                    found_head = True
            
            if not found_head:
                print("Warning: Could not find classification head to unfreeze! Freezing ALL weights.")
            
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Fine-tuning Loop
    if train_loader and args.epochs > 0:
        print(f"\nStarting Fine-tuning for {args.epochs} epochs...")
        model.train()
        
        for epoch in range(args.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, dynamic_ncols=True)
            for batch in pbar:
                if batch is None: continue
                # CSIDataset returns: (data, label, metadata)
                x, y, _ = batch
                
                x = x.to(device)
                y = y.to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            epoch_acc = 100 * correct / total if total > 0 else 0
            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Acc = {epoch_acc:.2f}%")
            
    # Evaluation Loop
    print("\nStarting Evaluation...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True):
            if batch is None: continue
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    acc = 100 * correct / total if total > 0 else 0.0
    print(f"\nTarget Domain Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()

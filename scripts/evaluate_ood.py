#!/usr/bin/env python3
"""
Standalone OOD Evaluation Script

Usage:
    python scripts/evaluate_ood.py \
        --weights results/HumanActivityRecognition/transformer/params_XYZ/best_model.pt \
        --config configs/supervised_transformer.yaml \
        --pipeline supervised \
        --tasks HumanActivityRecognition
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score

# Ensure we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load.dataloader import get_loaders
from scripts.train_supervised import MODEL_TYPES
from model.multitask.models import MultiTaskAdapterModel, PatchTSTAdapterModel, TimesFormerAdapterModel
from utils.config import update_args_with_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on OOD test splits.")
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained weights (.pt or .pth)')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file used during training')
    parser.add_argument('--pipeline', type=str, required=True, choices=['supervised', 'multitask'], help='Which pipeline was used to train this model')
    parser.add_argument('--tasks', type=str, required=True, help='Comma-separated list of tasks (e.g., HumanActivityRecognition)')
    parser.add_argument('--model', type=str, default='transformer', help='Backbone model architecture')
    parser.add_argument('--data_dir', type=str, default='data', help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    
    args, unknown = parser.parse_known_args()
    
    # Load model parameters from YAML so we build the exact same architecture
    if args.config is not None:
        args = update_args_with_yaml(args, args.config)
        
    return args

def evaluate(model, loader, device, task=None):
    """Run pure inference and calculate metrics."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # If it's a multitask model, ensure the correct head is active
    if hasattr(model, 'set_active_task') and task is not None:
        model.set_active_task(task)
        
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = correct / total if total > 0 else 0
    
    if len(all_preds) > 0 and len(set(all_labels)) > 1:
        try:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        except Exception:
            f1 = 0.0
    else:
        f1 = 0.0
        
    return accuracy, f1

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f" STARTING OOD EVALUATION ON DEVICE: {device}")
    print(f" WEIGHTS: {args.weights}")
    print(f"{'='*80}\n")

    tasks = args.tasks.split(',')
    test_loaders = {}
    task_classes = {}

    # 1. Custom Collate to handle None samples
    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return torch.zeros(0, 1, args.win_len, getattr(args, 'feature_size', 232)), torch.zeros(0, dtype=torch.long)
        return torch.utils.data.dataloader.default_collate(batch)

    # 2. Load Data
    for task in tasks:
        print(f"Loading test splits for: {task}...")
        data = get_loaders(
            root=args.data_dir,
            task=task,
            batch_size=args.batch_size,
            file_format=getattr(args, 'file_format', 'h5'),
            data_key=getattr(args, 'data_key', 'CSI_amps'),
            num_workers=args.num_workers,
            test_splits='all', # Force load all OOD splits
            collate_fn=custom_collate_fn
        )
        ld = data['loaders']
        test_loaders[task] = {k: v for k, v in ld.items() if k.startswith('test')}
        task_classes[task] = data['num_classes']

    # 3. Build Model Architecture
    print(f"\nBuilding {args.pipeline.upper()} {args.model.upper()} architecture...")
    ModelClass = MODEL_TYPES[args.model]
    
    # Grab feature size from data if missing
    feature_size = getattr(args, 'feature_size', None)
    if feature_size is None:
        sample_x, _ = next(iter(test_loaders[tasks[0]][list(test_loaders[tasks[0]].keys())[0]]))
        feature_size = sample_x.shape[-1]

    # Initialize Base kwargs
    model_kwargs = {'num_classes': task_classes[tasks[0]]}
    if args.model in ['mlp', 'vit', 'patchtst', 'timesformer1d', 'transformer']:
        model_kwargs.update({'win_len': getattr(args, 'win_len', 500), 'feature_size': feature_size})
    if args.model == 'transformer':
        model_kwargs.update({'d_model': getattr(args, 'emb_dim', 256), 'dropout': getattr(args, 'dropout', 0.1)})
        
    backbone_model = ModelClass(**model_kwargs)

    # Wrap for Multitask if necessary
    if args.pipeline == "multitask":
        if args.model == 'transformer':
            class TransformerEmbedding(nn.Module):
                def __init__(self, cls_model):
                    super().__init__()
                    self.input_proj = cls_model.input_proj
                    self.pos_encoder = cls_model.pos_encoder
                    self.transformer = cls_model.transformer
                def forward(self, x):
                    x = x.squeeze(1)
                    x = self.input_proj(x)
                    x = self.pos_encoder(x)
                    x = self.transformer(x)
                    return x.mean(dim=1)
            
            backbone = TransformerEmbedding(backbone_model)
            class ConfigDict(dict):
                def __getattr__(self, name): return self.get(name)
            
            backbone.config = ConfigDict(hidden_size=getattr(args, 'emb_dim', 256), num_hidden_layers=len(backbone_model.transformer.layers))
            model = MultiTaskAdapterModel(backbone, task_classes, lora_r=getattr(args, 'lora_r', 8), lora_alpha=getattr(args, 'lora_alpha', 32), lora_dropout=getattr(args, 'lora_dropout', 0.05))
        else:
            raise NotImplementedError("Eval script currently only configured for multitask transformer wrapping.")
    else:
        model = backbone_model

    model.to(device)

    # 4. Load Weights
    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device)
    
    # Handle different save formats
    if 'adapters' in state_dict and args.pipeline == 'multitask':
        # Multitask save format
        model.adapters.load_state_dict(state_dict['adapters'])
        model.heads.load_state_dict(state_dict['heads'])
    elif 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        # Standard state dict
        model.load_state_dict(state_dict)

    print("Weights loaded successfully!\n")

    # 5. Evaluate
    for task in tasks:
        print(f"{'-'*50}\n Results for: {task}\n{'-'*50}")
        tdict = test_loaders[task]
        
        print(f"{'Split':<20} {'Accuracy':<15} {'F1 Score':<15}")
        print('-' * 50)
        
        for split, tloader in tdict.items():
            acc, f1 = evaluate(model, tloader, device, task=task if args.pipeline == 'multitask' else None)
            print(f"{split:<20} {acc*100:>6.2f}%         {f1*100:>6.2f}%")
            
    print("\nEvaluation Complete!")

if __name__ == "__main__":
    main()
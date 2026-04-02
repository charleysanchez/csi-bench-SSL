#!/usr/bin/env python3
"""
Standalone OOD Evaluation Script with Test-Time Augmentation (TTA)

Usage:
    # Standard evaluation:
    python scripts/evaluate_ood.py \
        --weights results/HumanActivityRecognition/transformer/params_XYZ/best_model.pt \
        --config configs/supervised_ood.yaml \
        --pipeline supervised \
        --tasks HumanActivityRecognition

    # With TTA (recommended for OOD):
    python scripts/evaluate_ood.py \
        --weights results/.../best_model.pt \
        --config configs/supervised_ood.yaml \
        --pipeline supervised \
        --tasks HumanActivityRecognition \
        --tta --tta_rounds 10
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
from load.collate import CollateSkipNone
from model.registry import MODEL_TYPES
from model.wrappers import wrap_backbone_for_multitask_transformer, wrap_for_dann
from model.multitask.models import SimpleMultiTaskModel
from utils.config import update_args_with_yaml
from load.data_augmentation import CSIAugmentation

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

    # TTA options
    parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation')
    parser.add_argument('--tta_rounds', type=int, default=10, help='Number of TTA augmentation rounds')

    # TENT options (Test-Time Entropy minimization)
    parser.add_argument('--tent', action='store_true',
                        help='Enable TENT: adapt BatchNorm params at test time via entropy minimization '
                             '(Wang et al., "Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021)')
    parser.add_argument('--tent_steps', type=int, default=1,
                        help='Number of TENT adaptation steps per batch (default 1)')
    parser.add_argument('--tent_lr', type=float, default=1e-3,
                        help='Learning rate for TENT adaptation')

    # DANN options
    parser.add_argument('--use_dann', action='store_true', help='Enable DANN evaluation (wraps encoder)')
    
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
            # Handle variable-sized batches from dataset
            inputs, labels = batch[0], batch[1]
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


def evaluate_with_tta(model, loader, device, augmentor, n_rounds=10, task=None):
    """
    Run Test-Time Augmentation: for each batch, create n_rounds augmented versions,
    average the softmax predictions, then pick the argmax.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    if hasattr(model, 'set_active_task') and task is not None:
        model.set_active_task(task)
    
    with torch.no_grad():
        for batch in loader:
            # Handle variable-sized batches from dataset
            inputs, labels = batch[0], batch[1]
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            
            # Start with the clean (un-augmented) prediction
            clean_outputs = model(inputs)
            avg_probs = torch.softmax(clean_outputs, dim=1)
            
            # Add augmented predictions
            for _ in range(n_rounds):
                aug_inputs = augmentor(inputs.clone())
                aug_outputs = model(aug_inputs)
                avg_probs += torch.softmax(aug_outputs, dim=1)
            
            # Average over (1 clean + n_rounds augmented)
            avg_probs /= (n_rounds + 1)
            
            _, predicted = torch.max(avg_probs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    
    if len(set(all_labels)) > 1:
        try:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        except Exception:
            f1 = 0.0
    else:
        f1 = 0.0
    
    return accuracy, f1


def setup_tent(model, lr=1e-3):
    """
    TENT: Fully Test-Time Adaptation by Entropy Minimization.
    Wang et al., ICLR 2021.

    Configures a model for test-time adaptation by:
    1. Setting all BatchNorm layers to train mode (so they update running stats).
    2. Only optimizing the BatchNorm affine parameters (gamma, beta).
    3. Freezing everything else.

    Returns (model, optimizer) ready for TENT adaptation.
    """
    model.eval()  # start from eval

    # Collect BatchNorm affine parameters only
    tent_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.requires_grad_(True)
            # Set BN layers to train mode so they update running stats
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()
            for p in m.parameters():
                tent_params.append(p)
        else:
            # Freeze all other parameters
            for p in m.parameters(recurse=False):
                p.requires_grad = False

    optimizer = torch.optim.Adam(tent_params, lr=lr) if tent_params else None
    return model, optimizer


def evaluate_with_tent(model, loader, device, tent_optimizer, tent_steps=1, task=None):
    """
    Run TENT adaptation: for each batch, minimize prediction entropy
    w.r.t. BatchNorm affine parameters, then predict.

    This adapts the model online — each batch updates the model, and
    subsequent batches benefit from accumulated adaptation.
    """
    all_preds = []
    all_labels = []

    if hasattr(model, 'set_active_task') and task is not None:
        model.set_active_task(task)

    for batch in loader:
        inputs, labels = batch[0], batch[1]
        inputs, labels = inputs.to(device), labels.to(device)

        # TENT: adapt on this batch by minimizing entropy
        if tent_optimizer is not None:
            for _ in range(tent_steps):
                outputs = model(inputs)
                # Entropy of softmax predictions
                probs = torch.softmax(outputs, dim=1)
                entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1).mean()
                tent_optimizer.zero_grad()
                entropy.backward()
                tent_optimizer.step()

        # Predict (no grad)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    if len(set(all_labels)) > 1:
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
    if args.tta:
        print(f" TTA ENABLED: {args.tta_rounds} rounds")
    print(f"{'='*80}\n")

    tasks = args.tasks.split(',')
    test_loaders = {}
    task_classes = {}

    # 1. Custom collate to handle None samples
    feature_size = getattr(args, 'feature_size', 232)
    custom_collate_fn = CollateSkipNone(args.win_len, feature_size, for_supervised=True)

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
        sample_batch = next(iter(test_loaders[tasks[0]][list(test_loaders[tasks[0]].keys())[0]]))
        sample_x = sample_batch[0]
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
            backbone = wrap_backbone_for_multitask_transformer(backbone_model, args)
            model = SimpleMultiTaskModel(backbone, task_classes)
        else:
            raise NotImplementedError("Eval script currently only configured for multitask transformer wrapping.")
    else:
        model = backbone_model

    # Wrap for DANN if necessary
    if getattr(args, 'use_dann', False):
        print("Wrapping model in DANNTransformer for evaluation...")
        model = wrap_for_dann(model, task_classes[tasks[0]], args)

    model.to(device)

    # 4. Load Weights
    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device, weights_only=False)
    
    # Handle different save formats
    if 'backbone' in state_dict and 'heads' in state_dict and args.pipeline == 'multitask':
        # New multitask save format (backbone + heads)
        model.backbone.load_state_dict(state_dict['backbone'])
        for task_name, head_state in state_dict['heads'].items():
            model.heads[task_name].load_state_dict(head_state)
    elif 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    elif 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        # Standard state dict
        model.load_state_dict(state_dict)

    print("Weights loaded successfully!\n")

    # 5. Create TTA augmentor if needed (use mild augmentation for TTA)
    if args.tta:
        tta_augmentor = CSIAugmentation(
            amp_scale_range=(0.8, 1.2),     # Mild amplitude scaling
            noise_std=0.05,                  # Light noise
            time_shift_max=10,               # Small time shifts
            subcarrier_drop_max=5,           # Few subcarriers dropped
            subcarrier_drop_prob=0.3,
            time_shift_prob=0.3,
            noise_prob=0.5,
            freq_jitter_prob=0.2,
            freq_jitter_std=0.02,
        )

    # 6. Evaluate
    for task in tasks:
        print(f"{'-'*50}\n Results for: {task}\n{'-'*50}")
        tdict = test_loaders[task]

        if args.tent:
            # TENT evaluation: compare standard vs TENT-adapted
            import copy
            print(f"{'Split':<20} {'Standard':<25} {'TENT (steps=' + str(args.tent_steps) + ')':<25}")
            print('-' * 70)

            for split, tloader in tdict.items():
                task_arg = task if args.pipeline == 'multitask' else None

                # Standard evaluation
                acc_std, f1_std = evaluate(model, tloader, device, task=task_arg)

                # TENT: adapt a fresh copy per split (so splits don't leak into each other)
                tent_model = copy.deepcopy(model)
                tent_model, tent_opt = setup_tent(tent_model, lr=args.tent_lr)
                acc_tent, f1_tent = evaluate_with_tent(
                    tent_model, tloader, device, tent_opt,
                    tent_steps=args.tent_steps, task=task_arg
                )
                del tent_model  # free memory

                delta = acc_tent - acc_std
                delta_sign = "+" if delta >= 0 else ""
                print(f"{split:<20} {acc_std*100:>6.2f}% / {f1_std*100:>5.2f}%    "
                      f"{acc_tent*100:>6.2f}% / {f1_tent*100:>5.2f}%  ({delta_sign}{delta*100:.1f}%)")

        elif args.tta:
            print(f"{'Split':<20} {'Standard':<25} {'TTA (' + str(args.tta_rounds) + ' rounds)':<25}")
            print('-' * 70)

            for split, tloader in tdict.items():
                task_arg = task if args.pipeline == 'multitask' else None

                acc_std, f1_std = evaluate(model, tloader, device, task=task_arg)
                acc_tta, f1_tta = evaluate_with_tta(model, tloader, device, tta_augmentor,
                                                     n_rounds=args.tta_rounds, task=task_arg)
                delta = acc_tta - acc_std
                delta_sign = "+" if delta >= 0 else ""
                print(f"{split:<20} {acc_std*100:>6.2f}% / {f1_std*100:>5.2f}%    "
                      f"{acc_tta*100:>6.2f}% / {f1_tta*100:>5.2f}%  ({delta_sign}{delta*100:.1f}%)")
        else:
            print(f"{'Split':<20} {'Accuracy':<15} {'F1 Score':<15}")
            print('-' * 50)

            for split, tloader in tdict.items():
                task_arg = task if args.pipeline == 'multitask' else None
                acc, f1 = evaluate(model, tloader, device, task=task_arg)
                print(f"{split:<20} {acc*100:>6.2f}%         {f1*100:>6.2f}%")

    print("\nEvaluation Complete!")

if __name__ == "__main__":
    main()
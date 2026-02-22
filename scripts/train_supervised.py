#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # Ensure pandas is imported
from load.dataloader import get_loaders
from tqdm import tqdm
import json
import hashlib
import time
import matplotlib.pyplot as plt
import warnings
import yaml
from utils.config import update_args_with_yaml, save_config


# Mute sklearn warnings about precision/recall being ill-defined
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Import model classes from models.py
from model.models import (
    MLPClassifier, 
    LSTMClassifier, 
    ResNet18Classifier, 
    TransformerClassifier, 
    ViTClassifier,
    PatchTST,
    TimesFormer1D
)

# Import TaskTrainer
from engine.task_trainer import TaskTrainer

MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier,
    'patchtst': PatchTST,
    'timesformer1d': TimesFormer1D
}

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train a supervised model on WiFi benchmark dataset')
        parser.add_argument('--data_dir', type=str, default='data',
                            help='Root directory of the dataset')
        parser.add_argument('--task', type=str, default='MotionSourceRecognition',
                            help='Name of the task to train on')
        parser.add_argument('--model', type=str, default='vit', 
                            choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit', 'patchtst', 'timesformer1d'],
                            help='Type of model to train')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--data_key', type=str, default='CSI_amps',
                            help='Key for CSI data in h5 files')
        parser.add_argument('--save_dir', type=str, default='results',
                            help='Directory to save checkpoints')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save results (defaults to save_dir if not specified)')
        parser.add_argument('--weight_decay', type=float, default=1e-5, 
                            help='Weight decay for optimizer')
        parser.add_argument('--warmup_epochs', type=int, default=5,
                            help='Number of warmup epochs')
        parser.add_argument('--patience', type=int, default=15,
                            help='Patience for early stopping')
        # Additional model parameters
        parser.add_argument('--win_len', type=int, default=500, 
                            help='Window length for WiFi CSI data')
        parser.add_argument('--feature_size', type=int, default=232,
                            help='Feature size for WiFi CSI data')
        parser.add_argument('--in_channels', type=int, default=1, 
                            help='Number of input channels for convolutional models')
        parser.add_argument('--emb_dim', type=int, default=128, 
                            help='Embedding dimension for transformer models')
        parser.add_argument('--d_model', type=int, default=256, 
                            help='Model dimension for Transformer model')
        parser.add_argument('--dropout', type=float, default=0.1, 
                            help='Dropout rate for regularization')
        # PatchTST specific parameters
        parser.add_argument('--patch_len', type=int, default=16,
                            help='Patch length for PatchTST model')
        parser.add_argument('--stride', type=int, default=8,
                            help='Stride for patches in PatchTST model')
        parser.add_argument('--pool', type=str, default='cls', choices=['cls', 'mean'],
                            help='Pooling method for PatchTST (cls or mean)')
        parser.add_argument('--head_dropout', type=float, default=0.2,
                            help='Dropout rate for classification head')
        # TimesFormer-1D specific parameters
        parser.add_argument('--patch_size', type=int, default=4,
                            help='Patch size for TimesFormer-1D model')
        parser.add_argument('--attn_dropout', type=float, default=0.1,
                            help='Dropout rate for attention layers')
        parser.add_argument('--mlp_ratio', type=float, default=4.0,
                            help='MLP ratio for transformer blocks')
        parser.add_argument('--depth', type=int, default=6,
                            help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads')
        # Testing specific parameters
        parser.add_argument('--test_splits', type=str, default='all',
                            help='Comma-separated list of test splits to use. If "all", all available test splits will be used. '
                                 'Examples: test_id,test_cross_env,test_cross_user,test_cross_device,hard_cases')
        
        # Few-shot learning parameters
        parser.add_argument('--enable_few_shot', action='store_true',
                            help='Enable few-shot learning adaptation for cross-domain scenarios')
        parser.add_argument('--k_shot', type=int, default=5,
                            help='Number of examples per class for few-shot adaptation')
        parser.add_argument('--inner_lr', type=float, default=0.01,
                            help='Learning rate for few-shot adaptation')
        parser.add_argument('--num_inner_steps', type=int, default=10,
                            help='Number of gradient steps for few-shot adaptation')
        
        # Add data loading and optimization options
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of worker processes for data loading')
        parser.add_argument('--use_root_data_path', action='store_true',
                            help='Use root directory as task directory')
        parser.add_argument('--file_format', type=str, default='h5',
                            help='File format for data files (h5, mat, or npy)')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
        parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode')
        parser.add_argument('--pin_memory', action='store_true', default=True,
                            help='Enable pin memory for data loading (not recommended for MPS)')
        parser.add_argument('--no_pin_memory', action='store_true',
                            help='Disable pin memory for data loading (use for MPS devices)')
        parser.add_argument('--pretrained_encoder', type=str, default=None,
                    help='Path to pretrained encoder weights (.pt file)')
        parser.add_argument('--config', type=str, default=None,
                    help='Path to YAML config file to override args')
        
        args = parser.parse_args()

    # Override args with YAML config if provided
    if args.config is not None:
        args = update_args_with_yaml(args, args.config)
    
    # set output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.save_dir

    # ensure directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # generate a unique experiment ID based only on parameters hash
    # this way, same parameters will generate same experiment ID
    param_str = f"{args.learning_rate}_{args.batch_size}_{args.epochs}_{args.weight_decay}_{args.warmup_epochs}_{args.win_len}_{args.feature_size}"
    if hasattr(args, 'dropout') and args.dropout is not None:
        param_str += f"_{args.dropout}"
    if hasattr(args, 'emb_dim') and args.emb_dim is not None:
        param_str += f"_{args.emb_dim}"
    if hasattr(args, 'd_model') and args.d_model is not None:
        param_str += f"_{args.d_model}"
    if hasattr(args, 'in_channels') and args.in_channels is not None:
        param_str += f"_{args.in_channels}"
    
    experiment_id = f"params_{hashlib.md5(param_str.encode()).hexdigest()[:10]}"

    # use specified directories for local environment with task/model/experiment structure
    # directory structure: save_dir/task/model/experiment_id
    results_dir = os.path.join(args.output_dir, args.task, args.model, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    # create checkpoints directory too
    checkpoint_dir = os.path.join(args.save_dir, args.task, args.model, experiment_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # create best_performance.json file at model level if it doesn't exist
    model_level_dir = os.path.join(args.output_dir, args.task, args.model)
    best_performance_path = os.path.join(model_level_dir, "best_performance.json")
    if not os.path.exists(best_performance_path):
        with open(best_performance_path, "w") as f:
            json.dump({
                "best_test_accuracy": 0.0,
                "best_test_f1_score": 0.0,
                "best_experiment_id": None,
                "best_experiment_params": {},
            }, f, indent=4)
    
    print(f"Experiment ID: {experiment_id}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Model checkpoints will be saved to: {checkpoint_dir}")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # check for available test splits in the dataset
    task_dir = os.path.join(args.data_dir, args.task)
    splits_dir = os.path.join(task_dir, "splits")
    available_test_splits = []
    if os.path.exists(splits_dir):
        for filename in os.listdir(splits_dir):
            if filename.endswith(".json") and filename.startswith("test_"):
                fn, _ = os.path.splitext(filename)
                available_test_splits.append(fn)

    print(f"Available test splits: {available_test_splits}")

    # determine which test splits to use
    if args.test_splits == "all":
        test_splits = available_test_splits
    elif args.test_splits is not None:
        test_splits = args.test_splits.split(",")
        # validate that the requested splits exist
        for split in test_splits:
            if split not in available_test_splits and split != "test_id":
                print(f"Warning: Requested test split '{split}' not found. Available splits: {available_test_splits}")
    else:
        # default to test_id (in-distribution test)
        test_splits = ["test_id"]

    print(f"Using test splits: {test_splits}")

    # add handling for None values to prevent dataloader errors
    def custom_collate_fn(batch):
        # filter out None samples
        batch = [item for item in batch if item is not None]
        
        # if no samples remain after filtering, return empty tensors
        if len(batch) == 0:
            return torch.zeros(0, 1, args.win_len, args.feature_size), torch.zeros(0, dtype=torch.long)
        
        # use default collate function for the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)

    if args.no_pin_memory:
        args.pin_memory = False

    # load data
    print(f"Loading data from {args.data_dir} for task {args.task}...")
    data = get_loaders(
        root=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        file_format=args.file_format,
        data_key=args.data_key,
        num_workers=args.num_workers,
        test_splits=test_splits,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate_fn,
        debug=False,
    )

    # extract data from the returned dictionary
    loaders = data["loaders"]
    num_classes = data["num_classes"]
    label_mapper = data["label_mapper"]

    # get training and validation loaders
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    if val_loader is None:
        print("Warning: no validation data found. Using training data for validation.")
        val_loader = train_loader

    # countunique labels in the dataset
    all_labels = []
    dataset = train_loader.dataset
    print(f"Detected {num_classes} classes in the dataset.")

    # get test loaders
    test_loaders = {k: v for k, v in loaders.items() if k.startswith("test")}
    if not test_loaders:
        print("Warning: No test splits found in the dataset. Check split names and dataset structure.")
    else:
        print(f"Loaded {len(test_loaders)} test splits: {list(test_loaders.keys())}")

    # prepare model
    print(f"Creating {args.model.upper()} model...")
    ModelClass = MODEL_TYPES[args.model]

    model_kwargs = {"num_classes": num_classes}

    # add model-specific parameters
    if args.model in ["mlp", "vit", "patchtst", "timesformer1d"]:
        model_kwargs.update({"win_len": args.win_len, "feature_size": args.feature_size})

    # ResNet18 specific parameters:
    if args.model == "resnet18":
        model_kwargs.update({'in_channels': args.in_channels})
    
    # LSTM specific parameters
    if args.model == 'lstm':
        model_kwargs.update({'feature_size': args.feature_size})
    
    # Transformer specific parameters
    if args.model == 'transformer':
        model_kwargs.update({
            'feature_size': args.feature_size,
            'd_model': args.d_model,
            'dropout': args.dropout
        })
    
    # ViT specific parameters
    if args.model == 'vit':
        model_kwargs.update({
            'emb_dim': args.emb_dim,
            'dropout': args.dropout
        })
    
    # PatchTST specific parameters
    if args.model == 'patchtst':
        model_kwargs.update({
            'patch_len': args.patch_len,
            'stride': args.stride,
            'emb_dim': args.emb_dim, 
            'pool': args.pool,
            'head_dropout': args.head_dropout,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'dropout': args.dropout
        })
    
    # TimesFormer-1D specific parameters
    if args.model == 'timesformer1d':
        model_kwargs.update({
            'patch_size': args.patch_size,
            'emb_dim': args.emb_dim,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'attn_dropout': args.attn_dropout,
            'head_dropout': args.head_dropout,
            'mlp_ratio': args.mlp_ratio,
            'dropout': args.dropout
        })
    
    # initialize model
    model = ModelClass(**model_kwargs)

    # if coming from pretrained model then load that in
    if args.pretrained_encoder is not None:
        print(f"Loading pretrained encoder weights from {args.pretrained_encoder}")
        state_dict = torch.load(args.pretrained_encoder, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys (expected — new head): {missing}")
        print(f"  Unexpected keys: {unexpected}")
        print("Pretrained encoder loaded successfully.")

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # setup optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    # create step-level warmup cosine LR scheduler (matches original repo)
    num_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def warmup_cosine_schedule(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    print(f"Using step-level warmup cosine LR scheduler: {warmup_steps} warmup steps, {num_steps} total steps")

    # save configuration
    config_path = os.path.join(results_dir, f"{args.model}_{args.task}_config.yaml")
    save_config(args, config_path)
    print(f"Successfully saved configuration to {os.path.abspath(config_path)}")

    # create TaskTrainer
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=checkpoint_dir,
        num_classes=num_classes,
        config=args,
        label_mapper=label_mapper
    )

    # train model with early stopping
    model, training_results = trainer.train()
    best_epoch = training_results["best_epoch"]
    history = training_results["training_dataframe"]

    # evaluate on test dataset
    print("\nEvaluating on test splits:")
    all_results = {}
    
    import gc
    
    for key, loader in test_loaders.items():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Evaluating on {key} split:")
        loss, accuracy = trainer.evaluate(loader)
        f1_score, _ = trainer.calculate_metrics(loader)

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "f1_score": f1_score
        }
        all_results[key] = metrics
        print(f"{key} accuracy: {metrics['accuracy']:.4f}, F1-score: {metrics['f1_score']:.4f}")

        # generate confusion matrix
        print(f"Generating confusion matrix for {key} split...")
        confusion_path = os.path.join(results_dir, f"{args.model}_{args.task}_{key}_confusion.png")
        trainer.plot_confusion_matrix(data_loader=loader, mode=key)

    # create a summary table with all test results
    summary_table = []
    for split_name, metrics in all_results.items():
        row = {
            'Split': split_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        }
        summary_table.append(row)

    # print summary table
    print("\nTest Results Summary:")
    print(f"{'Split':<25} {'Accuracy':<15} {'F1-Score':<15}")
    print('-' * 55)
    for row in summary_table:
        print(f"{row['Split']:<25} {row['Accuracy']:<15} {row['F1-Score']:<15}")
    
    # save test results
    results_file = os.path.join(results_dir, f"{args.model}_{args.task}_results.json")

    # some objects in the metrics might not be JSON serializable, so we need to convert them
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # process all keys and values in all_results
    for key in all_results:
        all_results[key] = {k: convert_to_json_serializable(v) for k, v in all_results[key].items()}
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # save summary that includes training history, best epoch
    summary = {
        'best_epoch': best_epoch,
        'best_val_loss': float(history.iloc[best_epoch-1]['Val Loss']),
        'best_val_accuracy': float(history.iloc[best_epoch-1]['Val Accuracy']),
        'experiment_id': experiment_id,
        'experiment_completed': True
    }

    # add test results to summary
    for split_name, metrics in all_results.items():
        summary[f'{split_name}_accuracy'] = metrics['accuracy']
        summary[f'{split_name}_f1_score'] = metrics['f1_score']
    
    summary_file = os.path.join(results_dir, f"{args.model}_{args.task}_summary.json")
    summary = {k: convert_to_json_serializable(v) for k, v in summary.items()}
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save training history
    history_file = os.path.join(results_dir, f"{args.model}_{args.task}_train_history.csv")
    history.to_csv(history_file, index=False)
    
    print("\nTraining and evaluation completed.")
    print(f"Best model from epoch {best_epoch}, saved to {checkpoint_dir}")
    print(f"Results saved to {results_dir}")

    # Check if this is the best model for the task so far
    # Update best_performance.json if performance improved
    best_performance_path = os.path.join(model_level_dir, "best_performance.json")
    
    try:
        # Get current validation accuracy
        val_accuracy = float(history.iloc[best_epoch-1]['Val Accuracy'])
        
        # Create dictionaries for test accuracies and F1 scores
        test_accuracies = {}
        test_f1_scores = {}
        for split_name, metrics in all_results.items():
            test_accuracies[split_name] = metrics['accuracy']
            test_f1_scores[split_name] = metrics['f1_score']
        
        # Load current best performance 
        if os.path.exists(best_performance_path):
            with open(best_performance_path, 'r') as f:
                best_performance = json.load(f)
            
            # Get current best validation accuracy
            current_best_val = best_performance.get('best_val_accuracy', 0.0)
            
            # Compare using validation accuracy
            if val_accuracy > current_best_val:
                print(f"New best model! Validation accuracy: {val_accuracy:.4f} (previous best: {current_best_val:.4f})")
                
                # Create updated best performance
                updated_performance = {
                    'best_val_accuracy': val_accuracy,
                    'best_val_loss': float(history.iloc[best_epoch-1]['Val Loss']),
                    'best_experiment_id': experiment_id,
                    'best_experiment_params': vars(args),
                    'best_test_accuracies': test_accuracies,
                    'best_test_f1_scores': test_f1_scores,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save updated best performance
                with open(best_performance_path, 'w') as f:
                    json.dump(updated_performance, f, indent=4)
            else:
                print(f"Not the best model. Current validation accuracy: {val_accuracy:.4f} (best: {current_best_val:.4f})")
        else:
            # First time creating the file
            print(f"Creating initial best performance record with validation accuracy: {val_accuracy:.4f}")
            
            initial_performance = {
                'best_val_accuracy': val_accuracy,
                'best_val_loss': float(history.iloc[best_epoch-1]['Val Loss']),
                'best_experiment_id': experiment_id,
                'best_experiment_params': vars(args),
                'best_test_accuracies': test_accuracies,
                'best_test_f1_scores': test_f1_scores,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Create directory if needed
            os.makedirs(os.path.dirname(best_performance_path), exist_ok=True)
            
            # Save initial best performance
            with open(best_performance_path, 'w') as f:
                json.dump(initial_performance, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to update best_performance.json: {e}")
        import traceback
        traceback.print_exc()
    
    return summary, all_results, model

if __name__ == '__main__':
    import math  # Import math here for the scheduler function
    main()

import os
import sys
import argparse
import torch
import time
import uuid
import yaml
import json
import torch.nn as nn

# Ensure we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load.dataloader import get_loaders
from scripts.train_supervised import MODEL_TYPES
from engine.task_trainer import TaskTrainer
from model.multitask.models import MultiTaskAdapterModel, PatchTSTAdapterModel, TimesFormerAdapterModel
import pandas as pd
from sklearn.metrics import f1_score

def generate_experiment_id():
    """Generate a unique experiment ID for tracking results."""
    timestamp = int(time.time())
    short_uuid = str(uuid.uuid4())[:8]
    return f"multitask_{timestamp}_{short_uuid}"

def parse_args():
    # Get project root directory (two levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_save_dir = os.path.join(project_root, 'results', 'multitask')
    
    parser = argparse.ArgumentParser("Multi-task Adapter Training", add_help=False)
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of task names')
    parser.add_argument('--model', type=str, required=True,
                        help='Backbone model type, same as supervised pipeline')
    parser.add_argument('--data_dir', '--training_dir', dest='data_dir', type=str,
                        default='data',
                        help='Dataset root directory (alias: --training_dir)')
    parser.add_argument('--win_len', type=int, help='window length for MLP/VIT')
    parser.add_argument('--feature_size', type=int, help='feature size for LSTM/Transformer/VIT')
    parser.add_argument('--in_channels', type=int, help='input channels for ResNet18')
    parser.add_argument('--emb_dim', type=int, help='embedding dim for VIT/Transformer')
    parser.add_argument('--dropout', type=float, help='dropout for VIT/Transformer')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default=default_save_dir,
                       help='Directory to save results, defaults to PROJECT_ROOT/results/multitask')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience (in epochs) for early stopping')

    # PatchTST specific args
    parser.add_argument('--patch_len', type=int, default=16, 
                        help='Patch length for PatchTST')
    parser.add_argument('--stride', type=int, default=8, 
                        help='Stride for patches in PatchTST')
    parser.add_argument('--pool', type=str, default='cls', choices=['cls', 'mean'],
                        help='Pooling method for PatchTST')
    parser.add_argument('--head_dropout', type=float, default=0.2,
                        help='Dropout rate for classification head')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of transformer layers') 
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')

    # TimesFormer-1D specific args
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size for TimesFormer-1D')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='Dropout rate for attention layers')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP ratio for transformer blocks')
    
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--use_root_data_path', action='store_true',
                        help='Use root directory as task directory')
    parser.add_argument('--file_format', type=str, default='h5',
                        help='File format for data files (h5, mat, or npy)')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='Key for CSI data in h5 files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Enable pin memory for data loading (not recommended for MPS)')
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Disable pin memory for data loading (use for MPS devices)')
    parser.add_argument('--help', '-h', action='help')

    # optional pretrained weights
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='Path to pretrained encoder weights (.pt file)')
    
    args = parser.parse_args()
    return args


def main():
    # Now args contains everything from the CLI and the YAML
    args = parse_args()
    
    # Set up absolute path for saving
    if not os.path.isabs(args.save_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.save_dir = os.path.join(project_root, args.save_dir)
    
    os.makedirs(args.save_dir, exist_ok=True)
    experiment_id = generate_experiment_id()
    print(f"Experiment ID: {experiment_id}")

    # Task Handling
    tasks = args.tasks.split(',')
    print(f"Initializing Multitask Pipeline for: {tasks}")
    
    # Loader Initialization Loop
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    task_classes = {}

    # add handling for None values to prevent dataloader errors
    def custom_collate_fn(batch):
        # filter out None samples
        batch = [item for item in batch if item is not None]
        
        # if no samples remain after filtering, return empty tensors
        if len(batch) == 0:
            return torch.zeros(0, 1, args.win_len, args.feature_size), torch.zeros(0, dtype=torch.long)
        
        # use default collate function for the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)
    
    for task in tasks:
        print(f"--- Loading Data for Task: {task} ---")
        data = get_loaders(
            root=args.data_dir,
            task=task,
            batch_size=args.batch_size,
            file_format=args.file_format,
            data_key=args.data_key,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=custom_collate_fn,
            test_splits=args.test_splits.split(',') if args.test_splits != 'all' else 'all'
        )
        
        ld = data['loaders']
        train_loaders[task] = ld['train']
        val_loaders[task] = ld.get('val', ld['train'])
        test_loaders[task] = {k: v for k, v in ld.items() if k.startswith('test')}
        task_classes[task] = data['num_classes']

    # --- Dimension & Parameter Logic ---
    
    # Infer feature_size from data if it's still None
    if args.feature_size is None:
        first_loader = next(iter(train_loaders.values()))
        sample_x, _ = next(iter(first_loader))
        args.feature_size = sample_x.shape[-1]

    # Set defaults for attributes that might be None
    args.win_len = args.win_len or 232
    args.emb_dim = args.emb_dim or 128
    args.dropout = args.dropout if args.dropout is not None else 0.1
    args.depth = args.depth or 4
    args.num_heads = args.num_heads or 4

    # --- Build Backbone ---
    ModelClass = MODEL_TYPES[args.model]
    # Get num_classes from the first task
    first_task = next(iter(task_classes))
    model_kwargs = {'num_classes': task_classes[first_task]}

    # Map parameters to model_kwargs
    if args.model in ['mlp', 'vit', 'patchtst', 'timesformer1d', 'transformer']:
        model_kwargs.update({'win_len': args.win_len, 'feature_size': args.feature_size})
    
    if args.model == 'resnet18':
        model_kwargs.update({'in_channels': args.in_channels or 1})
    elif args.model == 'vit':
        model_kwargs.update({'emb_dim': args.emb_dim, 'dropout': args.dropout})
    elif args.model == 'transformer':
        model_kwargs.update({'d_model': args.emb_dim, 'dropout': args.dropout})
    elif args.model == 'patchtst':
        model_kwargs.update({
            'patch_len': args.patch_len or 16,
            'stride': args.stride or 8,
            'emb_dim': args.emb_dim,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'head_dropout': args.head_dropout or 0.2,
            'pool': args.pool or 'cls'
        })
    elif args.model == 'timesformer1d':
        model_kwargs.update({
            'patch_size': args.patch_size or 4,
            'emb_dim': args.emb_dim,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'attn_dropout': args.attn_dropout or 0.1,
            'head_dropout': args.head_dropout or 0.2,
            'mlp_ratio': args.mlp_ratio or 4.0
        })

    print(f"Creating {args.model} backbone model with kwargs: {model_kwargs}")
    backbone_model = ModelClass(**model_kwargs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # load pretrained weights if there are any
    pretrained_path = getattr(args, 'pretrained_encoder', None)
    if pretrained_path is not None:
        # Resolve relative paths
        if not os.path.isabs(pretrained_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pretrained_path = os.path.join(project_root, pretrained_path)
            
        if os.path.exists(pretrained_path):
            print(f"\n[+] Loading pretrained encoder weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device)
            
            # Handle nested state_dicts (common in SSL frameworks)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            missing, unexpected = backbone_model.load_state_dict(state_dict, strict=False)
            print(f"    Missing keys (typically new heads): {len(missing)}")
            print(f"    Unexpected keys (typically old heads): {len(unexpected)}")
            print("[+] Pretrained encoder loaded successfully.\n")
        else:
            print(f"\n[!] WARNING: Pretrained weights not found at {pretrained_path}. Starting from scratch.\n")

    # --- Transformer Embedding Wrapper ---
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
        
        # Attach config with both dict and attribute access
        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"No such attribute: {name}")
        
        d_model = args.emb_dim
        num_layers = len(backbone_model.transformer.layers)
        backbone.config = ConfigDict(
            model_type="bert",
            hidden_size=d_model,
            num_attention_heads=8,
            num_hidden_layers=num_layers,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=args.dropout,
            use_return_dict=False
        )

        # Create MultiTaskAdapterModel with the backbone
        model = MultiTaskAdapterModel(
            backbone, task_classes,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )

    elif args.model == 'patchtst':
        class PatchTSTEmbedding(nn.Module):
            def __init__(self, cls_model):
                super().__init__()
                self.patch_embedding = cls_model.patch_embedding
                self.pos_embedding = cls_model.pos_embedding
                self.cls_token = cls_model.cls_token
                self.dropout = cls_model.dropout
                self.transformer = cls_model.transformer
                self.norm = cls_model.norm
                self.pool = cls_model.pool
                self.emb_dim = cls_model.emb_dim
                self.feature_size = cls_model.feature_size
                self.win_len = cls_model.win_len

            def forward(self, x):
                # Handle input dimensions
                if len(x.shape) == 4:
                    x = x.squeeze(1)  # Remove channel dimension
                
                # Ensure correct shape [batch, feature_size, win_len]
                if x.shape[1] != x.shape[2] and x.shape[2] == self.feature_size:
                    # Swap dimensions if necessary
                    x = x.transpose(1, 2)
                    
                # Apply patch embedding
                x = self.patch_embedding(x)
                
                # Transpose to [batch, num_patches, emb_dim]
                x = x.transpose(1, 2)
                
                # Add CLS token if using 'cls' pooling
                batch_size = x.shape[0]
                if self.pool == 'cls' and self.cls_token is not None:
                    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                    x = torch.cat((cls_tokens, x), dim=1)
                
                # Add positional encoding
                x = x + self.pos_embedding
                
                # Apply dropout
                x = self.dropout(x)
                
                # Transformer encoder
                x = self.transformer(x)
                
                # Apply layer norm
                x = self.norm(x)
                
                # Pool features according to strategy
                if self.pool == 'cls':
                    x = x[:, 0]  # Take CLS token representation
                else:  # 'mean'
                    x = x.mean(dim=1)  # Mean pooling over patches
                
                return x
                
        backbone = PatchTSTEmbedding(backbone_model)
        
        # Create config for the backbone
        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"No such attribute: {name}")
        
        # Extract information from the backbone for config
        num_layers = 1  # Default value
        if hasattr(backbone_model.transformer, 'layers'):
            num_layers = len(backbone_model.transformer.layers)
        elif hasattr(backbone_model.transformer, 'encoder'):
            num_layers = backbone_model.transformer.encoder.num_layers
        
        backbone.config = ConfigDict(
            hidden_size=backbone_model.emb_dim,
            num_hidden_layers=num_layers
        )
        
        # Create PatchTSTAdapterModel
        model = PatchTSTAdapterModel(
            backbone, task_classes,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        
    elif args.model == 'timesformer1d':
        class TimesFormerEmbedding(nn.Module):
            def __init__(self, cls_model):
                super().__init__()
                self.patch_embed = cls_model.patch_embed
                self.pos_embed = cls_model.pos_embed
                self.cls_token = cls_model.cls_token
                self.blocks = cls_model.blocks
                self.norm = cls_model.norm
                self.emb_dim = cls_model.emb_dim

            def forward(self, x):
                # Handle input dimensions
                if len(x.shape) == 4:
                    x = x.squeeze(1)  # Remove channel dimension
                
                # Ensure shape is [batch, feature_size, win_len]
                if x.shape[1] != x.shape[2] and x.shape[2] == self.feature_size:
                    x = x.transpose(1, 2)
                    
                # Patch embedding: [batch, feature_size, win_len] -> [batch, num_patches, emb_dim]
                x = self.patch_embed(x)
                
                # Add CLS token
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                
                # Add position embedding
                x = x + self.pos_embed
                
                # Apply transformer blocks
                for block in self.blocks:
                    x = block(x)
                    
                # Apply layer norm
                x = self.norm(x)
                
                # Use CLS token for representation
                x = x[:, 0]
                
                return x
                
        backbone = TimesFormerEmbedding(backbone_model)
        
        # Create config for the backbone
        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"No such attribute: {name}")
        
        backbone.config = ConfigDict(
            hidden_size=backbone_model.emb_dim,
            num_hidden_layers=len(backbone_model.blocks)
        )
        
        # Create TimesFormerAdapterModel
        model = TimesFormerAdapterModel(
            backbone, task_classes,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    
    else:
        raise NotImplementedError(f"Multi-task adapter doesn't support {args.model} backbone yet.")

    model.to(device)

    # Optimizer & criterion
    opt_params = list(model.adapters.parameters()) + list(model.heads.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = []
    best_val, no_improve, best_state = float('inf'), 0, None
    
    # Store best metrics and states for each task
    best_metrics = {task: {'val_loss': float('inf'), 'val_acc': 0, 'best_epoch': 0} for task in task_classes}
    best_task_states = {}


    def evaluate(model, loader, criterion, device):
        """Evaluate model performance on the given data loader.
        
        Args:
            model: Model to evaluate
            loader: Data loader
            criterion: Loss function
            device: Device to run evaluation on
            
        Returns:
            tuple: (loss, accuracy, F1 score)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                # Record predictions and true labels for calculating F1 score
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Calculate F1 score
                if len(set(labels.cpu().numpy())) > 1:  # Ensure there are multiple classes
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        if len(all_preds) > 0 and len(all_labels) > 0:
            try:
                f1 = f1_score(all_labels, all_preds, average='weighted')
            except Exception as e:
                print(f"Error calculating F1 score: {e}")
                f1 = 0.0
        else:
            f1 = 0.0
        
        return total_loss / len(loader), correct / total, f1

    for epoch in range(1, args.epochs + 1):
        model.train()
        for task, loader in train_loaders.items():
            model.set_active_task(task)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validation
        row = {'epoch': epoch}
        val_losses = []
        print(f"\nValidation after epoch {epoch}:")
        for task, vloader in val_loaders.items():
            model.set_active_task(task)
            v_l, v_a, v_f1 = evaluate(model, vloader, criterion, device)
            val_losses.append(v_l)
            row[f"{task}_val_loss"] = v_l
            row[f"{task}_val_acc"] = v_a
            row[f"{task}_val_f1"] = v_f1
            print(f"  {task}: loss={v_l:.4f}, acc={v_a:.4f}, f1={v_f1:.4f}")
            
            # Create task-specific directory structure for results
            task_dir = os.path.join(args.save_dir, task, args.model, experiment_id)
            os.makedirs(task_dir, exist_ok=True)
            
            # Update best metrics if improved
            if v_a > best_metrics[task]['val_acc']:
                print(f"  New best accuracy for {task}: {v_a:.4f} (previous: {best_metrics[task]['val_acc']:.4f})")
                best_metrics[task]['val_loss'] = v_l
                best_metrics[task]['val_acc'] = v_a
                best_metrics[task]['val_f1'] = v_f1
                best_metrics[task]['best_epoch'] = epoch
                
                # Save task-specific best state
                model.set_active_task(task)
                best_task_states[task] = {
                    'adapters': model.adapters.state_dict(),
                    'head': model.heads[task].state_dict()
                }

        history.append(row)
        avg_val = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch}: avg_val_loss={avg_val:.4f}")

        # Early stopping based on average validation loss
        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
            best_state = {
                'adapters': model.adapters.state_dict(),
                'heads': model.heads.state_dict()
            }
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best overall state
    if best_state:
        model.adapters.load_state_dict(best_state['adapters'])
        model.heads.load_state_dict(best_state['heads'])

    # Now generate confusion matrices and classification reports for the best model of each task
    print("\nGenerating visualization for the best model of each task...")
    for task in task_classes:
        # Create task-specific directory structure
        task_dir = os.path.join(args.save_dir, task, args.model, experiment_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Create checkpoints directory
        checkpoint_dir = os.path.join(task_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Extract task-specific history
        task_history = []
        for row in history:
            task_row = {
                'epoch': row['epoch'],
                'val_loss': row.get(f"{task}_val_loss", None),
                'val_acc': row.get(f"{task}_val_acc", None),
                'val_f1': row.get(f"{task}_val_f1", None)
            }
            task_history.append(task_row)
        
        # Save history
        pd.DataFrame(task_history).to_csv(os.path.join(task_dir, f"{args.model}_{task}_train_history.csv"), index=False)
        
        # Save configuration
        config = {
            'model': args.model,
            'task': task,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'emb_dim': args.emb_dim,
            'dropout': args.dropout,
            'num_classes': task_classes[task],
            'experiment_id': experiment_id,
            'best_epoch': best_metrics[task]['best_epoch']
        }
        
        # Add model-specific parameters to config
        if args.model == 'patchtst':
            config.update({
                'patch_len': args.patch_len,
                'stride': args.stride,
                'depth': args.depth,
                'num_heads': args.num_heads,
                'pool': args.pool,
                'head_dropout': args.head_dropout
            })
        elif args.model == 'timesformer1d':
            config.update({
                'patch_size': args.patch_size,
                'depth': args.depth,
                'num_heads': args.num_heads,
                'attn_dropout': args.attn_dropout,
                'head_dropout': args.head_dropout,
                'mlp_ratio': args.mlp_ratio
            })
        
        with open(os.path.join(task_dir, f"{args.model}_{task}_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Load task-specific best state if available
        if task in best_task_states:
            model.adapters.load_state_dict(best_task_states[task]['adapters'])
            model.heads[task].load_state_dict(best_task_states[task]['head'])
        
        # Set active task
        model.set_active_task(task)
        
        # Save task-specific model weights
        task_model_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save({
            'adapters': model.adapters.state_dict(),
            'heads': {task: model.heads[task].state_dict()}
        }, task_model_path)
        
        # Generate confusion matrix for the best model on validation set
        print(f"Generating confusion matrix for best {task} model (epoch {best_metrics[task]['best_epoch']})")
        tm = TaskTrainer(
            model=model,
            train_loader=None,
            val_loader=val_loaders[task],
            test_loader=None,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            device=device,
            save_path=task_dir,
            num_classes=task_classes[task],
            label_mapper=val_loaders[task].dataset.label_mapper,
            config=None
        )
        tm.plot_confusion_matrix(data_loader=val_loaders[task], epoch=best_metrics[task]['best_epoch'], mode='val_best')
        
        # Save summary
        summary = {
            'experiment_id': experiment_id,
            'best_val_loss': best_metrics[task]['val_loss'],
            'best_val_acc': best_metrics[task]['val_acc'],
            'best_val_f1': best_metrics[task]['val_f1'],
            'best_epoch': best_metrics[task]['best_epoch'],
            'total_epochs': len(history),
            'early_stopped': no_improve >= args.patience,
            'model': args.model,
            'task': task
        }
        with open(os.path.join(task_dir, f"{args.model}_{task}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Update best_performance.json
        best_perf_path = os.path.join(args.save_dir, task, args.model, "best_performance.json")
        try:
            if os.path.exists(best_perf_path):
                with open(best_perf_path, 'r') as f:
                    best_perf = json.load(f)
                
                # Update only when validation accuracy improves
                if best_metrics[task]['val_acc'] > best_perf.get('best_val_accuracy', 0):
                    best_perf = {
                        'best_experiment_id': experiment_id,
                        'best_val_accuracy': best_metrics[task]['val_acc'],
                        'best_val_loss': best_metrics[task]['val_loss'],
                        'best_val_f1': best_metrics[task]['val_f1'],
                        'best_epoch': best_metrics[task]['best_epoch'],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
            else:
                best_perf = {
                    'best_experiment_id': experiment_id,
                    'best_val_accuracy': best_metrics[task]['val_acc'],
                    'best_val_loss': best_metrics[task]['val_loss'],
                    'best_val_f1': best_metrics[task]['val_f1'],
                    'best_epoch': best_metrics[task]['best_epoch'],
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            with open(best_perf_path, 'w') as f:
                json.dump(best_perf, f, indent=2)
        except Exception as e:
            print(f"Error updating best_performance.json: {e}")

    # Final test evaluation
    print("\nFinal test results (using best model for each task):")
    for task, tdict in test_loaders.items():
        task_dir = os.path.join(args.save_dir, task, args.model, experiment_id)
        
        # Load task-specific best state if available
        if task in best_task_states:
            model.adapters.load_state_dict(best_task_states[task]['adapters'])
            model.heads[task].load_state_dict(best_task_states[task]['head'])
        
        # Set active task
        model.set_active_task(task)
        
        test_results = {}
        
        # Create a summary table for this task's test results
        print(f"\nTest Results for {task}:")
        print(f"{'Split':<20} {'Accuracy':<15} {'Loss':<15} {'F1 Score':<15}")
        print('-' * 50)
        
        for split, tloader in tdict.items():
            t_l, t_a, t_f1 = evaluate(model, tloader, criterion, device)
            print(f"{split:<20} {t_a:.4f}{'':<10} {t_l:.4f} F1: {t_f1:.4f}")
            
            # Save test results
            test_results[split] = {
                'loss': t_l,
                'accuracy': t_a,
                'f1_score': t_f1
            }
            
            # Generate confusion matrix for test set
            tm = TaskTrainer(
                model=model,
                train_loader=None,
                val_loader=tloader,
                test_loader=None,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                device=device,
                save_path=task_dir,
                num_classes=task_classes[task],
                label_mapper=tloader.dataset.label_mapper,
                config=None
            )
            tm.plot_confusion_matrix(data_loader=tloader, epoch=None, mode=split)
            
            # Update best_performance.json, add test results
            try:
                best_perf_path = os.path.join(args.save_dir, task, args.model, "best_performance.json")
                if os.path.exists(best_perf_path):
                    with open(best_perf_path, 'r') as f:
                        best_perf = json.load(f)
                    
                    # Update only when validation accuracy improves
                    if best_perf.get('best_experiment_id') == experiment_id:
                        # Update test results
                        if 'best_test_accuracies' not in best_perf:
                            best_perf['best_test_accuracies'] = {}
                        
                        if 'best_test_f1_scores' not in best_perf:
                            best_perf['best_test_f1_scores'] = {}
                        
                        # Update test results
                        best_perf['best_test_accuracies'][split] = t_a
                        best_perf['best_test_f1_scores'][split] = t_f1
                        
                        with open(best_perf_path, 'w') as f:
                            json.dump(best_perf, f, indent=2)
            except Exception as e:
                print(f"Error updating best_performance.json with test results: {e}")
        
        # Save test results to a file
        with open(os.path.join(task_dir, f"{args.model}_{task}_test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)


         # Create a comprehensive test summary that includes all splits
        test_summary = {
            'task': task,
            'model': args.model,
            'experiment_id': experiment_id,
            'test_splits': list(test_results.keys()),
            'best_validation': {
                'accuracy': best_metrics[task]['val_acc'],
                'loss': best_metrics[task]['val_loss'],
                'f1_score': best_metrics[task]['val_f1'],
                'epoch': best_metrics[task]['best_epoch']
            },
            'test_results': test_results
        }

        with open(os.path.join(task_dir, f"{args.model}_{task}_test_summary.json"), 'w') as f:
            json.dump(test_summary, f, indent=2)

        # Save full multitask model (the overall best state)
    full_model_dir = os.path.join(args.save_dir, "shared_models")
    os.makedirs(full_model_dir, exist_ok=True)
    torch.save(
        {'adapters': best_state['adapters'], 'heads': best_state['heads']},
        os.path.join(full_model_dir, f'multitask_adapters_{experiment_id}.pt')
    )
    
    print("\nMultitask training completed successfully.")
    print(f"Results saved to: {args.save_dir}")
    print(f"Experiment ID: {experiment_id}")

if __name__ == "__main__":
    main()
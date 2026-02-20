import os
import yaml
import argparse
from typing import Dict, Any

def update_args_with_yaml(args: argparse.Namespace, yaml_path: str) -> argparse.Namespace:
    """Read a YAML config and update the argparse namespace."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config:
        return args
        
    # Flatten the config to update arguments.
    # Prioritizes CLI arguments if they were specifically provided, 
    # but for simplicity we directly update args from config keys.
    flat_config = {}
    for section_name, section_dict in config.items():
        if isinstance(section_dict, dict):
            # Special handling for common sections like optimizer, scheduler, model_params, training
            for k, v in section_dict.items():
                if k == "name" and section_name in ["optimizer", "scheduler"]:
                    flat_config[section_name] = v
                else:
                    flat_config[k] = v
        else:
            flat_config[section_name] = section_dict
            
    # Update args
    for k, v in flat_config.items():
        setattr(args, k, v)
        
    return args

def save_config(args: argparse.Namespace, save_path: str) -> None:
    """Save the arguments to a YAML config organically structured."""
    
    # Extract as dict
    try:
        args_dict = vars(args)
    except TypeError:
        # It's already a dict or can't be vars()'d
        args_dict = dict(args) if isinstance(args, dict) else {}
        
    # Group logically
    training_keys = [
        'epochs', 'batch_size', 'learning_rate', 'lr', 'weight_decay', 
        'warmup_epochs', 'patience', 'grad_clip', 'log_interval'
    ]
    model_keys = [
        'emb_dim', 'd_model', 'dropout', 'depth', 'num_heads', 'patch_len',
        'stride', 'pool', 'head_dropout', 'patch_size', 'attn_dropout', 
        'mlp_ratio', 'in_channels', 'win_len', 'feature_size'
    ]
    
    config_struct = {
        'training': {},
        'model_params': {},
        'optimizer': {},
        'scheduler': {},
        'general': {}
    }
    
    for k, v in args_dict.items():
        # Handle renaming
        if k == 'optimizer':
            config_struct['optimizer']['name'] = v
        elif k == 'scheduler':
            config_struct['scheduler']['name'] = v
        elif k == 'learning_rate' or k == 'lr':
            config_struct['optimizer']['lr'] = v
        elif k == 'weight_decay':
            config_struct['optimizer']['weight_decay'] = v
        elif k in training_keys:
            config_struct['training'][k] = v
        elif k in model_keys:
            config_struct['model_params'][k] = v
        else:
            config_struct['general'][k] = v
            
    # Clean up empty sections
    config_struct = {k: v for k, v in config_struct.items() if v}
    
    with open(save_path, 'w') as f:
        yaml.dump(config_struct, f, sort_keys=False)

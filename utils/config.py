import os
import yaml
import argparse
from typing import Dict, Any

import sys

def update_args_with_yaml(args: argparse.Namespace, yaml_path: str) -> argparse.Namespace:
    """Read a YAML config and update the argparse namespace."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config:
        return args
        
    # Flatten the config to update arguments.
    flat_config = {}
    for section_name, section_dict in config.items():
        if isinstance(section_dict, dict):
            for k, v in section_dict.items():
                if k == "name" and section_name in ["optimizer", "scheduler"]:
                    flat_config[section_name] = v
                else:
                    flat_config[k] = v
                    # Handle common aliases automatically
                    if k == "lr":
                        flat_config["learning_rate"] = v
                    elif k == "learning_rate":
                        flat_config["lr"] = v
        else:
            flat_config[section_name] = section_dict

    # Identify which arguments were explicitly provided in the command line
    cli_args = []
    for arg in sys.argv:
        if arg.startswith('--'):
            cli_args.append(arg.lstrip('-').split('=')[0])
            
    # Update args
    for k, v in flat_config.items():
        # 1. COMMAND LINE WINS: Skip if we explicitly passed this flag in the terminal
        if k in cli_args:
            continue
            
        # 2. Check if the argument exists in the parser
        if hasattr(args, k):
            existing_v = getattr(args, k)
            
            # FIX THE BRACKETS: Safely convert YAML lists to comma-separated strings
            if isinstance(v, list) and isinstance(existing_v, str):
                v = ",".join(str(x) for x in v)

            if existing_v is not None and v is not None:
                try:
                    # Special case for booleans
                    if isinstance(existing_v, bool):
                        if isinstance(v, str):
                            v = v.lower() in ("yes", "true", "t", "1")
                        else:
                            v = bool(v)
                    else:
                        v = type(existing_v)(v)
                except (ValueError, TypeError):
                    pass
            setattr(args, k, v)
        else:
            # 3. OVERRIDE: If the parser didn't know about this argument, 
            # inject it into the namespace anyway so the model can access it!
            if isinstance(v, list):
                # If it's a list that needs to act like a CLI string, convert it
                v = ",".join(str(x) for x in v)
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

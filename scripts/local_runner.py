#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script serves as the main entry point for WiFi sensing benchmark.
It incorporates functionality from train.py, run_model.py, and the original local_runner.py.

Configuration File Management:
1. The configs folder now only contains template configuration files
2. Generated configuration files are saved to the results folder using a unified directory structure: results/TASK/MODEL/EXPERIMENT_ID/
   - Supervised learning: results/TASK/MODEL/EXPERIMENT_ID/supervised_config.json
   - Multitask learning: results/TASK/MODEL/EXPERIMENT_ID/multitask_config.json
3. All runtime parameters should be loaded from the configuration file, command-line arguments are no longer used

Usage:
    python local_runner.py --config_file [config_path]
    
Additional parameters:
    --config_file: JSON configuration file to use for all settings
"""

import os
import sys
import subprocess
import torch
import time
import argparse
import json
from datetime import datetime
import importlib.util
import pandas as pd
import yaml

# Fix encoding issues on Windows
import io
import locale

# Try to set UTF-8 mode for Windows
if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')

# Set stdout encoding to UTF-8 if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"root_dir is {ROOT_DIR}")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "multitask_config_test.yaml")

# Ensure results directory exists
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

def validate_config(config, required_fields=None):
    """
    Validate if the configuration contains all necessary parameters
    
    Args:
        config: Configuration dictionary
        required_fields: List of required fields, if None use default required fields
        
    Returns:
        True if validation succeeds, False otherwise
    """
    if required_fields is None:
        # Define basic required fields
        required_fields = [
            ("training", "epochs"),
            ("training", "batch_size"),
            ("model_params", "win_len"),
            ("model_params", "feature_size"),
            ("pipeline", None), # Pipeline is top-level
            ("training_dir", None),
            ("output_dir", None)
        ]
        
    missing_fields = []
    for parent, child in required_fields:
        if child:
            if parent not in config or child not in config[parent]:
                missing_fields.append(f"{parent}.{child}")
        else:
            if parent not in config:
                missing_fields.append(parent)
    
    # Task validation (Supervised vs Multitask)
    if "task" not in config and "tasks" not in config:
        missing_fields.append("task (for supervised) or tasks (for multitask)")
    
    if missing_fields:
        print(f"Error: Missing YAML parameters: {', '.join(missing_fields)}")
        return False

    # Pipeline logic check
    valid_pipelines = ["supervised", "multitask"]
    if config.get("pipeline") not in valid_pipelines:
        print(f"Error: 'pipeline' must be one of {valid_pipelines}")
        return False
    
    return True


def load_config(config_path=None):
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if not config or not validate_config(config):
            sys.exit(1)
            
        print(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)


CONFIG = load_config(DEFAULT_CONFIG_PATH)

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("CUDA not available. Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Neither CUDA nor MPS available. Using CPU.")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Set device string for command line arguments
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

def run_command(cmd, display_output=True, timeout=1800):
    """
    Run command and display output in real-time with timeout handling.
    
    Args:
        cmd: Command to execute
        display_output: Whether to display command output
        timeout: Command execution timeout in seconds, default 30 minutes
        
    Returns:
        Tuple of (return_code, output_string)
    """
    try:
        # Start process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            shell=True
        )
        
        # For storing output
        output = []
        start_time = time.time()
        
        # Main loop
        while process.poll() is None:
            # Check for timeout
            if timeout and time.time() - start_time > timeout:
                if display_output:
                    print(f"\nError: Command execution timed out ({timeout} seconds), terminating...")
                process.kill()
                return -1, '\n'.join(output + [f"Error: Command execution timed out ({timeout} seconds)"])
            
            # Read output line by line without blocking
            try:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    if display_output:
                        print(line)
                    output.append(line)
                else:
                    # Small sleep to reduce CPU usage
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error reading output: {str(e)}")
                time.sleep(0.1)
        
        # Ensure all remaining output is read
        remaining_output, _ = process.communicate()
        if remaining_output:
            for line in remaining_output.splitlines():
                if display_output:
                    print(line)
                output.append(line)
                
        return process.returncode, '\n'.join(output)
        
    except KeyboardInterrupt:
        # User interruption
        if 'process' in locals() and process.poll() is None:
            print("\nUser interrupted, terminating process...")
            process.kill()
        return -2, "User interrupted execution"
        
    except Exception as e:
        # Other exceptions
        error_msg = f"Error executing command: {str(e)}"
        if display_output:
            print(f"\nError: {error_msg}")
        
        # Kill process if still running
        if 'process' in locals() and process.poll() is None:
            process.kill()
        
        return -1, error_msg
    

def get_supervised_config(custom_config=None):
    if custom_config is None:
        print("Error: Configuration parameters must be provided!")
        sys.exit(1)
    
    # Helper to pull from nested or flat config
    train_cfg = custom_config.get('training', {})
    opt_cfg = custom_config.get('optimizer', {})
    model_cfg = custom_config.get('model_params', {})
    
    config = {
        # Data parameters
        'training_dir': custom_config['training_dir'],
        'test_dirs': custom_config.get('test_dirs', []),
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config.get('model', 'model')}_{custom_config.get('task', 'task').lower()}",
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        
        # Training parameters (Pulling from 'training' block)
        'batch_size': train_cfg.get('batch_size', custom_config.get('batch_size')),
        'epochs': train_cfg.get('epochs', custom_config.get('epochs')),
        'patience': train_cfg.get('patience', 15),
        
        # Optimizer/Scheduler (Pulling from 'optimizer' block)
        'learning_rate': opt_cfg.get('lr', 1e-4),
        'weight_decay': opt_cfg.get('weight_decay', 1e-5),
        'warmup_epochs': custom_config.get('scheduler', {}).get('warmup_epochs', 5),
        
        # Model & Logic
        'integrated_loader': True,
        'task': custom_config.get('task'),
        'seed': custom_config.get('seed', 42),
        'device': DEVICE,
        'model': custom_config.get('model'),
        'win_len': model_cfg.get('win_len', custom_config.get('win_len')),
        'feature_size': model_cfg.get('feature_size', custom_config.get('feature_size')),
        'test_splits': custom_config.get('test_splits', 'all')
    }
    
    if 'model_params' in custom_config:
        config['model_params'] = custom_config['model_params']
    
    return config

def get_multitask_config(custom_config=None):
    if custom_config is None:
        print("Error: Configuration parameters must be provided!")
        sys.exit(1)
    
    # Extract tasks
    tasks = custom_config.get('tasks')
    if isinstance(tasks, str):
        tasks = tasks.split(',')
    
    if not tasks:
        print("Error: 'tasks' parameter is missing or empty!")
        sys.exit(1)
        
    train_cfg = custom_config.get('training', {})
    opt_cfg = custom_config.get('optimizer', {})
    model_cfg = custom_config.get('model_params', {})

    config = {
        'training_dir': custom_config['training_dir'],
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config.get('model', 'model')}_multitask",
        
        'batch_size': train_cfg.get('batch_size', custom_config.get('batch_size')),
        'epochs': train_cfg.get('epochs', custom_config.get('epochs')),
        'learning_rate': opt_cfg.get('lr', 5e-4),
        'weight_decay': opt_cfg.get('weight_decay', 1e-5),
        'num_workers': train_cfg.get('num_workers', 4),
        
        'win_len': model_cfg.get('win_len', custom_config.get('win_len')),
        'feature_size': model_cfg.get('feature_size', custom_config.get('feature_size')),
        
        'model': custom_config.get('model'),
        'emb_dim': model_cfg.get('emb_dim', 128),
        'dropout': model_cfg.get('dropout', 0.1),
        
        'task': 'multitask',
        'tasks': tasks,
    }

    # If you still want to allow an external override from a JSON
    transform_path = os.path.join(CONFIG_DIR, "transformer_config.json")
    if os.path.exists(transform_path):
        with open(transform_path, 'r') as f:
            transformer_config = json.load(f)
            config.update(transformer_config)
    
    if 'model_params' in custom_config:
        config['model_params'] = custom_config['model_params']
    
    return config

def run_supervised_direct(config):
    """
    Run supervised learning pipeline directly.
    """
    # Get necessary parameters
    task_name = config.get('task')
    model_name = config.get('model')
    training_dir = config.get('training_dir')
    base_output_dir = config.get('output_dir')
    
    # Build basic command
    executable = f'"{sys.executable}"' if ' ' in sys.executable else sys.executable
    script_path = f'"{os.path.join(SCRIPT_DIR, "train_supervised.py")}"' if ' ' in SCRIPT_DIR else os.path.join(SCRIPT_DIR, 'train_supervised.py')
    
    cmd = f"{executable} {script_path}"
    
    # Properly quote paths
    quoted_training_dir = f'"{training_dir}"'
    quoted_output_dir = f'"{base_output_dir}"'
    
    cmd += f" --data_dir={quoted_training_dir}"
    cmd += f" --task={task_name}"
    cmd += f" --model={model_name}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --win_len={config.get('win_len')}"
    cmd += f" --feature_size={config.get('feature_size')}"
    cmd += f" --save_dir={quoted_output_dir}"
    cmd += f" --output_dir={quoted_output_dir}"
    
    cmd += " --num_workers=0 --use_root_data_path --no_pin_memory"
    
    if 'test_splits' in config:
        cmd += f" --test_splits=\"{config['test_splits']}\""
    
    # Add other model-specific parameters
    important_params = ['learning_rate', 'weight_decay', 'warmup_epochs', 'patience', 
                         'emb_dim', 'dropout', 'd_model']
    for param in important_params:
        if param in config:
            cmd += f" --{param}={config[param]}"
    
    # Add parameters from model_params block (crucial for your new YAML structure)
    if 'model_params' in config:
        for key, value in config['model_params'].items():
            cmd += f" --{key}={value}"
    
    print(f"Running supervised learning: {cmd}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    experiment_id = None
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        if "Experiment ID:" in line:
            experiment_id = line.split("Experiment ID:")[1].strip()
    
    return_code = process.wait()
    
    if return_code == 0 and experiment_id:
        # Save as YAML instead of JSON
        exp_dir = os.path.join(base_output_dir, task_name, model_name, experiment_id)
        config_filename = os.path.join(exp_dir, "supervised_config.yaml") # Changed extension
        
        try:
            os.makedirs(exp_dir, exist_ok=True)
            with open(config_filename, 'w', encoding='utf-8') as f:
                # Using yaml.dump to keep your new style
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"Configuration saved to: {config_filename}")
        except Exception as e:
            print(f"Error saving YAML config: {str(e)}")
    
    return return_code


def run_multitask_direct(config):
    """
    Run multitask learning pipeline with YAML configuration support.
    """
    print("Running multitask learning with the following configuration:")
    # Pretty print for the console
    print(yaml.dump(config, default_flow_style=False))
    
    # Extract blocks for easier access
    opt_cfg = config.get('optimizer', {})
    model_params = config.get('model_params', {})
    
    # Get and format tasks
    tasks = config.get('tasks')
    if not tasks:
        print("Error: 'tasks' parameter is missing. Specify a list in your YAML.")
        return 1
    
    # Convert YAML list ['task1', 'task2'] to string "task1,task2" for the CLI
    if isinstance(tasks, list):
        tasks_str = ','.join(tasks)
    else:
        tasks_str = tasks

    # Basic metadata
    task_name = 'multitask'
    model_name = config.get('model')
    base_output_dir = config.get('output_dir')
    
    # Path handling
    executable = f'"{sys.executable}"' if ' ' in sys.executable else sys.executable
    script_path = os.path.join(SCRIPT_DIR, "train_multitask_adapter.py")
    if ' ' in script_path: script_path = f'"{script_path}"'
    
    cmd = f"{executable} {script_path}"
    
    # Core CLI arguments
    training_dir = config.get('training_dir')
    cmd += f" --tasks=\"{tasks_str}\""
    cmd += f" --model={model_name}"
    cmd += f" --data_dir=\"{training_dir}\""
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --num_workers={config.get('num_workers', 4)}"
    
    # Pass model_params (win_len, feature_size, etc.)
    for key, value in model_params.items():
        cmd += f" --{key}={value}"
        
    # Standard flags
    cmd += " --use_root_data_path --no_pin_memory"
    
    # Add optimizer details if they exist in the new YAML block
    if 'lr' in opt_cfg:
        cmd += f" --lr={opt_cfg['lr']}"
    if 'weight_decay' in opt_cfg:
        cmd += f" --weight_decay={opt_cfg['weight_decay']}"

    print(f"Running command: {cmd}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    experiment_id = None
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        if "Experiment ID:" in line:
            experiment_id = line.split("Experiment ID:")[1].strip()
    
    return_code = process.wait()
    
    if return_code == 0 and experiment_id:
        exp_dir = os.path.join(base_output_dir, task_name, model_name, experiment_id)
        config_filename = os.path.join(exp_dir, "multitask_config.yaml")
        
        try:
            os.makedirs(exp_dir, exist_ok=True)
            # Save the full config as YAML for reproducibility
            with open(config_filename, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"Configuration saved to: {config_filename}")
        except Exception as e:
            print(f"Error saving YAML config: {str(e)}")
    
    return return_code
import copy

def main():
    parser = argparse.ArgumentParser(description='Run WiFi Sensing Pipeline')
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG_PATH, help='YAML config file')
    args = parser.parse_args()

    config = load_config(args.config_file)
    pipeline = config.get('pipeline')
    
    if 'training_dir' in config:
        os.environ['WIFI_DATA_DIR'] = config['training_dir']
    
    available_models = config.get('available_models', ['mlp'])
    results = {}

    for model in available_models:
        print(f"\n{'='*60}\nRunning Model: {model}\n{'='*60}")
        
        # Use deepcopy to keep model runs isolated
        model_config = copy.deepcopy(config)
        model_config['model'] = model
        
        start = time.time()
        if pipeline == 'multitask':
            p_config = get_multitask_config(model_config)
            ret = run_multitask_direct(p_config)
        else:
            p_config = get_supervised_config(model_config)
            ret = run_supervised_direct(p_config)
        end = time.time()

        results[model] = {
            'status': 'SUCCESS' if ret == 0 else 'FAILED',
            'time': (end - start) / 60
        }

    # Summary Report
    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    for mod, res in results.items():
        print(f"Model: {mod:12} | Status: {res['status']:8} | Time: {res['time']:.2f} min")
    
    return 0 if all(r['status'] == 'SUCCESS' for r in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())

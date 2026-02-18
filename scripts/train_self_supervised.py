import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import importlib
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from load.dataloader import get_loaders
from pretext_tasks.base import PretextTask

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_task_class(task_name):
    """
    Dynamically load the task class from pretext_tasks module.
    Assumes the file is named {task_name}.py and class is named {TaskName}.
    """
    module_name = f"pretext_tasks.{task_name.lower()}"
    try:
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, PretextTask) and attr is not PretextTask:
                print(f"Loaded {attr_name} from {module_name}")
                return attr
        raise ImportError(f"No PretextTask subclass found in {module_name}")
    except ImportError as e:
        print(f"Error loading task module {module_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="CSI Self-Supervised Learning Framework")
    
    # Task selection
    parser.add_argument("--method", type=str, required=True, help="Name of the pretext task (e.g., vqcpc, masked_autoencoder)")
    
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
    parser.add_argument("--save_dir", type=str, default="./results_ssl")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Model specific args (generic, tasks can use what they need)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--feature_size", type=int, default=232)
    parser.add_argument("--win_len", type=int, default=500)
    
    # Allow extra args to be passed to the task config
    args, unknown = parser.parse_known_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    save_path = os.path.join(args.save_dir, args.method, args.task)
    os.makedirs(save_path, exist_ok=True)
    
    # Load Data
    print(f"Loading data for task {args.task}...")
    datasets = get_loaders(
        root=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_key=args.data_key,
        file_format=args.file_format,
        pin_memory=(get_device() != "cpu" and not args.no_pin_memory),
    )
    
    if isinstance(datasets, dict) and "loaders" in datasets:
        loaders = datasets["loaders"]
    else:
        loaders = datasets

    train_loader = loaders["train"]
    val_loader = loaders.get("val", None)
    
    # Initialize Task
    print(f"Initializing pretext task: {args.method}")
    TaskClass = load_task_class(args.method)
    
    # Config wrapper
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    config = Config(**vars(args))
    
    model = TaskClass(config, device)
    model.to(device)
    
    # Optimizers
    opts_and_scheds = model.configure_optimizers()
    
    if isinstance(opts_and_scheds, tuple):
        optimizers, schedulers = opts_and_scheds
    else:
        optimizers = opts_and_scheds
        schedulers = []
    
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
    if not isinstance(schedulers, list):
        schedulers = [schedulers]
        
    start_epoch = 0
    
    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"Checkpoint {args.resume} not found!")
    
    writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))
    
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device helper
            def to_device(b):
                if isinstance(b, torch.Tensor):
                    return b.to(device)
                elif isinstance(b, (list, tuple)):
                    return [to_device(x) for x in b]
                elif isinstance(b, dict):
                    return {k: to_device(v) for k, v in b.items()}
                return b
            
            batch = to_device(batch)
            
            step_metrics = {}
            
            # Optimization Loop
            for opt_idx, optimizer in enumerate(optimizers):
                optimizer.zero_grad()
                
                loss_output = model.training_step(batch, batch_idx, optimizer_idx=opt_idx)
                
                if isinstance(loss_output, dict):
                    loss = loss_output['loss']
                    # Log other metrics
                    for k, v in loss_output.items():
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        step_metrics[f"train_{k}"] = val
                else:
                    loss = loss_output
                    step_metrics[f"train_loss_{opt_idx}"] = loss.item()
                
                loss.backward()
                optimizer.step()
            
            # Logging
            if batch_idx % args.log_interval == 0:
                for k, v in step_metrics.items():
                    writer.add_scalar(k, v, epoch * len(train_loader) + batch_idx)
            
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items() if isinstance(v, float)})
        
        # Schedulers
        for scheduler in schedulers:
            scheduler.step()
            
        # End of Epoch Hook
        model.on_epoch_end(epoch)
        
        # Save Checkpoint
        save_file = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt")
        save_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        torch.save(save_state, save_file)
        
        # Always save best/latest encoder
        # This is the artifact we care about for transfer learning
        encoder_path = os.path.join(save_path, "encoder_latest.pt")
        try:
            torch.save(model.get_encoder().state_dict(), encoder_path)
        except Exception as e:
            print(f"Error saving encoder: {e}")
        
    print(f"Training complete. Encoder saved to {encoder_path}")
    writer.close()

if __name__ == "__main__":
    main()

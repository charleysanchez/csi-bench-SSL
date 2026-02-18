import os
from torch.utils.data import DataLoader
from .dataset import CSIDataset
import torch
import json

def get_loaders(
        root="../data/",
        task="BreathingDetection",
        batch_size=32,
        transform=None,
        target_transform=None,
        file_format="h5",
        data_column="file_path",
        label_column="label",
        data_key="CSI_amps",
        num_workers=4,
        shuffle_train=True,
        train_split="train_id",
        val_split="val_id",
        test_splits="all",
        label_mapper=None,
        distributed=False,
        collate_fn=None,
        pin_memory=True,
        train_filter=None,
        val_filter=None,
        debug=False,
):
    """
    Load benchmark dataset for supervised learning.
    
    Args:
        root: Root directory of the dataset.
        task: Name of the task (e.g., 'motion_source_recognition')
        batch_size: Batch size for DataLoader.
        transform: Optional transform to apply to data.
        target_transform: Optional transform to apply to labels.
        file_format: File format for data files ("h5", "mat", or "npy").
        data_column: Column in metadata that contains file paths.
        label_column: Column in metadata that contains labels.
        data_key: Key in h5 file for CSI data.
        num_workers: Number of worker processes for DataLoader.
        shuffle_train: Whether to shuffle training data.
        train_split: Name of training split.
        val_split: Name of validation split.
        test_splits: List of test split names or "all" to load all test_*.json splits.
        distributed: Whether to configure data loaders for distributed training
        collate_fn: Custom collate function for DataLoader
        pin_memory: Whether to use pinned memory (set to False for MPS device)
        train_filter: Dictionary to filter training data (e.g. {'user': ['F01']})
        val_filter: Dictionary to filter validation data
        debug: Whether to enable debug mode.
        
    Returns:
        Dictionary with data loaders and number of classes.
    """
    # set default test splits if not provided
    if test_splits is None:
        test_splits = ["test_id"]
    elif isinstance(test_splits, str):
        if test_splits.lower() == "all":
            test_splits = "all"
        else:
            test_splits = [test_splits]

    # define relevant directory paths
    task_dir = os.path.join(root, task)
    metadata_dir = os.path.join(task_dir, "metadata")
    splits_dir = os.path.join(task_dir, "splits")

    # now find all test_* json files if test == all
    if test_splits == "all":
        test_splits = []
        if os.path.exists(splits_dir) and os.path.isdir(splits_dir):
            for file in os.listdir(splits_dir):
                if file.startswith("test_") and file.endswith(".json"):
                    split_name, _ = os.path.splitext(file)
                    test_splits.append(split_name)
        if not test_splits:
            test_splits = ["test_id"]
            print("No test splits found, defaulting to 'test_id'.")

    all_splits = [train_split, val_split] + test_splits

    metadata_path = os.path.join(metadata_dir, "metadata.csv")
    map_path = os.path.join(metadata_dir, "label_mapping.json")

    with open(map_path, "r") as f:
        label_mapper = json.load(f)

    datasets = {}
    for split_name in all_splits:
        try:
            print(f"Using provided task directory {task_dir}")
            dataset = CSIDataset(
                root=root,
                task=task,
                split=split_name,
                transform=transform,
                target_transform=target_transform,
                file_format=file_format,
                data_column=data_column,
                label_column=label_column,
                data_key=data_key,
                label_mapper=label_mapper,
                task_dir=task_dir,
                filter_dict=train_filter if split_name == train_split else (val_filter if split_name == val_split else None),
                debug=debug
            )
            datasets[split_name] = dataset
            print(f"Loaded {len(dataset)} samples for {task} - {split_name}")
        except Exception as e:
            print(f"Error loading split '{split_name}': {str(e)}")
            datasets[split_name] = None

    # create dataloaders
    loaders = {}

    # check if distributed is required
    if distributed and torch.distributed.is_initialized():
        print(f"Setting up distributed samplers for {task}")

        # training loader with DistributedSampler
        train_sampler = torch.utils.data.DistributedSampler(
            dataset=datasets[train_split],
            shuffle=shuffle_train
        )

        loaders['train'] = DataLoader(
            dataset=datasets[train_split],
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        # Validation loader
        val_sampler = torch.utils.data.DistributedSampler(
            dataset=datasets[val_split],
            shuffle=False
        )

        loaders['val'] = DataLoader(
            dataset=datasets[val_split],
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        # test loaders
        for test_split in test_splits:
            # special case for backward compatibility
            if test_split == "test_id":
                loader_name = "test"
            else:
                loader_name = f"test_{test_split}" if not test_split.startswith('test_') else test_split
            
            test_sampler = torch.utils.data.DistributedSampler(
                dataset=datasets[test_split],
                shuffle=False
            )

            loaders[loader_name] = DataLoader(
                dataset=datasets[test_split],
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
    else:
        # regular non distributed data loaders
        loaders['train'] = DataLoader(
            dataset=datasets[train_split],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        loaders['val'] = DataLoader(
            dataset=datasets[val_split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

        for test_split in test_splits:
            # special case for backward compatibility
            if test_split == "test_id":
                loader_name = "test"
            else:
                loader_name = f"test_{test_split}" if not test_split.startswith('test_') else test_split
            

            loaders[loader_name] = DataLoader(
                dataset=datasets[test_split],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )

        num_classes = label_mapper["num_classes"]

        # Return dictionary with additional distributed information if needed
        result = {
            'loaders': loaders,
            'datasets': datasets,
            'num_classes': num_classes,
            'label_mapper': label_mapper
        }
        
        # Include samplers in the result if distributed
        if distributed and torch.distributed.is_initialized():
            samplers = {
                'train': train_sampler,
                'val': val_sampler
            }
            # Add test samplers
            for test_split in test_splits:
                if test_split == 'test_id':
                    loader_name = 'test'
                else:
                    loader_name = f'test_{test_split}' if not test_split.startswith('test_') else test_split
                samplers[loader_name] = loaders[loader_name].sampler
            
            result['samplers'] = samplers
            result['is_distributed'] = True
        else:
            result['is_distributed'] = False
        
        return result

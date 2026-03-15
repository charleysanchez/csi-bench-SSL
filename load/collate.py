"""
Shared collate utilities for CSI dataloaders.
Filters out None samples and returns consistent empty-batch tensors.
"""
import torch


class CollateSkipNone:
    """
    Collate that filters None samples. Used by supervised, multitask, eval, and pretrain.
    Use for_supervised=True for train_supervised, evaluate_ood, train_multitask_adapter
    (empty batch returns labels as long tensor). Use for_supervised=False for pretrain
    (empty batch returns second tensor as float; non-empty may add channel dim to 3D inputs).
    When the dataset returns 5 elements (inputs, label, user, env, device), for_supervised
    returns all 5 so DANN can use domain labels; non-DANN code can use batch[0], batch[1].
    """

    def __init__(self, win_len, feature_size, for_supervised=True):
        self.win_len = win_len
        self.feature_size = feature_size
        self.for_supervised = for_supervised

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            empty_x = torch.zeros(0, 1, self.win_len, self.feature_size)
            empty_y = torch.zeros(0, dtype=torch.long) if self.for_supervised else torch.zeros(0)
            if self.for_supervised:
                empty_domain = torch.zeros(0, dtype=torch.long)
                return empty_x, empty_y, empty_domain, empty_domain, empty_domain
            return empty_x, empty_y

        collated = torch.utils.data.dataloader.default_collate(batch)

        if self.for_supervised:
            # Pass through all elements so DANN gets (inputs, labels, user, env, device)
            if len(collated) >= 5:
                return collated[0], collated[1], collated[2], collated[3], collated[4]
            return collated[0], collated[1] if len(collated) > 1 else collated[0]

        # Pretrain: optional channel dim and preserve (inputs,) or (inputs, labels)
        if isinstance(collated, (list, tuple)):
            inputs = collated[0]
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            return (inputs,) + tuple(collated[1:])
        inputs = collated
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        return inputs

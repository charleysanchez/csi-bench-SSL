import torch

def reduce_sum(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor

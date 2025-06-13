import os

import torch
import torch.distributed as dist

from torch.utils.data import Subset, DataLoader

def ddp_setup(rank, world_size):
    # Default master address and port for single-node setups
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the GPU to use
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def broadcast_scalar(value, src, device):
    tensor = torch.tensor([value], device=device)
    dist.broadcast(tensor, src=src)
    return tensor.item()

def broadcast_list(data_list, src, device):
    """ WARNING: data_list must contain the correct data type. Otherwise, conversion errors may occurs...
    """
    tensor = torch.tensor(data_list, device=device)
    dist.broadcast(tensor, src=src)
    dist.barrier()
    return tensor.tolist()

def split_dataloader_by_rank(data_loader, rank, shuffle=None, drop_last=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dataset = data_loader.dataset

    if rank == 0:
        num_samples = len(dataset)//world_size + len(dataset)%world_size
        indices = torch.arange(num_samples)
        device_subset = Subset(dataset, indices)
    else:
        num_samples = len(dataset)//world_size
        indices = torch.arange(num_samples)+num_samples*rank
        device_subset = Subset(dataset, indices)

    data_loader = DataLoader(
        device_subset, 
        batch_size=data_loader.batch_size, 
        shuffle=data_loader.shuffle if shuffle is None else shuffle,
        drop_last=data_loader.drop_last if drop_last is None else drop_last
    )

    return data_loader, indices
import torch
import torch.distributed as dist

from utils.logging import ddp_logging

from .al_utils import get_predictions, gather_uncertainties
from dp_al.dp_utils import laplace_mechanism

@torch.no_grad()
def minimum_margin(data_name, model, data_loader, query_size, epsilon, device, ddp):
    rank = dist.get_rank() if ddp else 0

    ddp_logging("Computing predictions...", rank)
    all_preds, true_indices = get_predictions(data_name, model, data_loader, device, ddp)

    ddp_logging("Computing margins...", rank)
    top2 = torch.topk(all_preds, k=2, dim=-1)[0]
    margin = top2[:, 0] - top2[:, 1]
    
    if epsilon > 0:
        ddp_logging("Adding Laplace noise to margins...", rank)
        margin = laplace_mechanism(margin, epsilon, sensitivity=1)

    if ddp:
        margin, true_indices = gather_uncertainties(margin, true_indices, device)

    indices = (torch.argsort(margin, descending=False)[:query_size]).cpu()
    indices = true_indices[indices]

    return indices
import torch
import torch.distributed as dist

from utils.logging import ddp_logging

from .al_utils import get_predictions, gather_uncertainties
from dp_al.dp_utils import laplace_mechanism

@torch.no_grad()
def least_confidence(data_name, model, data_loader, query_size, epsilon, device, multi_label, ddp):
    rank = dist.get_rank() if ddp else 0

    ddp_logging("Computing predictions...", rank)
    all_preds, true_indices = get_predictions(data_name, model, data_loader, device, ddp, logits=multi_label)
    num_classes = all_preds.shape[-1]

    ddp_logging("Computing max. confidence...", rank)
    if multi_label:
        confidence = (torch.sigmoid(all_preds)-0.5).abs().min(dim=-1)[0]
    else:
        confidence = torch.max(all_preds, dim=-1)[0]-(1/num_classes) # make minimum confidence 0
    
    if epsilon > 0:
        ddp_logging("Adding Laplace noise to max. confidence...", rank)
        sensitivity = 1-(1/num_classes) if not multi_label else 0.5
        confidence = laplace_mechanism(confidence, epsilon, sensitivity=sensitivity)

    if ddp:
        confidence, true_indices = gather_uncertainties(confidence, true_indices, device)

    indices = (torch.argsort(confidence, descending=False)[:query_size]).cpu()
    indices = true_indices[indices]

    return indices
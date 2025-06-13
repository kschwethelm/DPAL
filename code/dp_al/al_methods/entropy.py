import torch
import torch.distributed as dist
import numpy as np
import os

from utils.logging import ddp_logging

from .al_utils import get_predictions, gather_uncertainties, normalized_entropy
from dp_al.dp_utils import laplace_mechanism

@torch.no_grad()
def entropy_sampling(al_iter, data_name, model, data_loader, query_size, epsilon, device, multi_label, ddp, save_vals=False, max_entropy=0.8):
    rank = dist.get_rank() if ddp else 0

    ddp_logging("Computing predictions...", rank)
    all_preds, true_indices = get_predictions(data_name, model, data_loader, device, ddp, logits=multi_label)

    ddp_logging("Computing entropies...", rank)
    if multi_label:
        all_preds = torch.sigmoid(all_preds)
    entropies = normalized_entropy(all_preds, multi_label=multi_label)
    entropies_clean = entropies.clone()
    
    if epsilon > 0:
        ddp_logging("Adding Laplace noise to entropies...", rank)
        entropies = entropies.clamp(max=max_entropy)
        entropies = laplace_mechanism(entropies, epsilon, sensitivity=max_entropy)

    if ddp:
        entropies, true_indices = gather_uncertainties(entropies, true_indices, device)
        entropies_clean, _ = gather_uncertainties(entropies_clean, true_indices, device)

    if save_vals:
        np.savetxt(f"entropies_{al_iter}.txt", np.stack([entropies_clean.cpu().numpy(), entropies.cpu().numpy()]).T, fmt='%1.4f')

    pub_indices = (torch.argsort(entropies_clean, descending=True)[:query_size]).cpu() # true top k without noise
    indices = (torch.argsort(entropies, descending=True)[:query_size]).cpu()

    return true_indices[indices]
import torch
import torch.distributed as dist

from utils.logging import ddp_logging

from .al_utils import get_bayesian_predictions, gather_uncertainties, normalized_entropy
from dp_al.dp_utils import laplace_mechanism

@torch.no_grad()
def bald_sampling(data_name, model, data_loader, query_size, epsilon, device, multi_label, ddp, num_models, dropout, max_MI=1.0):
    """ Bayesian Active Learning by Disagreement (BALD) sampling strategy with MC dropout approximation.
    """
    rank = dist.get_rank() if ddp else 0

    ddp_logging("Computing predictions...", rank)
    all_preds, true_indices = get_bayesian_predictions(data_name, model, data_loader, device, ddp, num_models, dropout, logits=multi_label)

    ddp_logging("Computing entropies...", rank)
    if multi_label:
        all_preds = torch.sigmoid(all_preds)
    entropy_avg_pred = normalized_entropy(all_preds.mean(dim=0), multi_label=multi_label)
    entropy_avg = normalized_entropy(all_preds, multi_label=multi_label).mean(dim=0)

    mutual_information = entropy_avg_pred - entropy_avg
    
    if epsilon > 0:
        ddp_logging("Adding Laplace noise to entropies...", rank)
        mutual_information = mutual_information.clamp(min=0, max=max_MI)
        mutual_information = laplace_mechanism(mutual_information, epsilon, sensitivity=max_MI)
        
    if ddp:
        mutual_information, true_indices = gather_uncertainties(mutual_information, true_indices, device)

    indices = (torch.argsort(mutual_information, descending=True)[:query_size]).cpu()
    indices = true_indices[indices]

    return indices
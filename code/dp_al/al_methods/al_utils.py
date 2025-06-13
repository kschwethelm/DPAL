import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.ddp_utils import split_dataloader_by_rank

def normalized_entropy(p, multi_label=False):
    """ Compute the normalized entropy of a probability distribution.

    If multi_label, functions returns the average entropy across classes

    Args:
        p: tensor of shape (..., num_samples, num_classes)
    """
    if multi_label:
        entropy = (-(p*torch.log2(p) + (1 - p)*torch.log2(1 - p))).mean(-1)
    else:
        entropy = (-torch.sum(p * torch.log2(p+1e-10), dim=-1))/torch.log2(torch.tensor(p.shape[-1]))
    return entropy

@torch.no_grad()
def get_predictions(data_name, model, data_loader, device, ddp, logits=False):
    """ Get predictions from the model.

    Args:
        model: model to get predictions from
        data_loader: DataLoader object
        device: device to run the model on
        ddp: whether to use DistributedDataParallel
        logits: whether to return logits or probabilities
    
    Returns:
        all_preds: tensor of shape (num_samples, num_classes)
        local_indices: tensor of shape (num_samples) containing the indices from the dataset (important for DDP)
    """
    rank = dist.get_rank() if ddp else 0
    if ddp:
        data_loader, local_indices = split_dataloader_by_rank(data_loader, rank, shuffle=False, drop_last=False)
    else:
        local_indices = torch.arange(len(data_loader.dataset))

    model.eval()
    model.to(device)
    
    all_preds = []
    for batch in tqdm(data_loader):
        if data_name == "snli":
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            preds = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        else:
            data = batch[0].to(device)
            preds = model(data)
            if data_name == "chexpert":
                preds = preds[:, :5]

        if not logits:
            preds = torch.softmax(preds, dim=-1)
        all_preds.append(preds.detach())

    all_preds = torch.cat(all_preds, dim=0).to(device)

    return all_preds, local_indices

@torch.no_grad()
def get_bayesian_predictions(
    data_name,
    model, 
    data_loader, 
    device, ddp, 
    num_models=20, 
    dropout=0.5, 
    logits=False
):
    """ Get predictions from Bayesian inference using MC dropout approximation.

    Model must contain dropout layers.

    Args:
        model: model to get predictions from
        data_loader: DataLoader object
        device: device to run the model on
        ddp: whether to use DistributedDataParallel
        num_models: number of models to sample
        dropout: dropout rate
        logits: whether to return logits or probabilities
    
    Returns:
        all_preds_models: list of tensors of shape (num_models, num_samples, num_classes)
        local_indices: tensor of shape (num_samples) containing the indices from the dataset (important for DDP)
    """
    assert dropout > 0 and dropout < 1, "Dropout must be in (0, 1)"

    rank = dist.get_rank() if ddp else 0
    if ddp:
        data_loader, local_indices = split_dataloader_by_rank(data_loader, rank, shuffle=False, drop_last=False)
    else:
        local_indices = torch.arange(len(data_loader.dataset))

    model.eval()
    model.to(device)
    
    all_preds_models = []
    for i in range(num_models):
        all_preds = []
        for batch in tqdm(data_loader):
            if data_name == "snli":
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                token_type_ids = batch[2].to(device)
                preds = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, dropout=dropout)[0]
            else:
                data = batch[0].to(device)
                preds = model(data, dropout=dropout)
                if data_name == "chexpert":
                    preds = preds[:, :5]
            if not logits:
                preds = torch.softmax(preds, dim=-1)
            all_preds.append(preds.detach().cpu())

        all_preds_models.append(torch.cat(all_preds, dim=0))

    all_preds_models = torch.stack(all_preds_models, dim=0)

    return all_preds_models, local_indices

@torch.no_grad()
def gather_uncertainties(uncertainties, local_indices, device):
    """ Gather uncertainties and indices from all DDP processes.
    """
    world_size = dist.get_world_size()

    gathered_uncertainties = [torch.empty_like(uncertainties) for _ in range(world_size)]
    dist.all_gather(gathered_uncertainties, uncertainties)
    uncertainties = torch.cat(gathered_uncertainties, dim=0)

    if not type(local_indices) is torch.Tensor:
        local_indices = torch.tensor(local_indices, dtype=torch.long)

    local_indices = local_indices.to(device)

    gathered_indices = [torch.empty_like(local_indices) for _ in range(world_size)]
    dist.all_gather(gathered_indices, local_indices)
    true_indices = torch.cat(gathered_indices, dim=0)

    return uncertainties, true_indices.cpu()

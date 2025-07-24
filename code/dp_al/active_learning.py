from utils.logging import ddp_logging

import torch
from torch.utils.data import Subset, DataLoader
import torch.distributed as dist

from .al_methods import (
    entropy_sampling,
    least_confidence,
    minimum_margin,
    bald_sampling,
)


def select_samples(
    al_iter,
    data_name,
    model,
    unlabeled_loader,
    query_strategy,
    query_size,
    epsilon,
    device,
    multi_label,
    ddp,
):
    """Queries the most informative samples from the unlabeled dataset."""
    assert query_size <= len(unlabeled_loader.dataset), (
        "Query size is larger than the number of unlabeled samples."
    )

    rank = dist.get_rank() if ddp else 0

    if query_strategy == "random":  # or epsilon == 0:
        ddp_logging("Random sampling...", rank)

        num_unlabeled = len(unlabeled_loader.dataset)
        indices = torch.randperm(num_unlabeled)[:query_size]

        if ddp:
            indices = indices.to(device)
            dist.broadcast(indices, 0)
            indices = indices.cpu()
    elif query_strategy == "entropy":
        ddp_logging("Entropy sampling...", rank)
        if multi_label:
            max_entropy = 0.8
        else:
            max_entropy = 0.8
        indices = entropy_sampling(
            al_iter,
            data_name,
            model,
            unlabeled_loader,
            query_size,
            epsilon,
            device,
            multi_label,
            ddp,
            max_entropy=max_entropy,
        )
    elif query_strategy == "confidence":
        ddp_logging("Confidence sampling...", rank)
        indices = least_confidence(
            data_name,
            model,
            unlabeled_loader,
            query_size,
            epsilon,
            device,
            multi_label,
            ddp,
        )
    elif query_strategy == "margin":
        assert not multi_label, (
            "Margin sampling does not work in multi_label classification."
        )
        ddp_logging("Margin sampling...", rank)
        indices = minimum_margin(
            data_name, model, unlabeled_loader, query_size, epsilon, device, ddp
        )
    elif query_strategy == "bald":
        indices = bald_sampling(
            data_name,
            model,
            unlabeled_loader,
            query_size,
            epsilon,
            device,
            multi_label,
            ddp,
            num_models=20,
            dropout=0.3,
        )
    else:
        raise "Query strategy not implemented."

    return indices.cpu()


class AL_datahandler:
    def __init__(
        self,
        dataset,
        data_name,
        initial_labeled_size,
        labeling_budget,
        query_sizes,
        random_sampling,
        multi_label,
        ddp=False,
    ):
        assert random_sampling or (
            sum(query_sizes) + initial_labeled_size == labeling_budget
        ), "Query sizes not correct. Num labeled will be < labeling budget."
        self.dataset = dataset
        self.data_name = data_name
        self.labeling_budget = labeling_budget
        self.query_sizes = query_sizes

        if random_sampling:
            self.initial_labeled_size = labeling_budget
        else:
            self.initial_labeled_size = initial_labeled_size

        self.al_iterations = len(self.query_sizes) if not random_sampling else 0

        # split dataset
        self.data_indices = torch.randperm(len(dataset))
        self.labeled_indices = self.data_indices[:initial_labeled_size]
        self.unlabeled_indices = self.data_indices[initial_labeled_size:]
        self.new_indices = None

        self.labeled_dataset = None
        self.unlabeled_dataset = None

        self.multi_label = multi_label
        self.ddp = ddp

        self.create_subsets()

    def create_subsets(self):
        self.labeled_dataset = Subset(self.dataset, self.labeled_indices)
        self.unlabeled_dataset = Subset(self.dataset, self.unlabeled_indices)

    def get_labeled_dataset(self):
        return self.labeled_dataset

    def get_unlabeled_dataset(self):
        return self.unlabeled_dataset

    def get_num_labeled_data(self):
        return len(self.labeled_dataset)

    def get_num_unlabeled_data(self):
        return len(self.unlabeled_dataset)

    def get_labeled_loader(self, batch_size):
        labeled_loader = DataLoader(
            self.labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return labeled_loader

    def get_unlabeled_loader(self, batch_size):
        unlabeled_loader = DataLoader(
            self.unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        return unlabeled_loader

    def get_new_labeled_loader(self, batch_size):
        new_labeled_dataset = Subset(self.dataset, self.new_indices)
        new_labeled_loader = DataLoader(
            new_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return new_labeled_loader

    def get_dataloaders(self, batch_size):
        return self.get_labeled_loader(batch_size), self.get_unlabeled_loader(
            batch_size
        )

    def label_data(self, indices):
        assert len(indices) <= len(self.unlabeled_indices), "Too many samples queried."
        assert len(self.labeled_indices) + len(indices) <= self.labeling_budget, (
            "Labeling budget exceeded."
        )

        self.new_indices = self.unlabeled_indices[indices]
        self.labeled_indices = torch.cat([self.labeled_indices, self.new_indices])

        mask = torch.ones(len(self.unlabeled_indices), dtype=torch.bool)
        mask[indices] = False
        self.unlabeled_indices = self.unlabeled_indices[mask]

        self.candidate_indices = None

        self.create_subsets()

    def al_iteration(
        self, al_iter, model, query_strategy, query_size, batch_size, epsilon, device
    ):
        unlabeled_loader = self.get_unlabeled_loader(batch_size)

        indices = select_samples(
            al_iter,
            data_name=self.data_name,
            model=model,
            unlabeled_loader=unlabeled_loader,
            query_strategy=query_strategy,
            query_size=query_size,
            epsilon=epsilon,
            device=device,
            multi_label=self.multi_label,
            ddp=self.ddp,
        )

        self.label_data(indices)

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from datasets import CheXpert, get_slni_dataset
from medmnist import BloodMNIST, OCTMNIST

from dp_al.active_learning import AL_datahandler

from models import ResNet9, EquivariantResNet9, NFNetF0
from models.nfnets.optim import SGD_AGC
from transformers import BertTokenizer, BertForSequenceClassification

from dp_al.privacy_engine import DPSGD_PrivacyEngine
from dp_al.dp_utils import get_noise_multiplier
from dp_al.modes import step_amplification_iterations, noise_reduction_iterations

from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch import distributed as dist
from .ddp_utils import broadcast_list

from utils.logging import ddp_logging


def get_dataset(args, ddp=False):
    """ " Selects an available dataset and returns PyTorch dataloaders for training and testing."""

    class SqueezeTarget:
        def __call__(self, target):
            return target.item() if hasattr(target, 'item') else target
    
    if args.dataset == "cifar10":
        num_classes = 10
        num_channels = 3
        multi_label = False
        img_size = 32

        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )

        train_dataset = CIFAR10(
            root=args.data_root, train=True, download=True, transform=transforms
        )
        test_dataset = CIFAR10(
            root=args.data_root, train=False, download=True, transform=test_transform
        )

    elif args.dataset == "oct":
        num_classes = 4
        num_channels = 1
        multi_label = False
        img_size = 128

        transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
        )
        test_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
        )

        train_dataset = OCTMNIST(
            root=args.data_root,
            split="train",
            download=True,
            transform=transforms,
            target_transform=SqueezeTarget(),
            size=img_size,
        )
        test_dataset = OCTMNIST(
            root=args.data_root,
            split="test",
            download=True,
            transform=test_transform,
            target_transform=SqueezeTarget(),
            size=img_size,
        )

    elif args.dataset == "blood":
        num_classes = 8
        num_channels = 3
        multi_label = False
        img_size = 128

        transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )
        test_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

        train_dataset = BloodMNIST(
            root=args.data_root,
            split="train",
            download=True,
            transform=transforms,
            target_transform=SqueezeTarget(),
            size=img_size,
        )
        test_dataset = BloodMNIST(
            root=args.data_root,
            split="test",
            download=True,
            transform=test_transform,
            target_transform=SqueezeTarget(),
            size=img_size,
        )

    elif args.dataset == "chexpert":
        """ Following the standard methodology, we train on all classes and evaluate on a subset of 5 classes: 
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema' and 'Pleural Effusion
        '"""
        num_classes = 14
        num_channels = 3  # Pretrained model expects 3 channels
        multi_label = True
        img_size, test_size = 192, 192

        transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )
        test_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

        train_dataset = CheXpert(
            root=os.path.join(args.data_root, "chexpert"),
            size=img_size,
            split="train",
            transform=transforms,
            memmap_path=os.path.join(
                args.data_root, f"chexpert/memmap_{img_size}_train.memmap"
            ),
        )
        test_dataset = CheXpert(
            root=os.path.join(args.data_root, "chexpert"),
            size=img_size,
            split="test",
            transform=test_transform,
            memmap_path=os.path.join(
                args.data_root, f"chexpert/memmap_{test_size}_test.memmap"
            ),
        )

    elif args.dataset == "snli":
        num_classes = 3
        num_channels = None
        multi_label = False
        img_size = None

        # Preprocess the dataset
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=False
        )

        train_dataset = get_slni_dataset(args.data_root, tokenizer, "train")
        test_dataset = get_slni_dataset(args.data_root, tokenizer, "test")
    else:
        raise "Dataset not implemented."

    al_datahandler = AL_datahandler(
        train_dataset,
        args.dataset,
        args.initial_dataset_size,
        args.labeling_budget,
        args.query_sizes,
        multi_label=multi_label,
        random_sampling=not args.al,
        ddp=ddp,
    )

    sampler = DistributedSampler(test_dataset) if ddp else None
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.max_physical_batch_size,
        shuffle=False,
        sampler=sampler,
    )

    return al_datahandler, val_loader, num_channels, num_classes, multi_label, img_size


def get_model(args, num_channels, num_classes, img_size, device, ddp=False):
    """Selects and sets up an available model and returns it."""
    if args.model == "resnet9":
        model = ResNet9(
            in_channels=num_channels, num_classes=num_classes, norm_layer="group"
        )
    elif args.model == "eq_resnet9":
        model = EquivariantResNet9(
            input_channels=num_channels, num_classes=num_classes, spatial_dims=img_size
        )
    elif args.model == "nfnet_f0":
        model = NFNetF0(num_classes=num_classes)
    elif args.model == "bert" and args.dataset == "snli":
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=num_classes
        )
        # Reduce the number of trainable parameters
        trainable_layers = [
            model.bert.encoder.layer[-1],
            model.bert.pooler,
            model.classifier,
        ]
        for p in model.parameters():
            p.requires_grad = False
        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        raise "Model not implemented."

    if ddp:
        model.to(device)
        model = DPDDP(model)

    model.train()
    return model


def get_optimizer(args, model, ddp=False):
    """Selects and sets up an available optimizer and returns it."""

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd_agc":
        optimizer = SGD_AGC(
            model.named_parameters(), lr=args.learning_rate, clipping=0.1
        )
        # Find desired parameters and exclude them from clipping
        for group in optimizer.param_groups:
            name = group["name"]
            exclude = (
                model.module.exclude_from_clipping(name)
                if ddp
                else model.exclude_from_clipping(name)
            )
            if exclude:
                group["clipping"] = None

    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise "Optimizer not implemented."

    return optimizer


def get_loss(multi_label):
    """Selects and sets up an available loss function and returns it."""

    if multi_label:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return criterion


def make_private(model, optimizer, train_loader, al_iterations, args, device, ddp):
    """Makes the model and optimizer private using Opacus."""
    rank = 0 if not ddp else dist.get_rank()

    if rank == 0:
        add_iter = 0 if train_loader.drop_last else 1

        dataset_sizes = [
            args.initial_dataset_size + sum(args.query_sizes[:i])
            for i in range(al_iterations + 1)
        ]
        steps_per_epoch = [
            (dataset_sizes[i] // args.batch_size) + add_iter
            for i in range(al_iterations + 1)
        ]
        if steps_per_epoch[0] == 0:
            steps_per_epoch[0] = 1

        sample_rates = [1 / steps_per_epoch[i] for i in range(al_iterations + 1)]
        if args.epochs_al is None:
            steps = [steps_per_epoch[i] * args.epochs for i in range(al_iterations + 1)]
        else:
            steps = [
                steps_per_epoch[i] * args.epochs_al[i] for i in range(al_iterations + 1)
            ]

        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            step_schedule=steps,
            sample_rate_schedule=sample_rates,
        )

        ddp_logging(
            f"Training with eps = {args.epsilon}, delta = {args.delta}, sigma = {noise_multiplier}, clip = {args.max_sample_grad_norm}",
            rank,
        )

        if args.dp_al_mode == "naive":
            steps_new = steps
            sample_rates_old = sample_rates
            sample_rates_new = sample_rates
            noise_multipliers = [noise_multiplier] * (al_iterations + 1)
        elif "step_ampl_prob" in args.dp_al_mode:
            ddp_logging("Running step amplification...", rank)
            steps_new, sample_rates_old, sample_rates_new, noise_multipliers = (
                step_amplification_iterations(
                    noise_multiplier=noise_multiplier,
                    delta=args.delta,
                    steps=steps,
                    sample_rates=sample_rates,
                    al_iterations=al_iterations,
                    epsilon_al=args.epsilon_al,
                    query_sizes=args.query_sizes,
                    initial_dataset_size=args.initial_dataset_size,
                    batch_size=args.batch_size,
                    sigma_corr=args.dp_al_mode == "step_ampl_prob_noise",
                )
            )
        elif args.dp_al_mode == "noise_red_prob":
            ddp_logging("Running noise reduction...", rank)
            sample_rates_old, sample_rates_new, noise_multipliers = (
                noise_reduction_iterations(
                    noise_multiplier=noise_multiplier,
                    delta=args.delta,
                    steps=steps,
                    sample_rates=sample_rates,
                    al_iterations=al_iterations,
                    epsilon_al=args.epsilon_al,
                    query_sizes=args.query_sizes,
                    initial_dataset_size=args.initial_dataset_size,
                    batch_size=args.batch_size,
                )
            )
            steps_new = steps
        else:
            raise ValueError(f"Unknown AL DP mode {args.dp_al_mode}.")

    else:
        steps = [0] * (al_iterations + 1)
        steps_new = [0] * (al_iterations + 1)
        sample_rates_old = [0.0] * (al_iterations + 1)
        sample_rates_new = [0.0] * (al_iterations + 1)
        noise_multipliers = [0.0] * (al_iterations + 1)

    if ddp:
        steps = broadcast_list(steps, src=0, device=device)
        steps_new = broadcast_list(steps_new, src=0, device=device)
        sample_rates_old = broadcast_list(sample_rates_old, src=0, device=device)
        sample_rates_new = broadcast_list(sample_rates_new, src=0, device=device)
        noise_multipliers = broadcast_list(noise_multipliers, src=0, device=device)

    ddp_logging("Creating privacy engine...", rank)
    privacy_engine = DPSGD_PrivacyEngine(
        noise_multiplier=noise_multipliers[0],
        max_grad_norm=args.max_sample_grad_norm,
        delta=args.delta,
    )

    model, optimizer, train_loader = privacy_engine.make_private(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        ddp=ddp,
        use_functorch=args.use_functorch,
    )

    ddp_logging(
        f"Steps before: {steps}, Steps amplified: {steps_new}, Sample rates old: {sample_rates_old}, Sample rates new: {sample_rates_new}",
        rank,
    )
    ddp_logging(f"New noise multipliers: {noise_multipliers}", rank)

    return (
        model,
        optimizer,
        train_loader,
        privacy_engine,
        steps_new,
        [sample_rates_old, sample_rates_new],
        noise_multipliers,
    )

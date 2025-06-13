import torch

import argparse

import random
import numpy as np
import os

import torch.multiprocessing as mp

from opacus.data_loader import DPDataLoader

from utils.logging import setup_logging, ddp_logging
from utils.arg_utils import str2bool
from utils.initialize import (
    get_dataset,
    get_model,
    get_optimizer,
    get_loss,
    make_private
)
from utils.train_loop import Trainer
from utils.ddp_utils import ddp_setup, ddp_cleanup

def getArguments():
    """ Parses command-line options. """
    parser = argparse.ArgumentParser(description='Active learning classification training with Differential Privacy', add_help=True)

    parser.add_argument('--num_gpus', "-gpus", default=None, type=int,
                        help="Number of gpus for distributed training. If None, use all available GPUs.")

    # Output settings
    parser.add_argument('--exp_name', "-n", default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default="output", type=str,
                        help="Path for output files (relative to working directory). " +
                            "Note: args.exp_name subfolder will be created")

    #    Training parameters
    parser.add_argument('--seed', "-s", default=1, type=int,
                        help="Set seed for deterministic training.")
    parser.add_argument('--epochs', "-e", default=5, type=int,
                        help="Number of training epochs for each AL iteration.")
    parser.add_argument('--batch_size', "-bs", default=4096, type=int,
                        help="(Expected) training batch size.")
    parser.add_argument('--max_physical_batch_size', "-pbs", default=4096, type=int,
                        help="Maximum physical batch size for DP-SGD.")
    parser.add_argument('--learning_rate', "-lr", default=1e-3, type=float,
                        help="Training learning rate.")
    parser.add_argument('--optimizer', "-opt", default='nadam', type=str,
                        choices=["sgd", "sgd_agc", "adam", "nadam", "adamW"],
                        help="Select an optimizer. sgd + dp = DP-SGD, etc. AGC = Adaptive Gradient Clipping")
    parser.add_argument('--dataset', "-d", default='cifar10', type=str,
                        choices=["cifar10", "blood", "oct", "mimic", "chexpert", "snli"],
                        help="Select a dataset.")
    parser.add_argument('--data_root', default="data", type=str,
                        help="Path to the dataset root directory.")
    parser.add_argument('--model', "-m", default='resnet9', type=str,
                        choices=["resnet9", "eq_resnet9", "nfnet_f0", "bert"],
                        help="Select a model architecture.")
    
    # Active learning (AL) parameters
    parser.add_argument('--al', '-al', type=str2bool, default=True, 
                        help='Enable active learning (AL) process, otherwise random sampling is used.')
    parser.add_argument('--epochs_al', "-e_al", default=None, type=str,
                        help="Seperate number of training epochs for each AL iteration.")
    parser.add_argument("--epsilon_al", "-eps_al", default=2, type=float,
                        help="Target epsilon for DP-SGD")
    parser.add_argument("--dp_al_mode", "-mode", default="step_ampl_prob_noise", type=str,
                        choices=["naive", "step_ampl_prob", "step_ampl_prob_noise", "noise_red_prob"],
                        help="Select mode to use full privacy budget of all samples.")
    parser.add_argument('--al_batch_size', "-al_bs", default=4096, type=int,
                        help="Batch size for active learning.")
    parser.add_argument("--initial_dataset_size", "-init_size", default=10000, type=int,
                        help="Initial size of the labeled dataset")
    parser.add_argument("--query_sizes", "-query_sizes", default='10000,3000,1000,1000', type=str,
                        help="Number of samples to query in each AL iteration (top-k)")
    parser.add_argument("--labeling_budget", "-budget", default=25000, type=int,
                        help="Total labeling budget for the AL process")
    parser.add_argument("--query_strategy", "-query", default="entropy", type=str,
                        choices=["random", "entropy", "confidence", "margin", "bald", "badge"],
                        help="Select the query strategy for the AL process")
    
    # Differential Privacy (DP) parameters
    parser.add_argument("--epsilon", "-eps", default=8, type=float,
                        help="Target epsilon for DP-SGD")
    parser.add_argument("--delta", "-delta", default=1e-5, type=float,
                        help="Privacy delta for DP-SGD")
    parser.add_argument("--max_sample_grad_norm", "-clip", default=1.0, type=float,
                        help="Clip value for per-sample gradients")
    parser.add_argument("--use_functorch", "-func", default=False, type=str2bool,
                        help="If functorch should be used for DP-SGD (necessary for equivariant models), not possible for BERT")
    
    # Logging
    parser.add_argument('--resume_checkpoint', "-cp", default=None, type=str,
                        help="Path to a checkpoint to resume training.")
    parser.add_argument('--save_interval', "-save", default=-1, type=int,
                        help="Save model every n epochs. Negative numbers = no saving.")

    args = parser.parse_args()

    args.query_sizes = [int(q) for q in args.query_sizes.split(",")]

    if args.epochs_al is not None:
        args.epochs_al = [int(e) for e in args.epochs_al.split(",")]
        assert len(args.epochs_al) == len(args.query_sizes)+1, "Number of epochs_al must match number of query_sizes."

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    if not args.al: # Random sampling
        args.initial_dataset_size = args.labeling_budget

    return args

def main(rank, world_size, args):
    ddp = world_size > 1
    if ddp:
        ddp_setup(rank, world_size)

    setup_logging(os.path.join(args.output_dir, "log.log"))
    ddp_logging(f"Available gpus {torch.cuda.device_count()}", rank)
    ddp_logging(f"Running with {world_size} GPUs", rank)
    ddp_logging(f'Running experiment: {args.exp_name}', rank)
    ddp_logging(f'Config: {args}', rank)

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    ddp_logging(f"Using device: {device}", rank, force=True)
    device = torch.device(device)

    ddp_logging("Loading datasets...", rank)
    al_datahandler, val_loader, num_channels, num_classes, multi_label, img_size = get_dataset(args, ddp)
    train_loader = al_datahandler.get_labeled_loader(args.batch_size)

    ddp_logging("Creating model...", rank)
    model = get_model(args, num_channels, num_classes, img_size, device, ddp)
    ddp_logging('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ), rank)

    ddp_logging("Creating optimizer...", rank)
    optimizer = get_optimizer(args, model, ddp)
    criterion = get_loss(multi_label)

    ddp_logging("Computing privacy parameters...", rank)
    model, optimizer, train_loader, privacy_engine, steps, sample_rates, noise_multipliers = make_private(model, optimizer, train_loader, al_datahandler.al_iterations, args, device, ddp)
    ddp_logging(f"Initial noise multiplier = {privacy_engine.noise_multiplier}", rank)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        data_name=args.dataset,
        num_classes=num_classes,
        num_epochs=args.epochs if args.epochs_al is None else args.epochs_al[0],
        output_dir=args.output_dir,
        privacy_engine=privacy_engine,
        multi_label=multi_label,
        max_physical_batch_size=args.max_physical_batch_size,
        save_interval=args.save_interval,
        rank=rank,
        ddp=ddp,
        use_functorch=args.use_functorch
    )

    for al_iter in range(al_datahandler.al_iterations+1):
        if al_iter>0 and args.al:
            ddp_logging(f"AL: Querying new labels...", rank)
            al_datahandler.al_iteration(
                al_iter=al_iter,
                model=trainer.model,
                query_strategy=args.query_strategy,
                query_size=args.query_sizes[al_iter-1],
                batch_size=args.al_batch_size,
                epsilon=args.epsilon_al/al_datahandler.al_iterations,
                device=device
            )
            train_loader = al_datahandler.get_labeled_loader(args.batch_size)
            # The following line creates a list and does not do any addition or multiplication
            sample_rates_per_instance = [sample_rates[0][al_iter]]*(args.initial_dataset_size+sum(args.query_sizes[:(al_iter-1)])) + [sample_rates[1][al_iter]]*args.query_sizes[al_iter-1]
            trainer.train_loader = DPDataLoader.from_data_loader(train_loader, steps=steps[al_iter], custom_sample_rate=sample_rates_per_instance, distributed=ddp)
            trainer.optimizer.noise_multiplier = noise_multipliers[al_iter]
            trainer.privacy_engine.noise_multiplier = noise_multipliers[al_iter]
            trainer.privacy_engine.add_group()
            trainer.num_epochs = 1
            ddp_logging(f"AL iter ({al_iter}/{al_datahandler.al_iterations}): Labeled {args.query_sizes[al_iter-1]} samples with the highest entropy.", rank)

        trainer.train()

    ddp_logging("Training finished!", rank)

    if ddp:
        ddp_cleanup()


# ----------------------------------
if __name__ == '__main__':
    args = getArguments()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print("Creating missing output directory...")
            os.mkdir(args.output_dir)

    if args.num_gpus is not None:
        world_size = args.num_gpus
    else:
        world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.spawn(main,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        main(rank=0, world_size=1, args=args)
import torch
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
import numpy as np

from opacus.accountants import create_accountant
from typing import Union

def laplace_mechanism(x, epsilon, sensitivity):
    """ Add Laplace noise to input tensor x.
    """
    return x + torch.tensor(np.random.laplace(0, sensitivity/epsilon, x.shape), device=x.device)

def get_epsilon(
    noise_multiplier: Union[float, list],
    delta: float,
    step_schedule: list,
    sample_rate_schedule: list, 
    accountant: str="rdp",
):
    """ Compute epsilon for a given noise multiplier and delta, using specified accountant.
    """
    assert len(step_schedule) == len(sample_rate_schedule), "step_schedule and sample_rate_schedule must have the same length"
    if not isinstance(noise_multiplier, list):
        noise_multiplier = [noise_multiplier]*len(step_schedule)

    acc = create_accountant(accountant)
    for i in range(len(step_schedule)):
        acc.history.append([noise_multiplier[i], sample_rate_schedule[i], step_schedule[i]])
    
    return acc.get_epsilon(delta=delta)
    
def get_noise_multiplier(
    target_epsilon, 
    target_delta, 
    step_schedule, 
    sample_rate_schedule, 
    eps_error=0.01
    ):
    """ Compute the noise multiplier for a given epsilon and delta via binary search. 
    
        Inspired by https://github.com/pytorch/opacus/blob/main/opacus/accountants/utils.py#L23
    """
    assert len(step_schedule) == len(sample_rate_schedule), "step_schedule and sample_rate_schedule must have the same length"

    eps_high = float("inf")
    sigma_low, sigma_high = 0, 10

    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        
        eps_high = get_epsilon(
            noise_multiplier=sigma_high,
            delta=target_delta,
            step_schedule=step_schedule,
            sample_rate_schedule=sample_rate_schedule, 
        )

    while (target_epsilon-eps_high) > eps_error:
        sigma = (sigma_low + sigma_high) / 2

        eps = get_epsilon(
            noise_multiplier=sigma,
            delta=target_delta,
            step_schedule=step_schedule,
            sample_rate_schedule=sample_rate_schedule, 
        )

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high

def get_sample_rate(
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: Union[float, list],
    step_schedule: list,
    sample_rate_schedule: list,
    precision: float=0.001
):
    """ Compute the sample rate for a given epsilon and delta via binary search. 
    
        Inspired by Boenisch et al. (2023). Have it your way: Individualized Privacy Assignment for DP-SGD.
    """
    assert len(step_schedule) == len(sample_rate_schedule)+1, "step_schedule and sample_rate_schedule+1 must have the same length"

    q_low, q_high = 1e-9, 1.

    while q_low / q_high < 1 - precision:
        q = (q_low + q_high) / 2

        eps = get_epsilon(
            noise_multiplier=noise_multiplier,
            delta=target_delta,
            step_schedule=step_schedule,
            sample_rate_schedule=sample_rate_schedule+[q]
        )

        if eps < target_epsilon:
            q_low = q
        else:
            q_high = q

    return q_high

def get_num_steps(
    target_epsilon,
    target_delta,
    noise_multiplier,
    sample_rate, 
    eps_error=0.01
):
    """ Compute number of possible steps for a given epsilon, delta, and noise multiplier via binary search.

        Inspired by https://github.com/pytorch/opacus/blob/main/opacus/accountants/utils.py#L23
    """
    eps_high = 0
    steps_low, steps_high = 0, 100

    while eps_high < target_epsilon:
        steps_high = int(steps_high*2)

        eps_high = get_epsilon(
            noise_multiplier=noise_multiplier,
            delta=target_delta,
            step_schedule=[steps_high],
            sample_rate_schedule=[sample_rate]
        )

    while (target_epsilon-eps_high) > eps_error or eps_high > target_epsilon:
        steps = round((steps_low + steps_high) / 2)

        eps = get_epsilon(
            noise_multiplier=noise_multiplier,
            delta=target_delta,
            step_schedule=[steps],
            sample_rate_schedule=[sample_rate]
        )

        if eps < target_epsilon:
            steps_low = steps
            eps_high = eps
        
            # Check if neighbor is > target_eps (= optimum found)
            eps = get_epsilon(
                noise_multiplier=noise_multiplier,
                delta=target_delta,
                step_schedule=[steps+1],
                sample_rate_schedule=[sample_rate]
            )

            print(f"get_num_steps converged -> eps = {eps}, error = {abs(target_epsilon-eps)}.")
            break

        else:
            steps_high = steps

    return steps_high
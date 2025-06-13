import logging
from .dp_utils import get_epsilon, get_sample_rate

def step_amplification(
    target_epsilons,
    target_delta,
    noise_multipliers,
    step_schedule,
    sample_rate_schedule,
    dataset_sizes,
    batch_size,
    sigma_corr=False,
    batch_size_error=2
):
    """ Compute individual sampling rates for 2 groups with different privacy budgets by increasing number of steps.
    """
    assert len(step_schedule) == len(sample_rate_schedule)+1, "step_schedule and sample_rate_schedule+1 must have the same length"
    assert len(dataset_sizes)==2
    if dataset_sizes[0]+dataset_sizes[1] < 2*batch_size:
        return [1.,1.]

    steps_low, steps_high = step_schedule[-1], step_schedule[-1]*3
    step_schedule = step_schedule[:-1] # remove final step
    noise_multiplier = noise_multipliers[-1]

    q_i = [get_sample_rate(
            target_epsilon=target_epsilons[i],
            target_delta=target_delta,
            noise_multiplier=noise_multipliers,
            step_schedule=step_schedule+[steps_high] if i==0 else [steps_high],
            sample_rate_schedule=sample_rate_schedule if i==0 else [],
            ) for i in range(2)]

    expected_batch_size = q_i[0]*dataset_sizes[0]+q_i[1]*dataset_sizes[1]
    steps = None
    while abs(expected_batch_size-batch_size) > batch_size_error:
        steps_before = steps
        steps = round((steps_high + steps_low) / 2)
        if steps == steps_before:
            if sigma_corr:
                logging.info(f"Convergence in step amplification. Expected batch size: {expected_batch_size}. Fine-tuning batch size using noise multiplier.")
                if expected_batch_size < batch_size:
                    steps = steps-1 # make expected batch size larger, so sigma needs to be reduced
                noise_multiplier, q_i[0], q_i[1] = noise_reduction(
                    target_epsilons=target_epsilons,
                    target_delta=target_delta,
                    noise_multipliers=noise_multipliers,
                    step_schedule=step_schedule+[steps],
                    sample_rate_schedule=sample_rate_schedule, 
                    dataset_sizes=dataset_sizes,
                    batch_size=batch_size,
                    batch_size_error=batch_size_error
                )
            break
        else:
            q_i = [get_sample_rate(
                target_epsilon=target_epsilons[i],
                target_delta=target_delta,
                noise_multiplier=noise_multipliers,
                step_schedule=step_schedule+[steps] if i==0 else [steps],
                sample_rate_schedule=sample_rate_schedule if i==0 else [],
            ) for i in range(2)]

            expected_batch_size = q_i[0]*dataset_sizes[0]+q_i[1]*dataset_sizes[1]
            if expected_batch_size > batch_size:
                steps_low = steps
            else:
                steps_high = steps
    
    if not sigma_corr:
        logging.info(f"Convergence in step amplification. Expected batch size: {expected_batch_size}.")

    return noise_multiplier, steps_high, q_i[0], q_i[1]

def step_amplification_iterations(
    noise_multiplier, 
    delta, 
    steps, 
    sample_rates, 
    al_iterations, 
    epsilon_al, 
    query_sizes, 
    initial_dataset_size,
    batch_size, 
    sigma_corr=False
):
    """ Step amplification for multiple active learning iterations. """
    steps_new = steps.copy()
    sample_rates_old = sample_rates.copy()
    sample_rates_new = sample_rates.copy()
    noise_multipliers = [noise_multiplier]*(al_iterations+1)
    for i in range(1,al_iterations+1):
        epsilon_al_i = (i/al_iterations)*epsilon_al # privacy leakage from al selection query
        # 1. Compute used eps for labeled and eps for next iteration
        target_eps_old = get_epsilon(
            noise_multiplier=noise_multipliers[:i+1],
            delta=delta,
            step_schedule=steps_new[:i+1],
            sample_rate_schedule=sample_rates_old[:i+1]
        )

        target_eps_new = target_eps_old-epsilon_al_i

        # 2. Amplify steps
        noise_multipliers[i], steps_new[i], sample_rates_old[i], sample_rates_new[i] = step_amplification(
            target_epsilons=[target_eps_old, target_eps_new],
            target_delta=delta,
            noise_multipliers=noise_multipliers[:i+1],
            step_schedule=steps_new[:i+1],
            sample_rate_schedule=sample_rates_old[:i],
            dataset_sizes=[(initial_dataset_size+sum(query_sizes[:(i-1)])), query_sizes[i-1]],
            batch_size=batch_size,
            sigma_corr=sigma_corr
        )

    return steps_new, sample_rates_old, sample_rates_new, noise_multipliers


def noise_reduction(
    target_epsilons,
    target_delta,
    noise_multipliers,
    step_schedule,
    sample_rate_schedule,
    dataset_sizes,
    batch_size,
    batch_size_error=2
):
    """ Compute individual sampling rates for 2 groups with different privacy budgets by reducing noise multiplier.

        Inspired by Boenisch et al. (2023). Have it your way: Individualized Privacy Assignment for DP-SGD.
    """
    assert len(step_schedule) == len(sample_rate_schedule)+1, "step_schedule and sample_rate_schedule must have the same length"
    assert len(dataset_sizes)==2
    if dataset_sizes[0]+dataset_sizes[1] < 2*batch_size:
        return [1.,1.]

    sigma_low, sigma_high = noise_multipliers[-1]*0.6, noise_multipliers[-1]
    noise_multipliers = noise_multipliers[:-1] # remove final noise multiplier

    q_i = [get_sample_rate(
            target_epsilon=target_epsilons[i],
            target_delta=target_delta,
            noise_multiplier=noise_multipliers+[sigma_high] if i==0 else [sigma_high],
            step_schedule=step_schedule if i==0 else [step_schedule[-1]],
            sample_rate_schedule=sample_rate_schedule if i==0 else [],
            ) for i in range(2)]

    expected_batch_size = q_i[0]*dataset_sizes[0]+q_i[1]*dataset_sizes[1]

    while abs(expected_batch_size-batch_size) > batch_size_error:
        expected_batch_size_before = expected_batch_size
        sigma = (sigma_high + sigma_low) / 2

        q_i = [get_sample_rate(
            target_epsilon=target_epsilons[i],
            target_delta=target_delta,
            noise_multiplier=noise_multipliers+[sigma] if i==0 else [sigma],
            step_schedule=step_schedule if i==0 else [step_schedule[-1]],
            sample_rate_schedule=sample_rate_schedule if i==0 else [], 
        ) for i in range(2)]

        expected_batch_size = q_i[0]*dataset_sizes[0]+q_i[1]*dataset_sizes[1]
        if expected_batch_size > batch_size:
            sigma_high = sigma
        else:
            sigma_low = sigma

        if round(expected_batch_size, 1) == round(expected_batch_size_before, 1):
            break

    logging.info(f"Convergence in noise reduction. Expected batch size: {expected_batch_size}.")

    return sigma_high, q_i[0], q_i[1]

def noise_reduction_iterations(
    noise_multiplier, 
    delta, 
    steps, 
    sample_rates, 
    al_iterations, 
    epsilon_al, 
    query_sizes, 
    initial_dataset_size, 
    batch_size
):
    """ Noise reduction for multiple active learning iterations. """
    sample_rates_old = sample_rates.copy()
    sample_rates_new = sample_rates.copy()
    noise_multipliers = [noise_multiplier]*(al_iterations+1)
    for i in range(1,al_iterations+1):
        epsilon_al_i = (i/al_iterations)*epsilon_al # privacy leakage from al selection query
        # 1. Compute used eps for labeled and eps for next iteration
        target_eps_old = get_epsilon(
            noise_multiplier=noise_multipliers[:i+1],
            delta=delta,
            step_schedule=steps[:i+1],
            sample_rate_schedule=sample_rates_old[:i+1],
        )

        target_eps_new = target_eps_old-epsilon_al_i

        # 2. Reduce noise
        noise_multipliers[i], sample_rates_old[i], sample_rates_new[i] = noise_reduction(
            target_epsilons=[target_eps_old, target_eps_new],
            target_delta=delta,
            noise_multipliers=noise_multipliers[:i+1],
            step_schedule=steps[:i+1],
            sample_rate_schedule=sample_rates_old[:i],
            dataset_sizes=[(initial_dataset_size+sum(query_sizes[:(i-1)])), query_sizes[i-1]],
            batch_size=batch_size
        )

    return sample_rates_old, sample_rates_new, noise_multipliers
import random
from copy import deepcopy


def copy_model(model):
    return model.copy() if hasattr(model, 'copy') else deepcopy(model)

def sample_batch(dataset, batch_size=4):
    return random.sample(dataset, batch_size)

def compute_advantanges(rewards):
    pass

def grpo_loss():
    pass

def update_policy():
    pass


def train_grpo(
        initial_policy,
        traces,
        reward_function,
        config,
):
    # pi_theta is the updated policy model
    pi_theta = copy_model(initial_policy)

    for _ in range(config["I"]):
        rewards = [reward_function(trace) for trace in traces]
        advantages = compute_advantanges(rewards)

        for _ in range(config["mu"]):
            loss = grpo_loss(rewards, advantages)
            pi_theta = update_policy(loss)

    return pi_theta


if __name__ == "__main__":
    initial_policy = None
    traces = []
    reward_function = None
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "mu": 2, # Number of GRPO optimization steps per batch
        "I": 3, # Number of policy updates
        "M": 4, # Number of batches per iteration
    }
    train_grpo(
        initial_policy=initial_policy,
        traces=traces,
        reward_function=reward_function,
        config=config
    )
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from src.earl.models.facts.rl_fact import RLFact
from src.earl.models.facts.sl_fact import SLFact


def get_importance_thresholds(env, bb_model, perc=0.1):
    print('Calculating importance thresholds...')
    n_ep = 100

    importances_from = []
    importances_to = []
    importances_number = []
    total_importances = []

    for i in tqdm(range(n_ep)):
        done = False
        obs, _ = env.reset()

        while not done:
            action = bb_model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)

            torch_obs = torch.tensor(obs).unsqueeze(0)

            distribution = bb_model.model.policy.get_distribution(torch_obs.squeeze().reshape(1, -1)).distribution

            importance_from = abs(max(distribution[0].probs.squeeze()) - min(distribution[0].probs.squeeze()))
            importance_to = abs(max(distribution[1].probs.squeeze()) - min(distribution[1].probs.squeeze()))
            importance_number = abs(max(distribution[2].probs.squeeze()) - min(distribution[2].probs.squeeze()))

            total_importance = importance_from + importance_to + importance_number

            importances_from.append(importance_from.item())
            importances_to.append(importance_to.item())
            importances_number.append(importance_number.item())
            total_importances.append(total_importance.item())

    return (np.quantile(importances_from, perc),
            np.quantile(importances_to, perc),
            np.quantile(importances_number, perc),
            np.quantile(total_importances, perc))

def get_facts(env, bb_model, horizon=10, perc=0.1, n_states=100):
    '''
    Generates a set of interesting factual states
    Interesting state are those where there is a large difference in taking different actions
    Returns n_states facts whose importance is higher than the threshold in one action dimension
    Limits the fact actions and target actions to common actions to enable dataset-based methods like GANterfactual-RL
    '''
    thresholds = get_importance_thresholds(env, bb_model, perc)

    common_actions = get_common_actions(env, bb_model)

    sl_facts = []
    rl_facts = []

    n_ep = 100

    print('Collecting facts...')
    for i in tqdm(range(n_ep)):
        done = False
        obs, _ = env.reset()
        prev_states = []
        actions = []
        count = 0 
        while not done:
            action = bb_model.predict(obs)

            include, target_action = if_include(thresholds, bb_model, action, obs, common_actions)

            if include and len(prev_states) >= horizon and tuple(action) in common_actions:
                sl_fact = SLFact(obs, action, action)
                rl_fact = RLFact(obs, action, prev_states, env_states=[], actions=actions, horizon=horizon, target_action=action)

                rl_facts.append(rl_fact)
                sl_facts.append(sl_fact)
                prev_states = []
                actions = []

            actions.append(action)
            prev_states.append(env.get_state())
            obs, rew, done, trunc, info = env.step(action)
            count+=1

    # select random subset of facts
    collect_facts = list(zip(sl_facts, rl_facts))
    collect_facts = random.sample(collect_facts, n_states)
    sl_facts, rl_facts = zip(*collect_facts)  # separate the pairs

    print('Total number of states: ', count)
    print('Collected {} important states.'.format(len(sl_facts)))
    return sl_facts, rl_facts

def if_include(thresholds, bb_model, action, obs, common_actions):
    torch_obs = torch.tensor(obs).unsqueeze(0)
    distribution = bb_model.model.policy.get_distribution(torch_obs.squeeze().reshape(1, -1)).distribution

    importance_from = abs(max(distribution[0].probs.squeeze()) - min(distribution[0].probs.squeeze()))
    importance_to = abs(max(distribution[1].probs.squeeze()) - min(distribution[1].probs.squeeze()))
    importance_number = abs(max(distribution[2].probs.squeeze()) - min(distribution[2].probs.squeeze()))

    diffs = [thresholds[0] - importance_from.item(), thresholds[1] - importance_to.item(), thresholds[2] - importance_number.item()]

    include = False
    target_action = None

    # if any of the importances are within the highest threshold
    most_informative_dim = np.argmin(diffs)
    if diffs[most_informative_dim] < 0:
        # possible target actions differ in one dimension (the important one) from the original and it is a common action (important for GAN)
        possible_target_actions = [ta for ta in common_actions if action[most_informative_dim] != ta[most_informative_dim] and sum(np.array(action) == np.array(ta)) == 2]

        if len(possible_target_actions) > 0:
            target_action = random.sample(possible_target_actions, 1)[0]
            return True, target_action

    return include, target_action


def get_common_actions(env, bb_model):
    print('Calculating common actions...')
    n_ep = 100
    threshold = 100

    common_actions = {(i, j, n): 0 for i in range(0, 5) for j in range(0, 5) for n in range(0, 10)}

    for i in tqdm(range(n_ep)):
        obs, _ = env.reset()
        done = False

        while not done:
            action = bb_model.predict(obs)
            f, to, n = action

            common_actions[(f, to, n)] += 1

            obs, rew, done, trunc, info = env.step(action)

    common_actions = {action: freq for action, freq in common_actions.items() if freq >= threshold}

    return list(common_actions.keys())
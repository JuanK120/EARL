import copy
import math
from datetime import datetime

import numpy as np



class AbstractObjective:
    ''' Describes an objective function for counterfactual search '''

    def __init__(self, env, bb_model, horizon=5, n_sim=10, transition_model=None):
        self.horizon = horizon
        self.transition_model = transition_model
        self.env = env
        self.bb_model = bb_model

        self.n_sim = n_sim

        self.noop = -1

    def process_actions(self, actions, allow_first_noop=True):
        ''' Process a sequence of actions to remove NoOp from start and the end '''
        action = self.env.action_space.sample()
        if isinstance(action, np.ndarray):
            action_type = 'multi_discrete'
            action_size = len(action)
        elif isinstance(action, int):
            action_type = 'discrete'
            action_size = 1
        elif isinstance(action, np.integer):
            action= int(action)
            action_type = 'discrete'
            action_size = 1
        else:
            raise ValueError('Only Discrete and MultiDiscrete action spaces are supported')

        if action_type == 'multi_discrete':
            actions = np.array(actions).reshape(-1, action_size).tolist()

        first_real_action_index = 0
        while (first_real_action_index < len(actions)):
            if (action_type == 'discrete' and (actions[first_real_action_index] != self.noop)) or\
                    (action_type == 'multi_discrete' and self.noop not in actions[first_real_action_index]):
                break

            first_real_action_index += 1

        # this is specifically for prefactuals, where first actions cannot be noops
        if not allow_first_noop and first_real_action_index != 0:
            return [], 0

        # if all actions are NoOp
        if first_real_action_index >= len(actions):
            return [], 0

        last_real_action_index = len(actions) - 1
        while actions[last_real_action_index] == self.noop:
            last_real_action_index -= 1

        between_actions = actions[first_real_action_index: (last_real_action_index + 1)]

        # if NoOp actions are not on start or the end
        if self.noop in between_actions:
            return [], 0

        return between_actions, first_real_action_index

    def get_first_state(self, fact, first_action_index=0):
        ''' Returns the fist state given the first action index '''
        return None, None

    def validity(self, target_action, obs):
        ''' Evaluates validity based on the outcome '''
        # valid_outcome = outcome.sf_outcome(obs)
        valid_outcome = tuple(self.bb_model.predict(obs)) == target_action
        # IMPORTANT: return 1 if the class has changed -- to be compatible with minimization used by NSGA
        return not valid_outcome

    def recency(self, fact, actions):
        # TODO: this has to include first_action_index
        diff = [fact.actions[i] != actions[i] for i in range(len(actions))]

        n = len(actions)
        k = 2.0/(n * (n + 1))
        weights = [k * (i+1) for i in range(len(actions))]

        weights.reverse()  # the biggest penalty for the first (least recent) action

        recency = sum([diff[i] * weights[i] for i in range(len(actions))])

        return recency

    def sparsity(self, fact, actions, first_action_index):
        ''' Evaluates sparsity as the boolean difference between actions in fact and the passed list of actions'''
        num_actions = len(actions)
        fact_actions = fact.actions[first_action_index: (first_action_index + num_actions)]

        # diff actions + sum of all actions not taken
        diff_actions = sum(np.array(fact_actions) != np.array(actions))
        extra_actions = len(fact.actions) - num_actions
        return ((diff_actions + extra_actions) / len(fact.actions))

    def reachability(self, actions):
        ''' Evaluates reachability as the length of a sequence of actions'''
        if len(actions) == 0:
            return 1

        return len(actions) * 1.0 / self.horizon

    def calculate_stochastic_properties(self, fact, actions, bb_model, first_state, first_env_state ):
        ''' Calculates all properties that rely on stochastic simulations '''

        n_sim = self.n_sim
        cfs = []

        target_outcome = 0.0
        fidelities = []

        exceptionallities = []

        for s in range(n_sim): 
            self.env.reset()
 
            self.env.set_nonstoch_state(first_state, first_env_state)


            fid = 1.0
            exc = 0.0

            if len(actions) == 0:
                return 1, 1, 1, []

            done = False
            early_break = False

            first_state, first_env_state = self.get_first_state(fact)

            obs = copy.copy(first_state)

            for a in actions:
                if done:
                    early_break = True
                    break

                # calculate fidelity
                prob = bb_model.get_action_prob(obs, a)
                fid *= prob

                # step in the environment
                new_obs, rew, done, trunc, _ = self.env.step(a)

                # calculate exceptionality
                if 'exceptionality' in self.objectives:
                    trans_prob = self.transition_model.get_probability(list(obs), a, list(new_obs))
                    exc += trans_prob

                obs = new_obs

            if not early_break and not done:
                # check if validity is satisfied
                validity = self.validity(fact.target_action, obs)
                target_outcome += int(not validity)

                if validity == 0:
                    # since 0 indicates that validity is satisfied
                    cfs.append((list(copy.copy(obs)), {'validity': 0, 'fidelity': 1 - fid,  'exceptionality': exc / len(actions)}))

                fidelities.append(fid / len(actions))

        # calculate stochasticity
        # if outcome is confirmed everytime that is a bad thing
        stochasticity = (target_outcome / n_sim)

        # calculate average fidelity over all paths
        if len(fidelities):
            fidelity = sum(fidelities) / (len(fidelities) * 1.0)
        else:
            fidelity = 0

        # calculate mean exceptionallity
        if len(exceptionallities):
            exceptional = sum(exceptionallities) / len(exceptionallities)
        else:
            exceptional = 1

        # calculate validity
        validity = 1 - target_outcome / n_sim

        # 1 - fidelity because we want to minimize it
        return stochasticity, 1 - fidelity, validity, exceptional, cfs
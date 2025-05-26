import copy

import numpy as np

from src.earl.objectives.abstract_obj_expl import AbstractObjective



class CfExplObj(AbstractObjective):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, n_sim=10, horizon=5):

        super(CfExplObj, self).__init__(env, bb_model, horizon=horizon, n_sim=n_sim)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'fidelity', 'reachability']
        self.constraints = ['validity']

        self.n_sim = n_sim

    def validity(self, target_action, obs):
        ''' Evaluates validity based on the outcome '''
        action = self.bb_model.predict(obs) 
        if isinstance(action, (np.ndarray,list)):
            valid_outcome = tuple(action) != target_action # For now -- valid if the action is different from the original action
        elif isinstance(action, (int, np.int64, np.integer)):
            valid_outcome = action != target_action
        # IMPORTANT: return 1 if the class has not changed -- to be compatible with minimization used by NSGA
        return not valid_outcome

    def evaluate(self, fact, actions):
        return self._evaluate(fact, actions, allow_first_noop=True)

    def _evaluate(self, fact, actions, allow_first_noop=False):
        actions, first_action_index = self.process_actions(actions, allow_first_noop=allow_first_noop)
        first_state, first_env_state = self.get_first_state(fact)

        if len(actions) == 0:
            return {'uncertainty': 1,
                    'fidelity': 1,
                    'reachability': 1
                    }, {'validity': True}, []

        stochasticity, fidelity, _, num_cfs, cfs = self.calculate_stochastic_properties(fact,
                                                                                        actions,
                                                                                        self.bb_model,
                                                                                        first_state,
                                                                                        first_env_state)
        reachability = self.reachability(actions)
        for cf in cfs:
            cf[1].update({'reachability': reachability, 'uncertainty': stochasticity})

        objectives = {'uncertainty': stochasticity,
                      'fidelity': fidelity,
                      'reachability': reachability}

        constraints = {'validity': num_cfs == 0}

        return objectives, constraints, cfs


    def get_first_state(self, fact):
        return copy.copy(fact.get_state(0)), copy.deepcopy(fact.get_env_state(0))




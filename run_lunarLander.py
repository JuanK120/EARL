import random
import time

import pandas as pd
from tqdm import tqdm

from gymnasium_examples.evaluation import evaluate_explanations
from gymnasium_examples.fact_generation import get_facts
from src.earl.methods.cf.ganterfactual import GANterfactual
from src.earl.methods.cf.raccer_advance import NSGARaccerAdvance
from src.earl.methods.cf.raccer_hts import RACCERHTS
from src.earl.methods.cf.raccer_rewind import NSGARaccerRewind
import gymnasium as gym
from gymnasium_examples.lunarLanderEnv import LunarLanderEnv

from src.earl.models.bb_models.ppo_model import PPOModel
from src.earl.models.bb_models.dqn_model import DQNModel

from src.earl.utils.util import seed_everything


def main():
    # ----------- user-defined ------------
    seed_everything(0)
    env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5) 
    env = LunarLanderEnv(env)
    
    print(env.action_space)

    bb_model = DQNModel(env, 'gymnasium_examples/trained_models/lunarLander', arch=[512, 512], training_timesteps=5e5, lr=0.0005, batch_size=512, verbose=1)

    state_feature_names = ["x_position", "y_position", "x_velocity", "y_velocity", 
                 "angle", "angular_velocity", "left_leg_contact", "right_leg_contact"] 
    categorical_features = state_feature_names
    continuous_features = []

    params = [f'--columns={state_feature_names}',
              f'--categorical_features={categorical_features}',
              f'--continuous_features={continuous_features}']

    horizon = 5
    sl_facts, rl_facts = get_facts(env, bb_model, horizon=horizon, perc=0.1, n_states=100)

    domains = list({bb_model.predict(f.state) for f in sl_facts}.union({f.target_action for f in sl_facts}))

    
    print("action space ", env.action_space, "\n sample ", env.action_space.sample() )
    RACCER_HTS = RACCERHTS(env, bb_model, horizon, n_expand=20, max_level=horizon, n_iter=300)
    RACCER_Advance = NSGARaccerAdvance(env, bb_model, horizon=horizon, n_gen=24, pop_size=25, xl=[0], xu=[3])
    RACCER_Rewind = NSGARaccerRewind(env, bb_model, horizon=horizon, n_gen=24, pop_size=50, xl=[0], xu=[3])


    rl_methods = [RACCER_Advance, RACCER_Rewind, RACCER_HTS]
    rl_eval_paths = ['raccer_advance', 'raccer_rewind', 'raccer_hts']

    for i, m in enumerate(rl_methods):
        record = []
        print('Running {}'.format(rl_eval_paths[i]))

        for f in tqdm(rl_facts):
            start = time.time()
            action = f.target_action
            cfs = m.explain(f, target=action)
            end = time.time()
            if len(cfs):
                print('Generated {} cfs'.format(len(cfs)))
                for cf in cfs:
                    record.append((list(f.state), list(cf.cf), action, end-start))

        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'action', 'gen_time'])
        record_df.to_csv('gymnasium_examples/results/{}.csv'.format(rl_eval_paths[i]), index=False)
    
    ganterfactual = GANterfactual(env,
                                  bb_model,
                                  batch_size=64,
                                  num_features=8,
                                  domains=domains,
                                  training_timesteps=1500,
                                  dataset_size=5e5,
                                  dataset_path='gymnasium_examples/datasets/ganterfactual_data',
                                  model_save_path= 'gymnasium_examples/trained_models/ganterfactual',
                                  params=params)

    sl_methods = [ganterfactual]
    sl_eval_paths = ['ganterfactual']

    for i, m in enumerate(sl_methods):
        record = []
        print('Running {}'.format(sl_eval_paths[i]))

        for f in tqdm(sl_facts):
            start = time.time()
            # choose one target action randomly as long as it's
            # different than the one being chosen by the agent
            target_action = random.choice([a for a in domains if a != f.target_action])
            cfs = m.explain(f, target=target_action)
            end = time.time()
            for cf in cfs:
                record.append((list(f.state), list(cf), target_action, end-start))

        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'action','gen_time'])
        record_df.to_csv('gymnasium_examples/results/{}.csv'.format(sl_eval_paths[i]), index=False)

    evaluate_explanations(env, 'gymnasium_examples/results/', sl_eval_paths + rl_eval_paths, N_TEST=100)

if __name__ == '__main__':
    main()
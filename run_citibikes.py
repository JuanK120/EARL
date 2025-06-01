import random
import time

import pandas as pd
from tqdm import tqdm

from citibikes.evaluation import evaluate_explanations
from citibikes.fact_generation import get_facts
from src.earl.methods.cf.ganterfactual import GANterfactual
from src.earl.methods.cf.raccer_advance import NSGARaccerAdvance
from src.earl.methods.cf.raccer_hts import RACCERHTS
from src.earl.methods.cf.raccer_rewind import NSGARaccerRewind
from citibikes.citibikes_env import CitiBikes

from src.earl.models.bb_models.ppo_model import PPOModel

from src.earl.utils.util import seed_everything


def main():
    # ----------- user-defined ------------
    seed_everything(0)
    env = CitiBikes()

    bb_model = PPOModel(env, 'citibikes/trained_models/citibikes', arch=[512, 512], training_timesteps=5e5, lr=0.0005, batch_size=512, verbose=1)

    params = [f'--columns={env.state_feature_names}',
              f'--categorical_features={env.categorical_features}',
              f'--continuous_features={env.continuous_features}']

    horizon = 5
    sl_facts, rl_facts = get_facts(env, bb_model, horizon=horizon, perc=0.1, n_states=100)

    domains = list({tuple(bb_model.predict(f.state)) for f in sl_facts}.union({tuple(f.target_action) for f in sl_facts}))

    RACCER_HTS = RACCERHTS(env, bb_model, horizon, n_expand=20, max_level=horizon, n_iter=300)
    RACCER_Advance = NSGARaccerAdvance(env, bb_model, horizon=horizon, n_gen=24, pop_size=25, xl=[0, 0, 0], xu=[4, 4, 9])
    RACCER_Rewind = NSGARaccerRewind(env, bb_model, horizon=horizon, n_gen=24, pop_size=50, xl=[0, 0, 0], xu=[4, 4, 9])

    ganterfactual = GANterfactual(env,
                                  bb_model,
                                  batch_size=64,
                                  num_features=38,
                                  domains=domains,
                                  training_timesteps=1500,
                                  dataset_size=5e5,
                                  dataset_path='citibikes/datasets/ganterfactual_data',
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
            cfs = m.explain(f, target=tuple(target_action))
            end = time.time()
            for cf in cfs:
                record.append((list(f.state), list(cf), end-start))

        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'gen_time'])
        record_df.to_csv('citibikes/results/{}.csv'.format(sl_eval_paths[i]), index=False)

    rl_methods = [RACCER_HTS, RACCER_Advance, RACCER_Rewind, ]
    rl_eval_paths = ['raccer_hts', 'raccer_advance', 'raccer_rewind',]

    """for i, m in enumerate(rl_methods):
        record = []
        print('Running {}'.format(rl_eval_paths[i]))

        for f in tqdm(rl_facts):
            start = time.time()
            cfs = m.explain(f, target=f.target_action)
            end = time.time()
            if len(cfs):
                print('Generated {} cfs'.format(len(cfs)))
                for cf in cfs:
                    record.append((list(f.state), list(cf.cf), end-start))

        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'gen_time'])
        record_df.to_csv('citibikes/results/{}.csv'.format(rl_eval_paths[i]), index=False)"""

    

    evaluate_explanations(env, 'citibikes/results/', sl_eval_paths + rl_eval_paths, N_TEST=100)

if __name__ == '__main__':
    main()
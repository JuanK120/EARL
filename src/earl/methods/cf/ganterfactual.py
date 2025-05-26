# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
import os
import time
import shutil

import numpy as np
import pandas as pd
import torch

from src.earl.algorithms.star_gan.dataset_generation import generate_dataset_gan
from src.earl.algorithms.star_gan.model import Generator
from src.earl.algorithms.star_gan.train import train_star_gan
from src.earl.methods.abstract_method import AbstractMethod


class GANterfactual(AbstractMethod):
    def __init__(self, env, bb_model, dataset_size=int(5e5), num_features=10, training_timesteps=int(5e3),
                 batch_size=512, domains=None, dataset_path='citibikes/datasets/ganterfactual_data', model_save_path = 'citibikes/trained_models/ganterfactual', params=None):
        super().__init__()

        if params is None:
            params = {}

        self.env = env
        self.bb_model = bb_model
        self.dataset_size = dataset_size
        self.num_features = num_features
        self.training_timesteps = training_timesteps
        self.batch_size = batch_size
        self.params = self.process_params(params)
        self.dataset_path = dataset_path

        # TODO: generator and discriminator architecture should be here too
        if domains is None:
            self.domains = self.generate_domains(self.env)
        else:
            self.domains = domains

        print('GANterfactual domains = {}'.format(self.domains))

        self.nb_domains = len(self.domains)

        self.label_mapping = self.get_label_mapping()

        self.model_save_path = model_save_path

        self.generator_path = os.path.join(self.model_save_path, '{}-G.ckpt'.format(training_timesteps))

        try:
            self.generator = Generator(image_size=self.num_features, c_dim=self.nb_domains)
            self.generator.eval()
            self.generator.load_state_dict(torch.load(self.generator_path))
        except FileNotFoundError:
            self.run_ganterfactual()

    def generate_domains(self, env):
        action = env.action_space.sample()

        if isinstance(action, (int, np.int64, np.integer)):
            return np.arange(env.action_space.n)
        elif isinstance(action, np.ndarray) and len(action.shape) == 1:
            ns = [list(np.arange(0, env.action_space[i].n)) for i in range(len(env.action_space))]

            els = list(itertools.product(*ns))
            return els
        else:
            raise ValueError('Only Discrete and MultiDiscrete action spaces are supported')

    def run_ganterfactual(self):
        # generate datasets for training ganterfactual if it does not exist already
        if not os.path.isdir(os.path.join(self.dataset_path, 'test')):
            try:
                print('Preparing dataset for GANterfactual-RL...')
                generate_dataset_gan(self.bb_model,
                                     self.env,
                                     self.dataset_path,
                                     self.dataset_size,
                                     self.nb_domains,
                                     self.domains)
            except (ValueError, TypeError) as err:
                print(err)
                if os.path.exists(self.dataset_path):
                    shutil.rmtree(self.dataset_path)

        # train
        print('Training GANterfactual-RL...')
        train_star_gan(image_size=self.num_features,
                       image_channels=1,
                       c_dim=self.nb_domains,
                       batch_size=self.batch_size,
                       domains=self.domains,
                       agent=self.bb_model,
                       num_iters=self.training_timesteps,
                       save_path=self.model_save_path,
                       dataset_path=self.dataset_path)

    def explain(self, fact, target):
        ''' Returns all cfs found in the tree '''
        tensor_fact = torch.tensor(fact.state, dtype=torch.double).unsqueeze(0)

        target_label = self.label_mapping[target]

        cf = self.generate_counterfactual(tensor_fact, target_label, self.nb_domains)

        cf = cf.squeeze().tolist()

        # rounding for categorical features
        discrete_feature_ids = [i for i, c in enumerate(self.params.columns) if c in self.params.categorical_features]
        cf = [round(feature) if i in discrete_feature_ids else feature for i, feature in enumerate(cf)]

        return [cf]

    def generate_counterfactual(self, fact, target, nb_domains):
        # convert target class to onehot
        onehot_target_class = np.zeros(nb_domains, dtype=int)
        onehot_target_class[target] = 1
        onehot_target_class = torch.tensor([onehot_target_class])

        # generate counterfactual
        counterfactual = self.generator.double()(fact, onehot_target_class)

        return counterfactual

    def get_label_mapping(self):
        label_mapping = {}
        i = 0
        for d in self.domains:
            label_mapping[d] = i
            i += 1

        return label_mapping
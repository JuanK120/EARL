import os
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import gymnasium as gym
from PIL import Image, ImageOps




def  create_dataset(env, size, target_path, agent, agent_type="deepq", seed=None, noop_range=(0, 30),
                   epsilon=0.0, power_pill_objective=False, domains=None, deepq_preprocessing = True,
                   ablate_agent=False):
    """
    Creates a data set with the following structure. A directory is created at target_path with the subdirectories
    'train' and 'test'. Each of these has a subdirectory for every domain. The domain directories contain the generated
    sample images. This function puts all samples into the 'train' directory. Use split_dataset() to split the data set.

    :param env_name: Name of the gym environment to generate a data set for.
    :param size: The total amount of samples that should be generated.
    :param target_path: The path at which the data set should be saved.
    :param agent: The agent that should be used to classify samples.
    :param agent: The type of agent. "deepq" for Keras DeepQ or "acer" for gym.baselines ACER.
        For Space Invader, a pytorch agent is expected and this flag is not used.  (Keras DeepQ, Pytorch and Baselines ACER are supported).
    :param seed: Seed for random number generator.
    :param noop_range: Range (min, max) of NOOPs that are executed at the beginning of an episode to generate a random
        offset.
    :param epsilon: epsilon value in range [0, 1] for the epsilon-greedy policy that is used to reach more diverse
        states.
    :param power_pill_objective: Whether or not to use the power pill objective on Pac-Man.
    :param domains: List of domain names. If None, the amount of domains will automatically be determined by the given
        gym environment and the names will be 0, 1, 2...
    :param ablate_agent: Whether or not to ablate the laser canon from frames when input to the space invaders agent.
    :return: None
    """
    # init environment
    # create domains and folder structure
    if domains is None:
        domains = list(map(str, np.arange(env.action_space.n)))
    train_path, test_path, success = _setup_dicts(target_path, domains)
    if not success:
        raise FileExistsError(f"Target directory '{target_path}' already exists")

    # generate train and test set
    print("Generating datasets...")
    _generate_set(size, train_path, env, agent, agent_type, domains, noop_range, epsilon)


def split_dataset(dataset_path, test_portion, domains):
    """
    Splits an existing data set into a train and a test set by randomly selecting samples for the test set.

    :param dataset_path: Path to the data set directory. All samples have to be in 'train'.
    :param test_portion: Proportion of the test set in range [0, 1] (e.g. 0.1 for 10% test samples).
    :param domains: List of domain names that for domains that should be effected by the split.
    :return: None
    """
    print("Splitting datasets...")
    for domain in domains:
        # get absolute split size per domain
        train_path = os.path.join(dataset_path, "train", '{}.csv'.format(domain))
        test_path = os.path.join(dataset_path, "test", '{}.csv'.format(domain))

        df = pd.read_csv(train_path, header=0)
        test_size = int(len(df) * test_portion)

        test_indices = random.sample(range(len(df)), test_size)

        test_df = df.iloc[test_indices]
        train_df = df.drop(test_indices, axis=0)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)


def under_sample(dataset_path, domains, min_size=None):
    """
    Under-samples the given data set by removing randomly chosen samples from domains that contain more samples until
    each domain contains an equal amount of samples or min_size is reached.

    :param dataset_path: Path to the data set directory.
    :param min_size: The minimum amount of samples that domains with more samples should keep. If None, domains will
        be under-sampled to the size of the smallest domain.
    :return: None
    """

    print("Down-sampling datasets...")
    for subset in ["train"]:
        # get domain sizes
        domain_sizes = []
        for domain in domains:
            domain_path = os.path.join(dataset_path, subset, '{}.csv'.format(domain))
            df = pd.read_csv(domain_path, header=0)
            size = len(df)
            domain_sizes.append(size)

        # get minimum size (used for down sampling)
        new_size = int(min(domain_sizes))
        if min_size is not None:
            new_size = int(max(new_size, min_size))

        # remove files to get an equal distribution
        for domain in domains:
            domain_path = os.path.join(dataset_path, subset, '{}.csv'.format(domain))
            df = pd.read_csv(domain_path, header=0)
            size = len(df)

            # randomly choose (size - new_size) samples to remove
            indices_to_remove = random.sample(range(size), int(max(size - new_size, 0)))
            df = df.drop(indices_to_remove)

            df.to_csv(domain_path, index=False)


def create_unique_dataset(new_dataset_path, old_dataset_path, domains):
    """
    Creates a data set without duplicate samples on the basis of a given data set.

    :param new_dataset_path: Path for the newly created unique data set.
    :param old_dataset_path: Path to an existing data set that possibly contains duplicate samples.
    :return: None
    """
    print("Creating a datasets with unique samples...")

    for domain in domains:
        train_path = os.path.join(old_dataset_path, 'train', '{}.csv'.format(domain))
        _save_unique_samples(train_path, domain)
        print(f"Finished domain {domain}.")


def create_clean_test_set(dataset_path, samples_per_domain):
    """
    Creates a clean test set for a possibly dirty data set. The clean test set is generated by selecting random samples
    from the train set as test samples. Duplicates of the selected test samples within the train set are removed from the train set.

    :param dataset_path: Path to the possibly dirty data set. It is assumed that the existing data set only contains all
        samples in the train set.
    :param samples_per_domain: Amount of test samples that should be selected per domain.
    :return: None
    """
    print("Creating a clean test set...")
    for domain in os.listdir(os.path.join(dataset_path, "train")):
        domain_path = os.path.join(dataset_path, "train", domain)
        if os.path.isdir(domain_path):
            domain_file_names = []
            for i, item in enumerate(os.listdir(domain_path)):
                # get file name
                file_name = os.path.join(domain_path, item)
                domain_file_names.append(file_name)

            random_indices = random.sample(range(len(domain_file_names)), len(domain_file_names))
            tabu_list = []
            collected_samples = 0

            for i in random_indices:
                if i in tabu_list:
                    continue
                # open image sample
                img = Image.open(domain_file_names[i])
                sample = np.array(img)

                # delete from train and copy to test
                os.remove(domain_file_names[i])
                img.save(os.path.join(dataset_path, "test", domain, f"{i}.png"))

                # add to tabu
                tabu_list.append(i)
                collected_samples += 1

                # search for duplicates and delete and tabu them
                for j, other_file in enumerate(domain_file_names):
                    if j in tabu_list:
                        continue
                    other_sample = np.array(Image.open(domain_file_names[j]))
                    if (other_sample == sample).all():
                        tabu_list.append(j)
                        os.remove(other_file)

                if collected_samples >= samples_per_domain:
                    # finished
                    break
        print(f"Finished Domain {domain}.")


def _generate_set(size, path, env, agent, agent_type, domains, noop_range, epsilon):
    try :
        obs, _ = env.reset(int(datetime.timestamp(datetime.now())*10000))
    except: 
        obs, _ = env.reset()
    # not using a for loop because the step counter should only be increased
    # if the frame is actually saved as a training sample
    step = 0

    action = env.action_space.sample()

    print("ACTION SPACE TYPE =", type(env.action_space))
    print("ACTION SPACE SAMPLE =", env.action_space.sample())
    print("ACTION SPACE SAMPLE TYPE =", type(env.action_space.sample()))

    

    if isinstance(action, (int, np.integer, np.int64)):
        # for discrete spaces
        observations = {action: [] for action in domains}
    elif isinstance(action, np.ndarray) and len(action.shape) == 1:
        # for multi discrete spaces
        observations = {tuple(action): [] for action in domains}
    else:
        raise ValueError('Only Discrete and MultiDiscrete Gym action spaces are supported for GANterfactual.')

    while step < size:
        done = False
        obs, _ = env.reset()

        while not done:

            if np.random.uniform() < epsilon:
                # random exploration to increase state diversity (frame must not be saved as a training sample!)
                action = env.action_space.sample()
            else:
                action = agent.predict(obs)

                if isinstance(obs, list):
                    obs = np.array(obs)

                if isinstance(action, (int, np.integer, np.int64)):
                    # for discrete actions
                    if action in domains:
                        observations[action].append(obs.reshape((1, -1)).squeeze())
                elif isinstance(action, list):
                    if tuple(action) in domains:
                        observations[tuple(action)].append(obs.reshape((1, -1)).squeeze())
                else:
                    raise ValueError('Only Discrete and MultiDiscrete Gym action spaces are supported for GANterfactual.')

                step += 1

            obs, reward, done, trunc, info = env.step(action)

        if step % 100 == 0:
            print('Finished {} steps'.format(step))

    for a in domains:
        df = pd.DataFrame(observations[a])
        df.to_csv(os.path.join(path, '{}.csv'.format(a)), index=False)


def _save_unique_samples(train_path, domain):
    df = pd.read_csv(train_path, header=0)

    df = df.drop_duplicates()
    df.to_csv(train_path, index=False)


def _setup_dicts(target_path, domains):
    try:
        # creating the directory structure
        os.mkdir(target_path)
        # train dir
        train_path = os.path.join(target_path, "train")
        os.mkdir(train_path)

        # test dir
        test_path = os.path.join(target_path, "test")
        os.mkdir(test_path)

        return train_path, test_path, True
    except FileExistsError:
        return None, None, False

def generate_dataset_gan(agent, env, dataset_path, nb_samples, nb_domains, domains):
    agent_type = "acer"

    create_dataset(env, nb_samples, dataset_path, agent, agent_type=agent_type, seed=42, epsilon=0.2, domains=domains)

    under_sample(dataset_path, min_size=nb_samples / nb_domains, domains=domains)
    create_unique_dataset(dataset_path, dataset_path, domains)
    under_sample(dataset_path, domains)
    split_dataset(dataset_path, 0.1, domains)


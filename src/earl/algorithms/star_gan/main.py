import os
import argparse
from torch.backends import cudnn

from src.earl.algorithms.star_gan.data_loader import get_loader
from src.earl.algorithms.star_gan.solver import Solver


def str2bool(v):
    return v.lower() in ('true')

def main(config, agent):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    data_loader = get_loader(config.dataset_path, config.batch_size, config.mode, num_workers=1, n_domains=config.c_dim,domains=config.domains)

    # Solver for training and testing StarGAN.
    solver = Solver(agent, data_loader, config)

    if config.mode == 'train':
            solver.train()
    elif config.mode == 'test':
            solver.test()

def get_parser():
    def list_of_tuples(arg):
        import ast
        return ast.literal_eval(arg)

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st datasets)')
    parser.add_argument('--domains', type=list_of_tuples, default=[], help='domains')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd datasets)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA datasets')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD datasets')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=2, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=2, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--datasets', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=8e-5, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=8e-5, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA datasets',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='datasets/gridworld')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')
    parser.add_argument('--dataset_path', type=str, default="", help='path to the train and test datasets')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    # Extensions
    parser.add_argument('--image_channels', type=int, default=3, help='number of channels of each image')
    parser.add_argument('--agent_path', type=str, default=None, help='path to a h5 file containing a rl agent')
    parser.add_argument('--lambda_counter', type=float, default=10, help='weight for counter loss')
    parser.add_argument('--counter_mode', type=str, default='raw', help='whether to use "raw", "softmax",'
                                                                              '"advantage", or "z-score"')
    parser.add_argument('--selective_counter', type=bool, default=True, help='whether to only use samples where'
                                                                             'c_trg != c_org for counter-loss')
    parser.add_argument('--agent_type', type=str, default="deepq", help='which agent type to use (deepq,'
                                                                                'olson, acer)')
    parser.add_argument('--ablate_agent', type=bool, default=False, help='whether to ablate the laser canon before'
                                                                         'inputting a frame to the agent')

    return parser


if __name__ == '__main__':
    parser = get_parser()

    config = parser.parse_args()

    main(config)

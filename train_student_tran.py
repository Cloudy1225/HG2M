import argparse
import os.path as osp

import torch
import numpy as np

from dataloader import load_data
from train_and_eval import distill_run_transductive
import warnings

from utils import set_seed, get_training_config, get_logger, check_writable

warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DGL implementation')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--eval_interval', type=int, default=1,
        help='Evaluate once per how many epochs'
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs",
        help="Path to save outputs"
    )
    parser.add_argument(
        '--num_exp', type=int, default=5,
        help='Repeat how many experiments'
    )
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0.,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )

    """Dataset"""
    parser.add_argument(
        '--dataset', type=str, default='TMDB',
        choices=['TMDB', 'CroVal', 'ArXiv', 'IGB-549K-19', 'IGB-549K-2K', 'IGB-3M-19'],
        help='Dataset'
    )

    """Model"""
    parser.add_argument(
        '--model_config_path',
        type=str,
        default='./train.conf.yaml',
        help='Path to model configuration'
    )
    parser.add_argument(
        '--teacher', type=str, default='RSAGE',
        choices=['RSAGE', 'RGCN', 'RGAT'],
        help='Model'
    )
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of workers for sampler'
    )

    """Optimization"""
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--model_save', type=bool, default=True)

    """Distillation"""
    parser.add_argument(
        "--lamda",
        type=float,
        default=0.,
        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
    )
    args = parser.parse_args()
    return args


def run(args):
    """ Set seed, device, and logger """
    set_seed(args.seed)

    output_dir = args.output_path
    if args.feature_noise > 0:
        output_dir = str(osp.join(
            output_dir, 'noisy_features', f'noise_{args.feature_noise}'
        ))
    output_dir = str(osp.join(
        output_dir,
        'transductive',
        args.dataset,
        args.teacher,
        f'seed_{args.seed}',
    ))
    args.output_dir = output_dir
    check_writable(output_dir, overwrite=False)

    """ Load data """
    g, splits, generate_node_features = load_data(args.dataset)
    if 0 < args.feature_noise <= 1:
        feats = g.nodes[g.category].data['feat']
        feats = (1 - args.feature_noise
                ) * feats + args.feature_noise * torch.randn_like(feats)
        g.nodes[g.category].data['feat'] = feats
    g = generate_node_features(g)
    args.category = g.category

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, 'HGNN2MLP', args.dataset)
    conf = dict(args.__dict__, **conf)
    # conf = dict(conf, **args.__dict__)

    conf['device'] = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    conf['teacher_path'] = osp.join(output_dir, 'teacher.pt')
    conf['teacher_conf_path'] = osp.join(output_dir, 'teacher_conf.json')
    conf['student_path'] = osp.join(output_dir, 'student.pt')

    logger = get_logger(osp.join(output_dir, 'log_student.txt'))

    # logger.info = print

    logger.info(conf)
    logger.info(str(g))
    logger.info(f'Splits: Train {splits[0].shape[0]} '
                f'Val {splits[1].shape[0]} Test {splits[2].shape[0]}')

    return distill_run_transductive(conf, g, splits, log=logger.info)


if __name__ == "__main__":
    args = get_args()

    ACCs = []
    for seed in range(args.num_exp):
        args.seed = seed
        ACCs.append(run(args))

    print(f'Test ACC: {np.mean(ACCs):.2f}+-{np.std(ACCs):.2f}')

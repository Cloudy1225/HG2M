import argparse
import dgl, torch
import numpy as np
import os.path as osp

from dataloader import load_data
from train_and_eval import distill_run_inductive
import warnings

from utils import set_seed, get_training_config, get_logger, check_writable, graph_split

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

    """Dataset"""
    parser.add_argument(
        '--dataset', type=str, default='TMDB',
        choices=['TMDB', 'CroVal', 'ArXiv', 'IGB-549K-19', 'IGB-549K-2K', 'IGB-3M-19'],
        help='Dataset'
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
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

    output_dir = str(osp.join(
        args.output_path,
        'inductive',
        f'split_rate_{args.split_rate}',
        args.dataset,
        args.teacher,
        f'seed_{args.seed}',
    ))
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)

    """ Load data """
    g, (idx_train, idx_val, idx_test), generate_node_features = load_data(args.dataset)
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(idx_train, idx_val, idx_test,
                                                                                  args.split_rate, args.seed)
    nodes = {ntype: torch.full((g.num_nodes(ntype),), fill_value=True, dtype=torch.bool) for ntype in g.ntypes}
    nodes[g.category] = idx_obs
    obs_g = dgl.node_subgraph(g, nodes, store_ids=False)
    obs_g = dgl.compact_graphs(obs_g)  # delete isolated non-target-type nodes
    obs_g = generate_node_features(obs_g)
    g = generate_node_features(g)
    for ntype, idx in obs_g.ndata[dgl.NID].items():
        if ntype != g.category:
            g.nodes[ntype].data['feat'][idx] = obs_g.nodes[ntype].data['feat']
    obs_g.ndata.pop('_ID')

    splits = obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind
    args.category = g.category

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, 'HGNN2MLP_ind', args.dataset)
    conf = dict(args.__dict__, **conf)
    # conf = dict(conf, **args.__dict__)

    conf['device'] = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    conf['teacher_path'] = osp.join(output_dir, 'teacher.pt')
    conf['teacher_conf_path'] = osp.join(output_dir, 'teacher_conf.json')
    conf['student_path'] = osp.join(output_dir, 'student.pt')

    logger = get_logger(osp.join(output_dir, 'log_student.txt'))

    # logger.info = print

    logger.info(conf)
    logger.info(str(obs_g))
    logger.info(str(g))
    logger.info(f'Splits: Train {obs_idx_train.shape[0]} '
                f'Val {obs_idx_val.shape[0]} Test_tran {obs_idx_test.shape[0]} Test_ind {idx_test_ind.shape[0]}')

    return distill_run_inductive(conf, obs_g, g, splits, log=logger.info)


if __name__ == "__main__":
    args = get_args()

    ACCs = []
    ACC1s = []
    ACC2s = []
    for seed in range(args.num_exp):
        args.seed = seed
        acc_test, acc_test_tran, acc_test_ind = run(args)
        ACCs.append(acc_test)
        ACC1s.append(acc_test_tran)
        ACC2s.append(acc_test_ind)

    print(f'Test ACC: {np.mean(ACCs):.2f}+-{np.std(ACCs):.2f} '
          f'Tran: {np.mean(ACC1s):.2f}+-{np.std(ACC1s):.2f} '
          f'Ind: {np.mean(ACC2s):.2f}+-{np.std(ACC2s):.2f}')

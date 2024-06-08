import argparse
from data.utils import args_set_bool
from ml_collections import ConfigDict
from tqdm import tqdm
import wandb
import torch
import os
import yaml

def args_parser():
    #run.py --datapath DATA_TO_YOUR_INSTANCES --upper 1. --ipm_alpha 0.15 --weight_decay 1.2e-6 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --conv genconv

    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', type=str, default='false')

    # ipm processing
    parser.add_argument('--ipm_restarts', type=int, default=1)  # more does not help
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--ipm_alpha', type=float, default=0.9)
    parser.add_argument('--upper', type=float, default=1.0)

    # training dynamics
    parser.add_argument('--ckpt', type=str, default='true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--micro_batch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)  # must
    parser.add_argument('--use_norm', type=str, default='true')  # must
    parser.add_argument('--use_res', type=str, default='false')  # does not help

    # model related
    parser.add_argument('--bipartite', type=str, default='false')
    parser.add_argument('--conv', type=str, default='genconv')
    parser.add_argument('--lappe', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--share_conv_weight', type=str, default='false')
    parser.add_argument('--share_lin_weight', type=str, default='false')
    parser.add_argument('--conv_sequence', type=str, default='cov')

    # loss related
    parser.add_argument('--loss', type=str, default='primal+objgap+constraint')
    parser.add_argument('--loss_weight_x', type=float, default=1.0)
    parser.add_argument('--loss_weight_obj', type=float, default=1.0)
    parser.add_argument('--loss_weight_cons', type=float, default=1.0)  # does not work
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])  # no big different
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args) #convert for wandb and yaml

    # safe hyperparameters in yaml file
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exist_runs = [d for d in os.listdir('logs') if d.startswith('exp')]
        log_folder_name = f'logs/exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="dariusweber"
               )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    



import argparse
from data.utils import args_set_bool
from ml_collections import ConfigDict
from tqdm import tqdm
from torch import optim
import numpy as np
from data.dataset import QPDataset
from torch.utils.data import DataLoader
from data.utils import collate_fn_ip
from models.hetero_gnn import TripartiteHeteroGNN
import wandb
import torch
import os
import yaml
import copy
from trainer import Trainer


def args_parser():
    # run.py --datapath DATA_TO_YOUR_INSTANCES --upper 1. --ipm_alpha 0.15 --weight_decay 1.2e-6 --batchsize 512
    # --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov
    # --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --conv genconv


    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', type=str, default='True')

    # ipm processing
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--ipm_alpha', type=float, default=0.9)

    # training dynamics
    parser.add_argument('--ckpt', type=str, default='true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--micro_batch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)  # must
    parser.add_argument('--use_norm', type=str, default='true')  # must

    # model related
    parser.add_argument('--conv', type=str, default='genconv')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--conv_sequence', type=str, default='cov')

    # loss related
    parser.add_argument('--loss', type=str, default='primal+objgap+constraint')
    parser.add_argument('--loss_weight_x', type=float, default=1.0)
    parser.add_argument('--loss_weight_obj', type=float, default=1.0)
    parser.add_argument('--loss_weight_cons', type=float, default=1.0)  # does not work
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])  # no big different
    return parser.parse_args()


if __name__ == '__main__':
    log_folder: str = "../../../../work/log1/darius.weber/logs"
    wandb_folder: str = "../../../../work/log1/darius.weber/wandb"
    #log_folder: str = "logs"
    #wandb_folder: str = "wandb"
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)  # convert for wandb and yaml

    # safe hyperparameters in yaml file
    if args.ckpt:
        if not os.path.isdir(log_folder):
            os.mkdir(log_folder)
        exist_runs = [d for d in os.listdir(log_folder) if d.startswith('exp')]
        log_folder_name: str = f'{log_folder}/exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="dariusweber_",
               dir=wandb_folder
               )
    # TODO: look if colloate_fn_ip is right
    dataset = QPDataset(args.datapath, args.ipm_steps,
                        extra_path=f'{args.ipm_steps}steps')

    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collate_fn_ip)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=collate_fn_ip)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=collate_fn_ip)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # best_val_losses = []
    best_val_objgap_mean = []
    best_val_consgap_mean = []
    # test_losses = []
    test_objdiff_mean = []
    test_objgap_mean = []
    test_consgap_mean = []

    for run in range(args.runs):
        if args.ckpt:
            os.mkdir(os.path.join(log_folder_name, f'run{run}'))
        model = TripartiteHeteroGNN(conv=args.conv,
                                    in_shape=2,
                                    hid_dim=args.hidden,
                                    num_conv_layers=args.num_conv_layers,
                                    num_pred_layers=args.num_pred_layers,
                                    num_mlp_layers=args.num_mlp_layers,
                                    dropout=args.dropout,
                                    use_norm=args.use_norm,
                                    conv_sequence=args.conv_sequence).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        trainer = Trainer(device,
                          args.loss,
                          args.losstype,
                          args.micro_batch,
                          min(args.ipm_steps, args.num_conv_layers),
                          args.ipm_alpha,
                          loss_weight={'primal': args.loss_weight_x,
                                       'objgap': args.loss_weight_obj,
                                       'constraint': args.loss_weight_cons})

        pbar = tqdm(range(args.epoch))  # progress bar
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)
            #print(train_loss)
            with torch.no_grad():
                val_loss = trainer.eval(val_loader, model, scheduler)
                #train metric
                train_gaps, train_diff, train_constraint_gap_eq, train_constraint_gap_uq = trainer.eval_metrics(train_loader, model)
                #print("train_diff", train_diff)
                train_mean_diff = train_diff[:, -1].mean().item()
                train_mean_gap = train_gaps[:, -1].mean().item()
                train_constraint_gap_eq_mean = train_constraint_gap_eq[:, -1].mean().item() if train_constraint_gap_eq.shape[0] != 0 else 0
                train_constraint_gap_uq_mean = train_constraint_gap_uq[:, -1].mean().item() if train_constraint_gap_uq.shape[0] != 0 else 0
                train_cons_gap_mean = train_constraint_gap_eq_mean + train_constraint_gap_uq_mean
                #val metric
                val_gaps, val_diff,val_constraint_gap_eq, val_constraint_gap_uq = trainer.eval_metrics(val_loader, model)
                
                # metric to cache the best model
                cur_mean_diff = val_diff[:, -1].mean().item()
                cur_mean_gap = val_gaps[:, -1].mean().item()
                val_constraint_gap_eq_mean = val_constraint_gap_eq[:, -1].mean().item() if val_constraint_gap_eq.shape[0] != 0 else 0
                val_constraint_gap_uq_mean = val_constraint_gap_uq[:, -1].mean().item() if val_constraint_gap_uq.shape[0] != 0 else 0
                cur_cons_gap_mean = val_constraint_gap_eq_mean + val_constraint_gap_uq_mean
                if scheduler is not None:
                    scheduler.step(cur_mean_gap)

                if trainer.best_val_objgap > cur_mean_gap:
                    trainer.patience = 0
                    trainer.best_val_objgap = cur_mean_gap
                    trainer.best_val_consgap = cur_cons_gap_mean
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'run{run}', 'best_model.pt'))
                else:
                    trainer.patience += 1

            if trainer.patience > args.patience:
                break
            pbar.set_postfix({'train_loss': train_loss,
                              'val_loss': val_loss,
                              'val_obj': cur_mean_gap,
                              'val_cons': cur_cons_gap_mean,
                              'lr': scheduler.optimizer.param_groups[0]["lr"]})
            log_dict = {'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_obj_diff_last_mean': train_mean_diff, #train metrics
                        'train_obj_gap_last_mean': train_mean_gap, #train metrics
                        'train_cons_gap_last_mean': train_cons_gap_mean, #train metrics
                        'train_hybrid_gap': train_mean_gap + train_cons_gap_mean, #train metrics
                        'val_obj_diff_last_mean': cur_mean_diff,
                        'val_obj_gap_last_mean': cur_mean_gap,
                        'val_cons_gap_last_mean': cur_cons_gap_mean,
                        'lr': scheduler.optimizer.param_groups[0]["lr"]}

            wandb.log(log_dict)
            # best_val_losses.append(trainer.best_val_loss)
        best_val_objgap_mean.append(trainer.best_val_objgap)
        best_val_consgap_mean.append(trainer.best_val_consgap)

        model.load_state_dict(best_model)
        with torch.no_grad():
            # test_loss = trainer.eval(test_loader, model, None)
            test_gaps, test_diff,test_cons_gap_eq, test_cons_gap_uq = trainer.eval_metrics(test_loader, model)
            
            test_cons_gap_eq_mean = test_cons_gap_eq[:, -1].mean().item() if val_constraint_gap_eq.shape[0] != 0 else 0
            test_cons_gap_uq_mean = test_cons_gap_uq[:, -1].mean().item() if val_constraint_gap_uq.shape[0] != 0 else 0
            #obj_gap, cons_gap_eq, cons_gap_uq
        # test_losses.append(test_loss)
        test_objdiff_mean.append(test_diff[:, -1].mean().item())
        test_objgap_mean.append(test_gaps[:, -1].mean().item())
        test_consgap_mean.append(test_cons_gap_eq_mean + test_cons_gap_uq_mean)
        wandb.log({'test_objdiff': test_objdiff_mean[-1]})
        wandb.log({'test_objgap': test_objgap_mean[-1]})
        wandb.log({'test_consgap': test_consgap_mean[-1]})

    wandb.log({
        # 'best_val_loss': np.mean(best_val_losses),
        'best_val_objgap': np.mean(best_val_objgap_mean),
        # 'test_loss_mean': np.mean(test_losses),
        # 'test_loss_std': np.std(test_losses),
        'test_objgap_mean': np.mean(test_objgap_mean),
        'test_objgap_std': np.std(test_objgap_mean),
        'test_consgap_mean': np.mean(test_consgap_mean),
        'test_consgap_std': np.std(test_consgap_mean),
        'test_hybrid_gap': np.mean(test_objgap_mean) + np.mean(test_consgap_mean),  # for the sweep
    })

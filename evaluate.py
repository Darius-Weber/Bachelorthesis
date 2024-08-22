import time
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from data.utils import args_set_bool
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from data.dataset import QPDataset
from torch_scatter import scatter
from models.hetero_gnn import TripartiteHeteroGNN
import wandb
from solver import qp
from data.utils import collate_fn_ip
from cvxopt import matrix as cvxopt_matrix
#python evaluate.py --datapath Quadratic_Programming_Datasets --modelpath logs/exp0/ --use_wandb False --ipm_steps 8 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --conv_sequence cov --conv ginconv
    

def args_parser():
    parser = argparse.ArgumentParser(description='Hyper params for evaluating GNN on your dataset')
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--modelpath', type=str, required=True, help='path to your pre-trained model')
    parser.add_argument('--conv', type=str, default='genconv')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--conv_sequence', type=str, default='cov')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--use_norm', type=str, default='true')

    
    
    
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--use_wandb', type=str, default='True')
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    
    return parser.parse_args()

def eval_metrics(device, data, vals, args):
    constraint_violation_eq = get_constraint_violation_eq(args, vals, data)
    constraint_violation_uq = get_constraint_violation_uq(args, vals, data)

    cons_gap_eq = (np.abs(constraint_violation_eq.detach().cpu().numpy()))
    cons_gap_uq = (np.abs(constraint_violation_uq.detach().cpu().numpy()))
    obj_gap = (np.abs(get_obj_metric(device, data, vals, args, hard_non_negative=False)[0].detach().cpu().numpy()))
    obj_diff = (np.abs(get_obj_metric(device, data, vals, args, hard_non_negative=False)[1].detach().cpu().numpy()))
    return obj_gap, obj_diff, cons_gap_eq, cons_gap_uq
        
def get_obj_metric(device, data, pred, args, hard_non_negative=False):
        pred = pred[:, -args.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.q[:, None] * pred  #q*x
        obj_pred_c = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        x_gt = data.gt_primals[:, -args.ipm_steps:]
        c_times_xgt = data.q[:, None] * x_gt
        obj_gt_c = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')

        slice = data._slice_dict['Q_val']
        num_nonzero_Q = slice[1:] - slice[:-1]
        Q_batch = torch.arange(len(num_nonzero_Q)).repeat_interleave(num_nonzero_Q).to(device)
        xQx_pred = scatter(pred[data.Q_col, :] * data.Q_val[:, None] * pred[data.Q_row, :], Q_batch, reduce='sum', dim=0)

        xQx_gt = scatter(x_gt[data.Q_col, :] * data.Q_val[:, None] * x_gt[data.Q_row, :], Q_batch, reduce='sum', dim=0)
        obj_pred = obj_pred_c + xQx_pred
        obj_gt = obj_gt_c + xQx_gt
        diff = obj_pred - obj_gt
        return (torch.log1p(torch.abs(diff)), diff)
    
def get_constraint_violation_eq(args, pred, data):
    """
    Ax - b
    :param pred:
    :param data:
    :return:
    """
    #EQUALITY CONSTRAINT VIOLTATION
    pred = pred[:, -args.ipm_steps:]
    Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
    constraint_gap = Ax - data.b[:, None]
    return constraint_gap

    
def get_constraint_violation_uq(args, pred, data):
    """
    Gx - h
    :param pred:
    :param data:
    :return:
    """
    #INEQUALITY CONSTRAINT VIOLTATION
    pred = pred[:, -args.ipm_steps:]
    Gx = scatter(pred[data.G_col, :] * data.G_val[:, None], data.G_row, reduce='sum', dim=0, dim_size=data.h[:, None].shape[0])
    constraint_gap = torch.relu(Gx - data.h[:, None])
    return constraint_gap

if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)
    wandb_folder: str = "../../../../work/log1/darius.weber/wandb"

    # Initialize Weights and Biases.
    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="dariusweber_",
               dir=wandb_folder
               )

    # Create dataset and DataLoader
    dataset = QPDataset(args.datapath, args.ipm_steps, extra_path=f'{args.ipm_steps}steps')
    
    dataloader = DataLoader(dataset[int(len(dataset) * 0.9):],
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=collate_fn_ip)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    qp.options['show_progress'] = False
    test_objdiff = []
    test_objgap = []
    test_consgap_eq = []
    test_consgap_uq = []
    pred_times = []

    
    test_objdiff_mean = []
    test_objgap_mean = []
    test_consgap_mean = []
    
    # Initialize the GNN
    gnn = TripartiteHeteroGNN(conv=args.conv,
                              in_shape=2,
                              hid_dim=args.hidden,
                              num_conv_layers=args.num_conv_layers,
                              num_pred_layers=args.num_pred_layers,
                              num_mlp_layers=args.num_mlp_layers,
                              dropout=args.dropout,
                              use_norm=args.use_norm,
                              conv_sequence=args.conv_sequence).to(device)
    counter = 3
    modelbool = False
    if (modelbool):
        for root, dirs, files in os.walk(args.modelpath):
            for ckpt in files:
                if ckpt.endswith('.pt'):
                    gnn.load_state_dict(torch.load(os.path.join(root, ckpt), map_location=device))
                    gnn.eval()
                    counter+=1
                    for data in tqdm(dataloader):
                        data = data.to(device)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        t1 = time.perf_counter()
                        pred = gnn(data)

                        if torch.cuda.is_available():
                            # Synchronize because cuda works asynchronous
                            # and the timing is not accurate without sync
                            torch.cuda.synchronize()

                        t2 = time.perf_counter()
                        pred_times.append(t2 - t1)
                        
                        test_gaps, test_diff,test_cons_gap_eq, test_cons_gap_uq = eval_metrics(device, data, pred, args)
                        test_objdiff.append(test_diff)
                        test_objgap.append(test_gaps)
                        test_consgap_eq.append(test_cons_gap_eq)
                        test_consgap_uq.append(test_cons_gap_uq)
                        
                    obj_diff = np.concatenate(test_objdiff, axis=0)    
                    obj_gap = np.concatenate(test_objgap, axis=0)
                    cons_gap_eq = np.concatenate(test_consgap_eq, axis=0)
                    cons_gap_uq = np.concatenate(test_consgap_uq, axis=0)
                    
                    test_cons_gap_eq_mean = cons_gap_eq[:, -1].mean().item() if cons_gap_eq.shape[0] != 0 else 0
                    test_cons_gap_uq_mean = cons_gap_uq[:, -1].mean().item() if cons_gap_uq.shape[0] != 0 else 0
                    test_objdiff_mean.append(obj_diff[:, -1].mean().item())
                    test_objgap_mean.append(obj_gap[:, -1].mean().item())
                    test_consgap_mean.append(test_cons_gap_eq_mean + test_cons_gap_uq_mean)

                    stat_dict = {
                        "obj_diff": test_objdiff_mean[-1],
                        "obj_gap": test_objgap_mean[-1],
                        "cons_gap": test_consgap_mean[-1],
                    }
                    wandb.log(stat_dict)
            
    solver_times = []
    
    #ensure same number of iterations
    for _ in range(counter):
        for data in tqdm(dataloader):
            data = data.to(device)

            # Create indices tensor from row_Q and Q_col
            indices_Q = torch.stack([data.Q_row, data.Q_col])

            # Create sparse tensor with the specified size
            sparse_tensor_Q = torch.sparse_coo_tensor(indices_Q, data.Q_val, size=(data.Q_num_row, data.Q_num_col))

            # Convert sparse tensor to dense tensor
            Q = sparse_tensor_Q.to_dense()
            
            # Create indices tensor from row_Q and Q_col
            indices_A = torch.stack([data.A_row, data.A_col])

            # Create sparse tensor with the specified size
            sparse_tensor_A = torch.sparse_coo_tensor(indices_A, data.A_val, size=(data.A_num_row, data.A_num_col))

            # Convert sparse tensor to dense tensor
            A = sparse_tensor_A.to_dense()
            
            # Create indices tensor from row_Q and Q_col
            indices_G = torch.stack([data.G_row, data.G_col])

            # Create sparse tensor with the specified size
            sparse_tensor = torch.sparse_coo_tensor(indices_G, data.G_val, size=(data.G_num_row, data.G_num_col))

            # Convert sparse tensor to dense tensor
            G = sparse_tensor.to_dense()
            
            q = data.q.unsqueeze(1)
            h = data.h.unsqueeze(1)
            b = data.b.unsqueeze(1)
            
            # Convert PyTorch tensors to NumPy arrays with float64 type
            Q_np = Q.cpu().numpy().astype(np.float64)
            q_np = q.cpu().numpy().astype(np.float64)
            G_np = G.cpu().numpy().astype(np.float64)
            h_np = h.cpu().numpy().astype(np.float64)
            A_np = A.cpu().numpy().astype(np.float64)
            b_np = b.cpu().numpy().astype(np.float64)

            # Convert NumPy arrays to cvxopt matrices
            Q_cvx = cvxopt_matrix(Q_np)
            q_cvx = cvxopt_matrix(q_np)
            G_cvx = cvxopt_matrix(G_np)
            h_cvx = cvxopt_matrix(h_np)
            A_cvx = cvxopt_matrix(A_np)
            b_cvx = cvxopt_matrix(b_np)
            
            if torch.cuda.is_available():
                # Synchronize because cuda works asynchronous
                # and the timing is not accurate without sync
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            result = qp.qp(Q_cvx, q_cvx,G_cvx,h_cvx, A_cvx, b_cvx, callback=lambda res: res)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            solver_times.append(t2 - t1)
    if modelbool:
        wandb.log({
            'test_objgap_mean': np.mean(test_objgap_mean),
            'test_objgap_std': np.std(test_objgap_mean),
            'test_objdiff_mean': np.mean(test_objdiff_mean),
            'test_objdiff_std': np.std(test_objdiff_mean),
            'test_consgap_mean': np.mean(test_consgap_mean),
            'test_consgap_std': np.std(test_consgap_mean),
            'test_hybrid_gap': np.mean(test_objgap_mean) + np.mean(test_consgap_mean),
            'test_hybrid_diffgap': np.mean(test_objdiff_mean) + np.mean(test_consgap_mean),
            'pred_times_mean': np.mean(pred_times),
            'pred_times_std' : np.std(pred_times),
            'solver_times_mean': np.mean(solver_times),
            'solver_times_std': np.std(solver_times)
        })
import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from torch_sparse import SparseTensor
import cvxopt
from cvxopt import solvers
from tqdm import tqdm


#https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
def generate_matrix(c, o):
    row1 = torch.arange(c).unsqueeze(0).repeat(o, 1).flatten()
    row2 = torch.arange(o).unsqueeze(1).repeat(1, c).flatten()
    matrix = torch.stack((row1, row2))
    return matrix


def swap_rows(tensor, idx1, idx2):
    """
    Swap two rows in a tensor.

    Parameters:
    tensor (torch.Tensor): The tensor where rows will be swapped
    idx1, idx2 (int, int): The indices of the rows to be swapped

    Returns:
    torch.Tensor: The tensor after swapping the rows
    """
    #TODO make it maybe more efficient
    t = tensor.clone()  # make a copy of the original tensor
    t[idx1], t[idx2] = tensor[idx2].clone(), tensor[idx1].clone()
    return t

class QPDataset(InMemoryDataset):

    def __init__(
            self,
            root: str,
            ipm_steps: int, #TODO: look if this works
            extra_path: str,
    ):
        self.ipm_steps = ipm_steps
        self.extra_path = extra_path
        super().__init__(root)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']  # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_' + self.extra_path)

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        data_list = []
        for i in range(num_instance_pkg):
            # load instance
            print(f"processing {i}th package, {num_instance_pkg} in total")
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                ip_pkgs = pickle.load(file)

            for ip_idx in tqdm(range(len(ip_pkgs))):
                (Q, q, G, h, A, b, S, sol, max_ipm_steps) = ip_pkgs[ip_idx]
                #TODO look if sparse works. Maybe not every matrix should be sparse
                sp_Q = SparseTensor.from_dense(Q, has_value=True)
                sp_G = SparseTensor.from_dense(G, has_value=True)
                sp_A = SparseTensor.from_dense(A, has_value=True)
                sp_S = SparseTensor.from_dense(S, has_value=True)

                row_Q = sp_Q.storage._row
                col_Q = sp_Q.storage._col
                val_Q = sp_Q.storage._value

                row_G = sp_G.storage._row
                col_G = sp_G.storage._col
                val_G = sp_G.storage._value

                row_A = sp_A.storage._row
                col_A = sp_A.storage._col
                val_A = sp_A.storage._value

                row_S = sp_S.storage._row
                col_S = sp_S.storage._col
                val_S = sp_S.storage._value

                #if self.using_ineq:
                #This tilde_mask could help while filtering or selecting specific elements/columns in tensor A based on this computed boolean mask.
                #    tilde_mask = torch.ones(row.shape, dtype=torch.bool)
                #else:
                #   tilde_mask = col < (A.shape[1] - A.shape[0])

                #c = c / (c.abs().max() + 1.e-10)  # does not change the result

                # merge inequality constrain matrix G with erquality constrain matrix A:
                qp_constraintmatrix = torch.vstack((G, A))
                qp_constraints = torch.vstack((h, b))

                # Size of number of constrains x number of objectives (+1 because of O1!)
                stackco = generate_matrix(qp_constraintmatrix.shape[0],
                                          S.shape[1] + 1)
                stackoc = swap_rows(stackco, 0, 1)
                oc_edge_attr = torch.cat((qp_constraints, torch.zeros(stackco.shape[1] - qp_constraintmatrix.shape[0],
                                                                      1)))  #h,b for O1 and rest 0 for O2 to On
                stackvo = generate_matrix(q.shape[0],
                                          S.shape[1] + 1)
                stackov = swap_rows(stackvo, 0, 1)
                ov_edge_attr = torch.cat((q, S.t().flatten().unsqueeze(1)))

                x_values = [iteration['x'] for iteration in sol['intermediate']]
                x = np.stack(x_values, axis=1)
                x = x.reshape(x.shape[0], -1) # should work!

                # padding is a matrix that repeats the last element of each row.
                # Need it because x must be the same size for all instances. As
                # ipm can have different iterations for each instance, we need to pad
                if (max_ipm_steps+1 - x.shape[1] > 0):
                    x = np.hstack((x, np.repeat(x[:, -1:], max_ipm_steps+1 - x.shape[1], axis=1)))

                # look that number of ipm steps given is not smaller than the actual number of steps:
                if (self.ipm_steps - x.shape[1]>0):
                    x = np.hstack((x, np.repeat(x[:, -1:], self.ipm_steps - x.shape[1], axis=1)))

                gt_primals = torch.from_numpy(x).to(torch.float)
                # Dual Solution for inequality constraints
                #z = np.array(sol['z'])

                # Dual Solution for equality constraints
                #y = np.array(sol['y'])
                # gt_slacks = np.array(sol['s'])
                stacked_tensor = torch.cat([
                    torch.cat([q.mean(0, keepdim=True), q.std(0, keepdim=True)], dim=0).squeeze().unsqueeze(0),
                    torch.cat([S.mean(0, keepdims=True), S.std(0, keepdims=True)], dim=0).T
                ], dim=0)
                #print("cons", torch.cat([qp_constraintmatrix.mean(1, keepdim=True),
                                          #qp_constraintmatrix.std(1, keepdim=True)], dim=1))
                #print(qp_constraintmatrix.shape)
                #print(A.shape)
                #print(G.shape)
                #print("vals", torch.cat([qp_constraintmatrix.mean(0, keepdim=True),
                                          #qp_constraintmatrix.std(0, keepdim=True)], dim=0).T)
                #print(S.shape)
                #print(qp_constraints.shape)
                #print("obj",stacked_tensor)
                #print("qp_constraints", qp_constraints)
                #print("S",S)
                #print("q",q)
                #print("qp_constrainmatrix", qp_constraintmatrix)

                #print("cons__to__obj_EDGE_index", stackco)
                #print("cons__to__obj_EDGE_attr", oc_edge_attr)

                #print("obj__to__cons_EDGE_index", stackoc)
                #print("obj__to__cons_EDGE_attr", oc_edge_attr)

                #print("obj__to__vals=_EDGE_index", stackov)
                #print("obj__to__vals=EDGE_attr", ov_edge_attr)

                #print("vals__to__obj_EDGE_index", stackvo)
                #print("vals__to__obj_EDGE_attr", ov_edge_attr)
                #print("Q",Q)
                # TODO evtl. consider SVD instead of mean and std
                data = HeteroData(
                    #node features
                    cons={'x': torch.cat([qp_constraintmatrix.mean(1, keepdim=True),
                                          qp_constraintmatrix.std(1, keepdim=True)], dim=1)}, #RIGHT
                    vals={'x': torch.cat([qp_constraintmatrix.mean(0, keepdim=True),
                                          qp_constraintmatrix.std(0, keepdim=True)], dim=0).T},#RIGHT
                    obj={'x': stacked_tensor},#RIGHT

                    #edges
                    cons__to__vals={'edge_index': torch.vstack(torch.where(qp_constraintmatrix)),#RIGHT
                                    #row and column indices where 0 values left out
                                    'edge_attr': qp_constraintmatrix[torch.where(qp_constraintmatrix)][:, None]},#RIGHT
                    #values from qp_constraintmatrix where 0 is left out
                    vals__to__cons={'edge_index': torch.vstack(torch.where(qp_constraintmatrix.T)),#RIGHT
                                    'edge_attr': qp_constraintmatrix.T[torch.where(qp_constraintmatrix.T)][:, None]},#RIGHT


                    # TODO make it more sparse for cons__to__obj and obj__to__cons
                    cons__to__obj={'edge_index': stackco, #RIGHT
                                   'edge_attr': oc_edge_attr}, #RIGHT
                    obj__to__cons={'edge_index': stackoc, #RIGHT
                                   'edge_attr': oc_edge_attr}, #RIGHT
                    obj__to__vals={'edge_index': stackov, #RIGHT
                                   'edge_attr': ov_edge_attr}, #RIGHT
                    vals__to__obj={'edge_index': stackvo, #RIGHT
                                   'edge_attr': ov_edge_attr}, #RIGHT
                    gt_primals=gt_primals,
                    # gt_duals=gt_duals,
                    # gt_slacks=gt_slacks,
                    obj_value=torch.tensor(sol['primal objective'], dtype=torch.float32), # changed
                    #obj_const=c,
                    q=q,
                    h=h,
                    b=b,
                    Q_row=row_Q,
                    Q_col=col_Q,
                    Q_val=val_Q,
                    Q_nnz=len(val_Q),
                    G_row=row_G,
                    G_col=col_G,
                    G_val=val_G,
                    G_nnz=len(val_G),
                    A_row=row_A,
                    A_col=col_A,
                    A_val=val_A,
                    A_nnz=len(val_A),
                    S_row=row_S,
                    S_col=col_S,
                    S_val=val_S,
                    S_nnz=len(val_S),
                    A_num_row=A.shape[0],
                    A_num_col=A.shape[1],
                    G_num_row=G.shape[0],
                    G_num_col=G.shape[1],
                    Q_num_row=Q.shape[0],
                    Q_num_col=Q.shape[1],
                    S_num_row=S.shape[0],
                    S_num_col=S.shape[1],
                    GA_num_row=qp_constraintmatrix.shape[0],
                    GA_num_col=qp_constraintmatrix.shape[1],
                    #A_tilde_mask=tilde_mask,
                    rhs=qp_constraints)

                data_list.append(data)
            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))

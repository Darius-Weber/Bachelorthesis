import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from torch_sparse import SparseTensor
from cvxopt import matrix as cvxopt_matrix
from solver import qp
from tqdm import tqdm


#https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
def remove_zero_columns(S):
    # Compute a boolean mask where each element is True if the corresponding column is not a zero column
    mask = torch.any(S != 0, dim=0)

    # Use the mask to select the non-zero columns
    S_reduced = S[:, mask]

    return S_reduced

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
                (Q_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx, S_cvx, max_ipm_steps) = ip_pkgs[ip_idx]
                # Solve the quadratic program
                qp.options['show_progress'] = False
                sol = qp.qp(Q_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx, callback=lambda res: res)
                Q = torch.from_numpy(np.array(Q_cvx)).to(torch.float)
                q = torch.from_numpy(np.array(q_cvx)).to(torch.float)
                G = torch.from_numpy(np.array(G_cvx)).to(torch.float)
                h = torch.from_numpy(np.array(h_cvx)).to(torch.float)
                A = torch.from_numpy(np.array(A_cvx)).to(torch.float)
                b = torch.from_numpy(np.array(b_cvx)).to(torch.float)
                S_dense = torch.from_numpy(np.array(S_cvx)).to(torch.float)
                S = remove_zero_columns(S_dense)
                # merge inequality constrain matrix G with equality constrain matrix A:
                qp_constraintmatrix = torch.vstack((A, G))
                qp_constraints = torch.vstack((b, h))

                sp_Q = SparseTensor.from_dense(Q, has_value=True)
                sp_G = SparseTensor.from_dense(G, has_value=True)
                sp_A = SparseTensor.from_dense(A, has_value=True)
                sp_S = SparseTensor.from_dense(S, has_value=True)
                sp_q = SparseTensor.from_dense(q, has_value=True)

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

                row_q = sp_q.storage._row
                col_q = sp_q.storage._col
                val_q = sp_q.storage._value

                #print(sol['intermediate'])
                x_values = [iteration['x'] for iteration in sol['intermediate']]
                x = np.stack(x_values, axis=1)
                x = x.reshape(x.shape[0], -1) # should work!
                #print("x", x)
                # padding is a matrix that repeats the last element of each row.
                # Need it because x must be the same size for all instances. As
                # ipm can have different iterations for each instance, we need to pad
                if (max_ipm_steps+1 - x.shape[1] > 0):
                    #print("TRUE1")
                    x = np.hstack((x, np.repeat(x[:, -1:], max_ipm_steps+1 - x.shape[1], axis=1)))

                # look that number of ipm steps given is not smaller than the actual number of steps:
                if (self.ipm_steps - x.shape[1]>0):
                    #print("TRUE2")
                    x = np.hstack((x, np.repeat(x[:, -1:], self.ipm_steps - x.shape[1], axis=1)))

                primal = x[:, -self.ipm_steps:]
                #print("primal", primal)
                # loses some precision
                gt_primals = torch.from_numpy(primal).to(torch.float)
                #print("Q",Q)
                #print("Q_shape", Q.shape)
                #print("c",q.squeeze(1))
                #print("c_shape", q.squeeze(1).shape)
                #print("h",h.squeeze(1))
                #print("h_shape", h.squeeze(1).shape)
                #print("b",b.squeeze(1))
                #print("b_shape", b.squeeze(1).shape)
                #print("qp_constraints", qp_constraints)
               #print("qp_constraints shape", qp_constraints.shape)
                # TODO evtl. consider SVD instead of mean and std
                data = HeteroData(
                    #node features
                    cons={'x': torch.cat([qp_constraintmatrix.mean(1, keepdim=True),
                                          qp_constraintmatrix.std(1, keepdim=True)], dim=1)},
                    vals={'x': torch.cat([qp_constraintmatrix.mean(0, keepdim=True),
                                          qp_constraintmatrix.std(0, keepdim=True)], dim=0).T},
                    obj={'x': torch.cat([torch.cat([q.mean(0, keepdim=True), q.std(0, keepdim=True)], dim=0).squeeze().unsqueeze(0),
                                torch.cat([S.mean(0, keepdims=True), S.std(0, keepdims=True)], dim=0).T], dim=0)},

                    #edges
                    cons__to__vals={'edge_index': torch.vstack(torch.where(qp_constraintmatrix)),
                                    'edge_attr': qp_constraintmatrix[torch.where(qp_constraintmatrix)][:, None]},
                    vals__to__cons={'edge_index': torch.vstack(torch.where(qp_constraintmatrix.T)),
                                    'edge_attr': qp_constraintmatrix.T[torch.where(qp_constraintmatrix.T)][:, None]},


                    # TODO make it more sparse for cons__to__obj and obj__to__cons
                    cons__to__obj={'edge_index': torch.stack([torch.arange(qp_constraints.shape[0]), torch.zeros(qp_constraints.shape[0])]).int(),
                                   'edge_attr': qp_constraints},
                    obj__to__cons={'edge_index': torch.stack([torch.zeros(qp_constraints.shape[0]), torch.arange(qp_constraints.shape[0])]).int(),
                                   'edge_attr': qp_constraints},
                    obj__to__vals={'edge_index':  torch.stack((torch.hstack((col_q, col_S+1)), torch.hstack((row_q, row_S)))),
                                   'edge_attr': torch.hstack((val_q, val_S))[:,None]},
                    vals__to__obj={'edge_index':  torch.stack((torch.hstack((row_q, row_S)),torch.hstack((col_q, col_S+1)))),
                                   'edge_attr': torch.hstack((val_q, val_S))[:,None]},
                    gt_primals=gt_primals,
                    obj_value=torch.tensor(sol['primal objective'], dtype=torch.float32),
                    q=q.squeeze(1),
                    h=h.squeeze(1),
                    b=b.squeeze(1),
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
                   # S_row=row_S,
                   # S_col=col_S,
                   # S_val=val_S,
                   # S_nnz=len(val_S),
                    A_num_row=A.shape[0],
                    A_num_col=A.shape[1],
                    G_num_row=G.shape[0],
                    G_num_col=G.shape[1],
                    Q_num_row=Q.shape[0],
                    Q_num_col=Q.shape[1],
                    #S_num_row=S.shape[0],
                    #S_num_col=S.shape[1],
                    #GA_num_row=qp_constraintmatrix.shape[0],
                    #GA_num_col=qp_constraintmatrix.shape[1],
                    rhs=qp_constraints.squeeze(1))

                data_list.append(data)
            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))

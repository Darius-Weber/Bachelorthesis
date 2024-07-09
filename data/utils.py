from typing import Dict, List
import torch
from torch_geometric.data import Batch, HeteroData

def args_set_bool(args: Dict):
    for k, v in args.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
    return args

def collate_fn_ip(graphs: List[HeteroData]):
    new_batch = Batch.from_data_list(graphs)
    #with open('debug.txt', 'a') as f:
    #    print("Batch",new_batch, file=f)
    #TODO look over this again. Get rid of some definitions

    #for Q
    row_bias_Q = torch.hstack([new_batch.Q_num_row.new_zeros(1), new_batch.Q_num_row[:-1]]).cumsum(dim=0)
    row_bias_Q = torch.repeat_interleave(row_bias_Q, new_batch.Q_nnz)
    new_batch.Q_row += row_bias_Q

    col_bias_Q = torch.hstack([new_batch.Q_num_col.new_zeros(1), new_batch.Q_num_col[:-1]]).cumsum(dim=0)
    col_bias_Q = torch.repeat_interleave(col_bias_Q, new_batch.Q_nnz)
    new_batch.Q_col += col_bias_Q

    #for G
    row_bias_G = torch.hstack([new_batch.G_num_row.new_zeros(1), new_batch.G_num_row[:-1]]).cumsum(dim=0)
    row_bias_G = torch.repeat_interleave(row_bias_G, new_batch.G_nnz)
    new_batch.G_row += row_bias_G

    col_bias_G = torch.hstack([new_batch.G_num_col.new_zeros(1), new_batch.G_num_col[:-1]]).cumsum(dim=0)
    col_bias_G = torch.repeat_interleave(col_bias_G, new_batch.G_nnz)
    new_batch.G_col += col_bias_G

    #for A
    row_bias_A = torch.hstack([new_batch.A_num_row.new_zeros(1), new_batch.A_num_row[:-1]]).cumsum(dim=0)
    row_bias_A = torch.repeat_interleave(row_bias_A, new_batch.A_nnz)
    new_batch.A_row += row_bias_A

    col_bias_A = torch.hstack([new_batch.A_num_col.new_zeros(1), new_batch.A_num_col[:-1]]).cumsum(dim=0)
    col_bias_A = torch.repeat_interleave(col_bias_A, new_batch.A_nnz)
    new_batch.A_col += col_bias_A

    # for S
    #row_bias_S = torch.hstack([new_batch.S_num_row.new_zeros(1), new_batch.S_num_row[:-1]]).cumsum(dim=0)
    #row_bias_S = torch.repeat_interleave(row_bias_S, new_batch.S_nnz)
    #new_batch.S_row += row_bias_S

    #col_bias_S = torch.hstack([new_batch.S_num_col.new_zeros(1), new_batch.S_num_col[:-1]]).cumsum(dim=0)
    #col_bias_S = torch.repeat_interleave(col_bias_S, new_batch.S_nnz)
    #new_batch.S_col += col_bias_S

    # Adjusting S adjacency matrix indices
    #row_bias_S = torch.hstack([new_batch.S_num_row.new_zeros(1), new_batch.S_num_row[:-1]]).cumsum(dim=0)
    #row_bias_S = torch.repeat_interleave(row_bias_S, new_batch.S_val.shape[0])
    #new_batch.S_row += row_bias_S

    #col_bias_S = torch.hstack([new_batch.S_num_col.new_zeros(1), new_batch.S_num_col[:-1]]).cumsum(dim=0)
    #col_bias_S = torch.repeat_interleave(col_bias_S, new_batch.S_val.shape[0])
    #new_batch.S_col += col_bias_S

    return new_batch

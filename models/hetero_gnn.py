import torch
import torch.nn.functional as F

from models.genconv import GENConv
from models.gcnconv import GCNConv
from models.ginconv import GINEConv
from torch_geometric.nn import MLP
from models.hetero_conv import HeteroConv


def strseq2rank(conv_sequence):
    # TODO look what is best
    if conv_sequence == 'parallel':
        c2v = v2c = v2o = o2v = c2o = o2c = 0
    elif conv_sequence == 'cvo':
        v2c = o2c = 0
        c2v = o2v = 1
        c2o = v2o = 2
    elif conv_sequence == 'vco':
        c2v = o2v = 0
        v2c = o2c = 1
        c2o = v2o = 2
    elif conv_sequence == 'ocv':
        c2o = v2o = 0
        v2c = o2c = 1
        c2v = o2v = 2
    elif conv_sequence == 'ovc':
        c2o = v2o = 0
        c2v = o2v = 1
        v2c = o2c = 2
    elif conv_sequence == 'voc':
        c2v = o2v = 0
        c2o = v2o = 1
        v2c = o2c = 2

    # used in paper
    elif conv_sequence == 'cov':
        v2c = o2c = 0
        c2o = v2o = 1
        c2v = o2v = 2
    else:
        raise ValueError
    return c2v, v2c, v2o, o2v, c2o, o2c


def get_conv_layer(conv: str,
                   in_dim: int,
                   hid_dim: int,
                   num_mlp_layers: int,
                   use_norm: bool):
    if conv.lower() == 'genconv':
        def get_conv():
            return GENConv(in_channels=in_dim,
                           out_channels=hid_dim,
                           num_layers=num_mlp_layers,
                           aggr='softmax',
                           msg_norm=use_norm,
                           learn_msg_scale=use_norm,
                           norm='batch' if use_norm else None,
                           bias=True,
                           edge_dim=1)
    elif conv.lower() == 'gcnconv':
        def get_conv():
            return GCNConv(in_dim=in_dim,
                           edge_dim=1,
                           hid_dim=hid_dim,
                           num_mlp_layers=num_mlp_layers,
                           norm='batch' if use_norm else None)
    elif conv.lower() == 'ginconv':
        def get_conv():
            return GINEConv(in_dim=in_dim,
                            edge_dim=1,
                            hid_dim=hid_dim,
                            num_mlp_layers=num_mlp_layers,
                            norm='batch' if use_norm else None)
    else:
        raise NotImplementedError

    return get_conv


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 in_shape,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 use_norm,
                 conv_sequence='cov'):  #In paper default cov
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        in_emb_dim = 2 * hid_dim

        # consists of three Multi-Layer Perceptron (MLP),
        # each of which is designed to encode one type of node in the graph:
        # 'vals', 'cons', and 'obj'
        # default norm is already batch:
        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(
                HeteroConv({
                    ('cons', 'to', 'vals'): (get_conv(), c2v),
                    ('vals', 'to', 'cons'): (get_conv(), v2c),
                    ('vals', 'to', 'obj'): (get_conv(), v2o),
                    ('obj', 'to', 'vals'): (get_conv(), o2v),
                    ('cons', 'to', 'obj'): (get_conv(), c2o),
                    ('obj', 'to', 'cons'): (get_conv(), o2c),
                }, aggr='cat'))

        self.pred_vals = torch.nn.ModuleList()
        self.pred_cons = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
            self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])  # from dataset apply MLP to 'cons', 'vals' or 'obj'
            x_dict[k] = x_emb  # safe applied LP in x_dict

        hiddens = []
        for i in range(self.num_layers):
            # h1 = x_dict
            # dictionary with HeteroConv to different edge types (hetero_conv is applied!)
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
            keys = h2.keys()
            hiddens.append(h2['vals'])
            h = {k: F.relu(h2[k]) for k in keys}

            # providing each neuron with the probability in self.dropout (default 0)
            # of being left out, or "dropped out", during the training process.
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
            x_dict = h
        vals = torch.cat([self.pred_vals[i](hiddens[i]) for i in range(self.num_layers)], dim=1)
        return vals

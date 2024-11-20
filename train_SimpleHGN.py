import copy
import time, datetime
import os.path as osp
import dgl, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dataloader import load_data

import dgl.function as Fn

from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear

from utils import set_seed, check_writable


def to_hetero_feat(h, type, name):
    """Feature convert API.

    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.

    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.

    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph
    """
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[torch.where(type == index)]

    return h_dict


class SimpleHGN(nn.Module):
    r"""
    This is a model SimpleHGN from `Are we really making much progress? Revisiting, benchmarking, and
    refining heterogeneous graph neural networks
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`__

    The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    hid_dim: int
        the output dimension
    num_classes: int
        the number of the output classes
    num_layers: int
        the number of layers we used in the computing
    heads: list
        the list of the number of heads in each layer
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    beta: float
        the hyperparameter used in edge residual
    ntypes: list
        the list of node type
    """

    def __init__(self, etypes, ntypes, in_dim, hid_dim, num_classes, category,
                 num_layers = 2, num_heads=4, feat_drop=0.2, attn_drop=0.6, negative_slope=0.2,
                 residual=True, beta=0.05):
        super(SimpleHGN, self).__init__()
        self.ntypes = ntypes
        edge_dim = hid_dim
        num_etypes = len(etypes)
        self.num_layers = num_layers
        self.category = category
        self.layers = nn.ModuleList()

        # input projection (no residual)
        self.layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dim,
                hid_dim // num_heads,
                num_heads,
                num_etypes,
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                activation=F.elu,
                beta=beta,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):  # noqa E741
            # due to multi-head, the in_dim = hid_dim * num_heads
            self.layers.append(
                SimpleHGNConv(
                    edge_dim,
                    hid_dim,
                    hid_dim // num_heads,
                    num_heads,
                    num_etypes,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    activation=F.elu,
                    beta=beta,
                )
            )
        self.layers.append(
            SimpleHGNConv(
                edge_dim,
                hid_dim,
                hid_dim // num_heads,
                num_heads,
                num_etypes,
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                activation=None,
                beta=beta,
            )
        )
        self.classifier = nn.Linear(hid_dim, num_classes)


    def forward(self, hg, h_dict):
        """
        The forward part of the SimpleHGN.

        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types

        Returns
        -------
        dict
            The embeddings after the output projection.
        """

        if hasattr(hg, 'ntypes'):
            # full graph training,
            with hg.local_scope():
                hg.ndata['h'] = h_dict
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
                for l in range(self.num_layers):  # noqa E741
                    h = self.layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                    h = h.flatten(1)
            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
        else:
            # for minibatch training, input h_dict is a tensor
            h = h_dict
            for layer, block in zip(self.layers, hg):
                h = layer(block, h, block.ndata['_TYPE']['_N'], block.edata['_TYPE'], presorted=False)
            h_dict = to_hetero_feat(h, block.ndata['_TYPE']['_N'][:block.num_dst_nodes()], self.ntypes)
        logits = self.classifier(h_dict[self.category])
        return logits

    @property
    def to_homo_flag(self):
        return True


class SimpleHGNConv(nn.Module):
    r"""
    The SimpleHGN convolution layer.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    """

    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(
            in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted=False):
        """
        The forward part of the SimpleHGNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        h: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``

        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1,
                                                                         self.num_heads, self.edge_dim)

        row = g.edges()[0]
        col = g.edges()[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)
        edge_attention = self.attn_drop(edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * \
                             (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
                         Fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)
            # h_prime = []
            # for i in range(self.num_heads):
            #     g.edata['alpha'] = edge_attention[:, i]
            #     g.srcdata.update({'emb': emb[i]})
            #     g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
            #                  Fn.sum('m', 'emb'))
            #     h_prime.append(g.ndata['emb'])
            # h_output = torch.cat(h_prime, dim=1)

        g.edata['alpha'] = edge_attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output


def evaluator(logits, y_true):
    y_pred = logits.argmax(1)
    return float(y_pred.eq(y_true).float().mean()) * 100


def run(seed):
    set_seed(seed)

    dataset = 'TMDB'

    output_dir = str(osp.join(
        'outputs',
        'transductive',
        dataset,
        'SimpleHGN',
        f'seed_{seed}',
    ))
    check_writable(output_dir, overwrite=False)
    out_path = osp.join(output_dir, 'out.pt')

    g, splits, generate_node_features = load_data(dataset)
    g = generate_node_features(g)
    print(g)
    category = g.category
    in_dim = g.ndata['feat'][category].shape[1]
    num_classes = max(g.ndata['label'][category]) + 1

    max_epoch = 50
    hid_dim = 128
    num_heads = 4
    lr, wd = 0.01, 0.
    feat_drop = 0.5
    attn_drop = 0.6
    num_layers = 2
    device = 'cuda:0'

    g = g.to(device)
    feats = g.ndata.pop('feat')
    labels = g.ndata.pop('label')[category]
    idx_train, idx_val, idx_test = splits

    model = SimpleHGN(g.etypes, g.ntypes, in_dim, hid_dim, num_classes, category,
                      num_layers=num_layers, num_heads=num_heads, feat_drop=feat_drop,
                      attn_drop=attn_drop, negative_slope=0.2, residual=True,
                      beta=0.05).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f'model size: {size_all_mb:.3f}MB')

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=wd)

    best_epoch = 0
    best_acc_val = 0
    training_start = time.time()
    for epoch in range(1, max_epoch + 1):
        epoch_start = time.time()
        model.train()
        logits = model(g, feats)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            model.eval()
            logits = model(g, feats)
            acc_val = evaluator(logits[idx_val], labels[idx_val])
            if best_acc_val < acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                best_state = copy.deepcopy(model.state_dict())
            time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
            print(
                f'Epoch {epoch:03d} | Loss={loss:.4f} | '
                f'Val={acc_val:.2f} | Time {time_taken}'
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(g, feats)
    torch.save(logits, out_path)

    acc_test = evaluator(logits[idx_test], labels[idx_test])
    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    print(f'Best Epoch {best_epoch:3d} | Test={acc_test :.2f} | Total time taken {time_taken}')
    return acc_test


if __name__ == '__main__':
    ACCs = []
    for seed in range(5):
        ACCs.append(run(seed))

    print(f'Test ACC: {np.mean(ACCs):.2f}+-{np.std(ACCs):.2f}')

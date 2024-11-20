import os.path as osp
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

import copy
import time, datetime

from torch import optim

from dataloader import load_data
from train_and_eval import evaluator
from utils import set_seed, check_writable


class ieHGCNConv(nn.Module):
    r"""
    The ieHGCN convolution layer.

    Parameters
    ----------
    in_size: int
        the input dimension
    out_size: int
        the output dimension
    attn_size: int
        the dimension of attention vector
    ntypes: list
        the node type list of a heterogeneous graph
    etypes: list
        the edge type list of a heterogeneous graph
    activation: str
        the activation function
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """

    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, activation=F.elu,
                 bias=False, batchnorm=False, dropout=0.0):
        super(ieHGCNConv, self).__init__()
        self.bias = bias
        self.batchnorm = batchnorm
        self.dropout = dropout
        node_size = {}
        for ntype in ntypes:
            node_size[ntype] = in_size
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = attn_size
        self.W_self = dglnn.HeteroLinear(node_size, out_size)
        self.W_al = dglnn.HeteroLinear(attn_vector, 1)
        self.W_ar = dglnn.HeteroLinear(attn_vector, 1)

        # self.conv = dglnn.HeteroGraphConv({
        #     etype: dglnn.GraphConv(in_size, out_size, norm = 'right', weight = True, bias = True)
        #     for etype in etypes
        # })
        self.in_size = in_size
        self.out_size = out_size
        self.attn_size = attn_size
        mods = {
            etype: dglnn.GraphConv(in_size, out_size, norm='right',
                                   weight=True, bias=True, allow_zero_in_degree=True)
            for etype in etypes
        }
        self.mods = nn.ModuleDict(mods)

        self.linear_q = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.linear_k = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})

        self.activation = activation
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_size)
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_size))
            nn.init.zeros_(self.h_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.

        Parameters
        ----------
        hg : object or list[block]
            the dgl heterogeneous graph or the list of blocks
        h_dict: dict
            the feature dict of different node types

        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        outputs = {ntype: [] for ntype in hg.dsttypes}
        if hg.is_block:
            src_inputs = h_dict
            dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_dict.items()}
        else:
            src_inputs = h_dict
            dst_inputs = h_dict
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            # formulas (2)-1
            dst_inputs = self.W_self(dst_inputs)
            query = {}
            key = {}
            attn = {}
            attention = {}

            # formulas (3)-1 and (3)-2
            for ntype in hg.dsttypes:
                query[ntype] = self.linear_q[ntype](dst_inputs[ntype])
                key[ntype] = self.linear_k[ntype](dst_inputs[ntype])
            # formulas (4)-1
            h_l = self.W_al(key)
            h_r = self.W_ar(query)
            for ntype in hg.dsttypes:
                attention[ntype] = F.elu(h_l[ntype] + h_r[ntype])
                attention[ntype] = attention[ntype].unsqueeze(0)

            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                # formulas (2)-2
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[srctype], dst_inputs[dsttype])
                )
                outputs[dsttype].append(dstdata)
                # formulas (3)-3
                attn[dsttype] = self.linear_k[dsttype](dstdata)
                # formulas (4)-2
                h_attn = self.W_al(attn)
                attn.clear()
                edge_attention = F.elu(h_attn[dsttype] + h_r[dsttype])
                attention[dsttype] = torch.cat((attention[dsttype], edge_attention.unsqueeze(0)))

            # formulas (5)
            for ntype in hg.dsttypes:
                attention[ntype] = F.softmax(attention[ntype], dim=0)

            # formulas (6)
            rst = {ntype: 0 for ntype in hg.dsttypes}
            for ntype, data in outputs.items():
                data = [dst_inputs[ntype]] + data
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]

            # h = self.conv(hg, hg.ndata['h'], aggregate = self.my_agg_func)

        def _apply(ntype, h):
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, feat) for ntype, feat in rst.items()}


class ieHGCN(nn.Module):
    """
    Parameters
    ----------
    num_layers: int
        the number of layers
    in_dim: int
        the input dimension
    hidden_dim: int
        the hidden dimension
    out_dim: int
        the output dimension
    attn_dim: int
        the dimension of attention vector
    ntypes: list
        the node type of a heterogeneous graph
    etypes: list
        the edge type of a heterogeneous graph
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """

    def __init__(self, category, num_layers, in_dim, hidden_dim, out_dim, attn_dim,
                 dropout, ntypes, etypes, bias=False, batchnorm=False):
        super(ieHGCN, self).__init__()
        self.category = category
        self.num_layers = num_layers
        self.activation = F.elu
        self.hgcn_layers = nn.ModuleList()
        self.hgcn_layers.append(
            ieHGCNConv(
                in_dim,
                hidden_dim,
                attn_dim,
                ntypes,
                etypes,
                self.activation,
                bias,
                batchnorm,
                dropout
            )
        )

        for i in range(1, num_layers - 1):
            self.hgcn_layers.append(
                ieHGCNConv(
                    hidden_dim,
                    hidden_dim,
                    attn_dim,
                    ntypes,
                    etypes,
                    self.activation,
                    bias,
                    batchnorm,
                    dropout
                )
            )
        self.hgcn_layers.append(
            ieHGCNConv(
                hidden_dim,
                hidden_dim,
                attn_dim,
                ntypes,
                etypes,
                None,
                False,
                False,
                0.0
            )
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCN.

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
        if hasattr(hg, "ntypes"):
            for l in range(self.num_layers):
                h_dict = self.hgcn_layers[l](hg, h_dict)
        else:
            for layer, block in zip(self.hgcn_layers, hg):
                h_dict = layer(block, h_dict)
        logits = self.classifier(h_dict[self.category])
        return logits


def run(seed):
    set_seed(seed)

    dataset = 'TMDB'
    output_dir = str(osp.join(
        'outputs',
        'transductive',
        dataset,
        'ieHGCN',
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
    lr, wd = 0.01, 0.
    dropout = 0.5
    num_layers = 2
    device = 'cuda:0'

    g = g.to(device)
    feats = g.ndata.pop('feat')
    labels = g.ndata.pop('label')[category]
    idx_train, idx_val, idx_test = splits

    model = ieHGCN(category, num_layers, in_dim, hid_dim, num_classes, hid_dim, dropout, g.ntypes, g.etypes).to(device)

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

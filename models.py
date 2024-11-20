import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, HeteroGraphConv


def get_model(model_type, etypes, in_dim, num_classes, conf):
    category = conf['category']
    hid_dim = conf['hid_dim']
    num_layers = conf['num_layers']
    dropout = conf['dropout']
    if 'RSAGE' in model_type:
        model = RSAGE(etypes, in_dim, hid_dim,
                      num_classes, category, num_layers, dropout)
    elif 'RGCN' in model_type:
        model = RGCN(etypes, in_dim, hid_dim,
                     num_classes, category, num_layers, dropout)
    elif 'RGAT' in model_type:
        model = RGAT(etypes, in_dim, hid_dim,
                     num_classes, category, num_layers, conf['num_heads'], dropout, attn_drop=conf['attn_drop'])
    elif 'MLP' in model_type:
        model = MLP(in_dim, hid_dim, num_classes, num_layers, dropout, norm_type=conf['norm_type'])
    else:
        model = RSAGE(etypes, in_dim, hid_dim,
                      num_classes, category, num_layers, dropout)

    return model


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, num_layers, dropout=0.2, norm_type='none'):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.classifier = nn.Linear(in_dim, num_classes)
        else:
            self.layers.append(nn.Linear(in_dim, hid_dim))
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hid_dim))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hid_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hid_dim, hid_dim))
                if self.norm_type == 'batch':
                    self.norms.append(nn.BatchNorm1d(hid_dim))
                elif self.norm_type == 'layer':
                    self.norms.append(nn.LayerNorm(hid_dim))

            self.classifier = nn.Linear(hid_dim, num_classes)

    def forward(self, _, feats, return_embeddings=False):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if self.norm_type != 'none':
                h = self.norms[l](h)
            h = F.relu(h)
            h = self.dropout(h)
        logits = self.classifier(h)

        if return_embeddings:
            return logits, h

        return logits


class RGCN(nn.Module):
    def __init__(self, etypes, in_dim, hid_dim, num_classes, category, num_layers=2, dropout=0.2):
        super().__init__()
        self.category = category
        self.layers = nn.ModuleList()

        self.layers.append(HeteroGraphConv({
            etype: GraphConv(in_dim, hid_dim, norm='both')
            for etype in etypes}, aggregate='mean'))
        for _ in range(num_layers - 1):
            self.layers.append(HeteroGraphConv({
                etype: GraphConv(hid_dim, hid_dim, norm='both')
                for etype in etypes}, aggregate='mean'))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hid_dim, num_classes)

    def forward(self, blocks, x, return_embeddings=False):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        logits = self.classifier(h[self.category])

        if return_embeddings:
            return logits, h

        return logits


class RSAGE(nn.Module):
    def __init__(self, etypes, in_dim, hid_dim, num_classes, category, num_layers=2, dropout=0.2):
        super().__init__()
        self.category = category
        self.layers = nn.ModuleList()

        self.layers.append(HeteroGraphConv({
            etype: SAGEConv(in_dim, hid_dim, 'gcn')
            for etype in etypes}, aggregate='mean'))
        for _ in range(num_layers - 1):
            self.layers.append(HeteroGraphConv({
                etype: SAGEConv(hid_dim, hid_dim, 'gcn')
                for etype in etypes}, aggregate='mean'))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hid_dim, num_classes)

    def forward(self, blocks, x, return_embeddings=False):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        logits = self.classifier(h[self.category])

        if return_embeddings:
            return logits, h

        return logits


class RGAT(nn.Module):
    def __init__(self, etypes, in_dim, hid_dim, num_classes, category,
                 num_layers=2, num_heads=4, dropout=0.2, attn_drop=0.6):
        super().__init__()
        self.category = category
        self.layers = nn.ModuleList()

        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_dim, hid_dim // num_heads, num_heads, attn_drop=attn_drop)
            for etype in etypes}))
        for _ in range(num_layers - 1):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(hid_dim, hid_dim // num_heads, num_heads, attn_drop=attn_drop)
                for etype in etypes}))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hid_dim, num_classes)

    def forward(self, blocks, x, return_embeddings=False):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        logits = self.classifier(h[self.category])

        if return_embeddings:
            return logits, h

        return logits

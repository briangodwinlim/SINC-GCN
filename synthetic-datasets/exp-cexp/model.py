import sys
sys.path.append('../..')

import dgl
import torch
from torch import nn
from torch_geometric.nn import EGConv
from models.utils import MLP, set_seed
from models.conv import SINCConv, SIRConv


class SINCModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, **kwargs):
        super(SINCModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            SINCConv(2 if i == 0 else hidden_dim, hidden_dim, hidden_dim, self.activation) 
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats


class GCNModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, **kwargs):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(2 if i == 0 else hidden_dim, hidden_dim, allow_zero_in_degree=True)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats


class SAGEModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, **kwargs):
        super(SAGEModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            dgl.nn.SAGEConv(2 if i == 0 else hidden_dim, hidden_dim, 'pool')
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats


class GATModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, num_heads=1, **kwargs):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            dgl.nn.GATv2Conv(2 if i == 0 else hidden_dim, hidden_dim, num_heads, allow_zero_in_degree=True, share_weights=True)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats).mean(dim=1)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats
    

class GINModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, mlp_layers=2, **kwargs):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            dgl.nn.GINConv(MLP(2 if i == 0 else hidden_dim, hidden_dim, hidden_dim, mlp_layers, 0, 'none', self.activation, True, False))
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats


class SIRModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, **kwargs):
        super(SIRModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            SIRConv(2 if i == 0 else hidden_dim, hidden_dim, hidden_dim, self.activation, agg_type='max') 
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats


class PNAModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, **kwargs):
        super(PNAModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            dgl.nn.PNAConv(2 if i == 0 else hidden_dim, hidden_dim, ['sum', 'max', 'std'], ['identity'], 1, residual=False)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats


class EGCSModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, **kwargs):
        super(EGCSModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()
        
        self.convs = nn.ModuleList([
            EGConv(2 if i == 0 else hidden_dim, hidden_dim, ['symnorm'], num_heads=1, num_bases=4, add_self_loops=False)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        edge_index = torch.stack(graphs.edges(), dim=0)
        
        for i in range(self.num_layers):
            feats = self.convs[i](feats, edge_index)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats


class EGCMModel(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, num_layers=1, dropout=0, **kwargs):
        super(EGCMModel, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.Tanh()

        self.convs = nn.ModuleList([
            EGConv(2 if i == 0 else hidden_dim, hidden_dim, ['sum', 'max', 'std'], num_heads=1, num_bases=4, add_self_loops=False)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.pooling = dgl.nn.MaxPooling()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(0.5), 
                                        nn.Linear(hidden_dim, 32), nn.ELU(), nn.Linear(32, output_dim))

    def forward(self, graphs, feats):
        edge_index = torch.stack(graphs.edges(), dim=0)
        
        for i in range(self.num_layers):
            feats = self.convs[i](feats, edge_index)
            feats = self.activation(feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.classifier(feats)
        return feats

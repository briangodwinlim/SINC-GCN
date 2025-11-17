import sys
sys.path.append('../..')

import dgl
import torch
import numpy as np
import networkx as nx
from dgl import function as fn
from dgl.data import DGLDataset
from models.utils import set_seed


class UniqueSignatureDataset(DGLDataset):
    def __init__(self, min_nodes=10, max_nodes=50, prob_edge=0.5, nfeat_range=10, num_samples=1000):
        super(UniqueSignatureDataset, self).__init__(name='UniqueSignatureDataset')
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.prob_edge = prob_edge
        self.nfeat_range = nfeat_range
        self.num_samples = num_samples
        
        self.data = []
        self.num_nodes = 0
        self.pos_nodes = 0
        for i in range(num_samples):
            self.data.append(self.generate_graph(i))

    def generate_graph(self, seed=0):
        set_seed(seed)
        
        num_nodes = np.random.randint(low=self.min_nodes, high=self.max_nodes + 1)
        g = nx.erdos_renyi_graph(n=num_nodes, p=self.prob_edge, directed=False)
        g = dgl.remove_self_loop(dgl.from_networkx(g))
        
        g.ndata['feat'] = torch.randint(low=0, high=self.nfeat_range, size=(num_nodes, 1)).float() - self.nfeat_range // 2
        g.update_all(fn.copy_u('feat', 'S'), fn.sum('S', 'S'))
        check_match = lambda edges: {'label': (edges.src['feat'] == edges.dst['S']).float()}
        g.update_all(check_match, fn.max('label', 'label'))
        
        self.num_nodes = self.num_nodes + num_nodes
        self.pos_nodes = self.pos_nodes + g.ndata['label'].sum().int().item()
        
        return g
    
    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return self.num_samples

import dgl
import torch
import pickle
import urllib.request
import torch.nn.functional as F
from dgl.data import DGLDataset


class PlanarSATDataset(DGLDataset):
    def __init__(self, dataset='EXP'):
        super(PlanarSATDataset, self).__init__(name='PlanarSATDataset')
        assert dataset in ['EXP', 'CEXP']
        
        self.graphs, self.labels = [], []
        with urllib.request.urlopen(f'https://github.com/ralphabb/GNN-RNI/raw/refs/heads/main/Data/{dataset}/raw/GRAPHSAT.pkl') as response:
            for data in pickle.load(response):
                data = data.__dict__
                graph = dgl.graph((data['edge_index'][0], data['edge_index'][1]), num_nodes=data['x'].shape[0])
                graph.ndata['feat'] = F.one_hot(data['x'][:, 0], num_classes=2).float()
                self.labels.append(data['y'].float())
                self.graphs.append(graph)
        self._generate_splits(splits=10)
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)
    
    def _generate_splits(self, splits=10):
        MODULO = 4
        MOD_THRESH = 1
    
        self.splits = []
        for idx in range(splits):
            train_mask = torch.zeros(len(self.graphs), dtype=torch.uint8)
            val_mask = torch.zeros(len(self.graphs), dtype=torch.uint8)
            test_mask = torch.zeros(len(self.graphs), dtype=torch.uint8)
            test_exp_mask = torch.zeros(len(self.graphs), dtype=torch.uint8)
            test_lrn_mask = torch.zeros(len(self.graphs), dtype=torch.uint8)
            
            n = len(self.graphs) // splits
            test_mask[idx * n : (idx + 1) * n] = 1
            test_exp_mask[[i for i in range(idx * n, (idx + 1) * n) if i % MODULO > MOD_THRESH]] = 1
            test_lrn_mask[[i for i in range(idx * n, (idx + 1) * n) if i % MODULO <= MOD_THRESH]] = 1
            
            n = (len(self.graphs) - test_mask.sum().item()) // splits
            val_mask[torch.nonzero(1 - test_mask, as_tuple=True)[0][idx * n : (idx + 1) * n]] = 1
            train_mask = (1 - val_mask - test_mask)
            
            assert torch.eq((train_mask + val_mask + test_mask), 1).all()
            self.splits.append((train_mask, val_mask, test_mask, test_lrn_mask, test_exp_mask))

import torch
from torch import nn
from dgl import function as fn
from dgl.utils import expand_as_pair


class SINCConv(nn.Module):
    r"""Soft-Isomorphic Neighborhood-Contextualized Graph Convolution Network (SINC-GCN)
    
    .. math::
        h_u^* = \sum_{v \in \mathcal{N}(u)} W_R ~ \sigma(W_Q h_u + W_K h_v + W_N \sum_{w \in \mathcal{N}(u)} h_w)

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden feature dimension
    output_dim : int
        Output feature dimension
    activation : a callable layer
        Activation function, the :math:`\sigma` in the formula
    dropout : float, optional
        Dropout rate for inner linear transformations, defaults to 0
    inner_bias : bool, optional
        Whether to learn an additive bias for inner linear transformations, defaults to True
    outer_bias : bool, optional
        Whether to learn an additive bias for outer linear transformations, defaults to True
    agg_type : str, optional
        Aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``), defaults to ``sum``
    neigh_agg_type : str, optional
        Neighborhood aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``), defaults to ``sum``
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout=0, inner_bias=True, outer_bias=True, agg_type='sum', neigh_agg_type='sum'):
        super(SINCConv, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(input_dim, hidden_dim, bias=inner_bias)
        self.linear_key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_neigh = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_relation = nn.Linear(hidden_dim, output_dim, bias=outer_bias)

        self._agg_type = agg_type
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
        
        self._neigh_agg_type = neigh_agg_type
        self._neigh_agg_func = fn.sum if neigh_agg_type == 'sym' else getattr(fn, neigh_agg_type)
    
    def neigh_message_func(self, edges):
        return {'m': edges.src['neigh_out_norm'] * edges.dst['neigh_in_norm'] * edges.src['en']}
    
    def message_func(self, edges):
        if self._agg_type in ['sum', 'mean', 'sym']:
            return {'m': edges.src['out_norm'] * edges.dst['in_norm'] * self.activation(edges.dst['eq'] + edges.src['ek'] + edges.dst['en'])}
        else:
            return {'m': self.linear_relation(self.activation(edges.dst['eq'] + edges.src['ek'] + edges.dst['en']))}
    
    def forward(self, graph, feat):
        with graph.local_scope():
            in_degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            out_degs = graph.out_degrees().float().clamp(min=1).to(graph.device)
            
            in_norm = torch.pow(in_degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            graph.ndata['in_norm'] = in_norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
            out_norm = torch.pow(out_degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            graph.ndata['out_norm'] = out_norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
            
            neigh_in_norm = torch.pow(in_degs, -0.5) if self._neigh_agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            graph.ndata['neigh_in_norm'] = neigh_in_norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
            neigh_out_norm = torch.pow(out_degs, -0.5) if self._neigh_agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            graph.ndata['neigh_out_norm'] = neigh_out_norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
 
            feat_key, feat_query = expand_as_pair(feat, graph)
            graph.ndata['en'] = feat_key
            graph.ndata['ek'] = self.dropout(self.linear_key(feat_key))
            graph.ndata['eq'] = self.dropout(self.linear_query(feat_query))
            graph.update_all(self.neigh_message_func, self._neigh_agg_func('m', 'en'))
            graph.ndata['en'] = self.dropout(self.linear_neigh(graph.ndata.pop('en')))

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            rst = self.linear_relation(rst) if self._agg_type in ['sum', 'mean', 'sym'] else rst
            
            return rst


class SIRConv(nn.Module):
    r"""Soft-Isomorphic Relational Graph Convolution Network (SIR-GCN)
    
    .. math::
        h_u^* = \sum_{v \in \mathcal{N}(u)} W_R ~ \sigma(W_Q h_u + W_K h_v)

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden feature dimension
    output_dim : int
        Output feature dimension
    activation : a callable layer
        Activation function, the :math:`\sigma` in the formula
    dropout : float, optional
        Dropout rate for inner linear transformations, defaults to 0
    inner_bias : bool, optional
        Whether to learn an additive bias for inner linear transformations, defaults to True
    outer_bias : bool, optional
        Whether to learn an additive bias for outer linear transformations, defaults to True
    agg_type : str, optional
        Aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``), defaults to ``sum``
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout=0, inner_bias=True, outer_bias=True, agg_type='sum', **kwargs):
        super(SIRConv, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(input_dim, hidden_dim, bias=inner_bias)
        self.linear_key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_relation = nn.Linear(hidden_dim, output_dim, bias=outer_bias)

        self._agg_type = agg_type
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
    
    def message_func(self, edges):
        if self._agg_type in ['sum', 'mean', 'sym']:
            return {'m': edges.src['out_norm'] * edges.dst['in_norm'] * self.activation(edges.dst['eq'] + edges.src['ek'])}
        else:
            return {'m': self.linear_relation(self.activation(edges.dst['eq'] + edges.src['ek']))}
        
    def forward(self, graph, feat):
        with graph.local_scope():
            in_degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            out_degs = graph.out_degrees().float().clamp(min=1).to(graph.device)
            
            in_norm = torch.pow(in_degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            graph.ndata['in_norm'] = in_norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
            out_norm = torch.pow(out_degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            graph.ndata['out_norm'] = out_norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
 
            feat_key, feat_query = expand_as_pair(feat, graph)
            graph.ndata['ek'] = self.dropout(self.linear_key(feat_key))
            graph.ndata['eq'] = self.dropout(self.linear_query(feat_query))

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            rst = self.linear_relation(rst) if self._agg_type in ['sum', 'mean', 'sym'] else rst
            
            return rst

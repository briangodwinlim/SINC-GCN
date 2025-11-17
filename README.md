# Soft-Isomorphic Neighborhood-Contextualized Graph Convolution Network (SINC-GCN)

This is the official repository for the paper [Enhancing Graph Representations with Neighborhood-Contextualized Message-Passing](https://arxiv.org/abs/2511.11046).

## Method

SINC-GCN, as an instance of the neighborhood-contextualized message-passing (NCMP) GNN variant, integrates contextualized messages (i.e., anisotropic and dynamic messages) and neighborhood-contextualization (i.e., functional dependence of the convolution operation on the entire set of neighborhood features) for a maximally expressive GNN architecture. It may be expressed as

$$\boldsymbol{h^*_u} = \bigoplus_{v \in \mathcal{N}(u)} \boldsymbol{W_R} ~ \sigma \left(\boldsymbol{W_Q} \boldsymbol{h_u} + \boldsymbol{W_K} \boldsymbol{h_v} + \bigotimes_{w \in \mathcal{N}(u)} \boldsymbol{W_N} \boldsymbol{h_w}\right),$$

where $\bigoplus$ and $\bigotimes$ are some permutation-invariant aggregators (e.g., sum, mean, symmetric mean, and max), $\sigma$ is a non-linear activation function, $\boldsymbol{W_R} \in \mathbb{R}^{d_\text{out} \times d_\text{hidden}}$, and $\boldsymbol{W_Q}, \boldsymbol{W_K}, \boldsymbol{W_N} \in \mathbb{R}^{d_\text{hidden} \times d_\text{in}}$. SINC-GCN has a computational complexity of 

$$\mathcal{O}\left(\left|\mathcal{V}\right| \times d_{\text{hidden}} \times d_{\text{in}} + \left|\mathcal{E}\right| \times d_{\text{hidden}} + \left|\mathcal{V}\right| \times d_{\text{out}} \times d_{\text{hidden}}\right)$$

by leveraging linearity, applying only an activation function along edges, and performing a two-step convolution constrained to the one-hop neighborhood receptive field. 

## Experiments

All experiments are conducted on a single Nvidia A800 (40GB) card using the [Deep Graph Library (DGL)](https://www.dgl.ai/) with [PyTorch](https://pytorch.org/) backend.

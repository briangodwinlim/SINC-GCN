# SINC-GCN/GATv2 implementation on ogbn-arxiv

## Preparations

### GIANT-XRT Embeddings

Follow the instructions at the [GIANT-XRT repository](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt) (not used).

```
wget https://archive.org/download/pecos-dataset/giant-xrt/ogbn-arxiv.tar.gz
tar -zxvf ogbn-arxiv.tar.gz
mv ogbn-arxiv dataset/ogbn_arxiv_xrt
rm -r ogbn-arxiv.tar.gz
```

## Experiments

### SINC-GCN (100k)

```
python train.py --nhidden 75 --nlayers 4 --dropout 0.3 --norm bn --residual --feat-dropout 0.1 --agg-type mean --neigh-agg-type sym --add-self-loop --add-reverse-edge --epochs 1000 --lr 1e-2 --wd 1e-3 --factor 0.5 --patience 50
```

## Summary

|     Model     |    Test Accuracy    | Parameters |
| :------------: | :------------------: | :--------: |
| SINC-GCN (100k) | 0.726581 ± 0.000927 |  103,915  |

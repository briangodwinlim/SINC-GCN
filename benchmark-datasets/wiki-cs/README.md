# SINC-GCN/GATv2 implementation on WikiCSDataset

## Experiments

### SINC-GCN

```
python train.py --use-amp --nhidden 50 --nlayers 4 --input-dropout 0.3 --edge-dropout 0 --dropout 0.1 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0.2 --feat-dropout 0 --agg-type mean --neigh-agg-type mean --epochs 200 --lr 1e-3 --wd 1e-1 --factor 0.5 --patience 10
```

## Summary

|  Model  |    Test Accuracy    | Parameters |
| :-----: | :------------------: | :--------: |
| SINC-GCN | 0.781705 ± 0.006764 |   101,510   |

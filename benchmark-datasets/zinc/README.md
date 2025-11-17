# SINC-GCN/GIN implementation on ZINCDataset

## Experiments

### SINC-GCN

```
python train.py --nworkers 1 --use-amp --nhidden 70 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 2 --readout-dropout 0 --readout-pooling sum --residual --resid-layers 1 --resid-dropout 0 --feat-dropout 0 --agg-type sym --neigh-agg-type mean --epochs 500 --batch-size 128 --lr 1e-3 --wd 1e-1 --factor 0.5 --patience 10
```

## Summary

|  Model  |      Test MAE      | Parameters |
| :-----: | :------------------: | :--------: |
| SINC-GCN | 0.256261 ± 0.005701 |   106,401   |

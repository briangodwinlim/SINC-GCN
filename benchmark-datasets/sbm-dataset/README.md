# SINC-GCN/GATv2 implementation on SBMDataset

## Experiments

### SINC-GCN (PATTERN)

```
python train.py --nworkers 1 --use-amp --nhidden 70 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0 --feat-dropout 0 --agg-type sym --neigh-agg-type mean --dataset PATTERN --epochs 200 --batch-size 128 --lr 1e-3 --wd 1e-1 --factor 0.5 --patience 10
```

### SINC-GCN (CLUSTER)

```
python train.py --nworkers 1 --use-amp --nhidden 70 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 1 --readout-dropout 0 --residual --resid-layers 1 --resid-dropout 0 --feat-dropout 0 --agg-type sym --neigh-agg-type mean --dataset CLUSTER --epochs 300 --batch-size 128 --lr 1e-3 --wd 1e-1 --factor 0.5 --patience 10
```

## Summary

|       Model       |    Test Accuracy    | Parameters |
| :---------------: | :------------------: | :--------: |
| SINC-GCN (PATTERN) | 0.857906 ± 0.000182 |  99,752  |
| SINC-GCN (CLUSTER) | 0.635069 ± 0.001545 |  100,316  |

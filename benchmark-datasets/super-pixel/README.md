# SINC-GCN/GIN implementation on SuperPixelDataset

## Experiments

### SINC-GCN (MNIST)

```
python train.py --nworkers 1 --use-amp --nhidden 75 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 3 --readout-dropout 0 --readout-pooling mean --residual --resid-layers 1 --resid-dropout 0.2 --feat-dropout 0.1 --agg-type max --neigh-agg-type mean --dataset MNIST --epochs 200 --batch-size 128 --lr 1e-3 --wd 1e-1 --factor 0.5 --patience 10
```

### SINC-GCN (CIFAR10)

```
python train.py --nworkers 1 --use-amp --nhidden 75 --nlayers 4 --input-dropout 0 --edge-dropout 0 --dropout 0 --norm bn --readout-layers 3 --readout-dropout 0 --readout-pooling mean --residual --resid-layers 1 --resid-dropout 0.2 --feat-dropout 0.1 --agg-type max --neigh-agg-type mean --dataset CIFAR10 --epochs 200 --batch-size 128 --lr 1e-3 --wd 1e-1 --factor 0.5 --patience 10
```

## Summary

|       Model       |    Test Accuracy    | Parameters |
| :---------------: | :------------------: | :--------: |
|  SINC-GCN (MNIST)  | 0.982800 ± 0.000522 |   104,560   |
| SINC-GCN (CIFAR10) | 0.733720 ± 0.004072 |  105,160  |

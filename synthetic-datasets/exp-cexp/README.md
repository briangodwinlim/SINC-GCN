# SINC-GCN/GCN/GraphSAGE/GATv2/GIN/SIR-GCN/PNA/EGC-S/EGC-M implementation on PlanarSAT

## Experiments

### SINC-GCN

```
python train.py --nworkers 1 --model SINC --nhidden 64 --nlayers 8 --dataset EXP --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### GCN

```
python train.py --nworkers 1 --model GCN --nhidden 64 --nlayers 8 --dataset EXP --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### GraphSAGE

```
python train.py --nworkers 1 --model SAGE --nhidden 64 --nlayers 8 --dataset EXP --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### GAT

```
python train.py --nworkers 1 --model GAT --nhidden 64 --nlayers 8 --dataset EXP --nheads 1 --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### GIN

```
python train.py --nworkers 1 --model GIN --nhidden 64 --nlayers 8 --dataset EXP --nlayers-mlp 2 --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### SIR-GCN

```
python train.py --nworkers 1 --model SIR --nhidden 64 --nlayers 8 --dataset EXP --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### PNA

```
python train.py --nworkers 1 --model PNA --nhidden 64 --nlayers 8 --dataset EXP --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### EGC-S

```
python -u train.py --nworkers 1 --model EGC-S --nhidden 64 --nlayers 8 --dataset EXP --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

### EGC-M

```
python -u train.py --nworkers 1 --model EGC-M --nhidden 64 --nlayers 8 --dataset EXP --epochs 500 --batch-size 20 --lr 1e-3 --factor 0.7 --patience 5
```

## Summary

|   Model   |    $\text{EXP}$    |  $\text{CORRUPT}$  | $\overline{\text{EXP}}$ |   $\text{CEXP}$   | Parameters |
| :-------: | :------------------: | :------------------: | :-----------------------: | :------------------: | :--------: |
| SINC-GCN | 0.500000 ± 0.000000 | 0.855000 ± 0.206162 |   0.500000 ± 0.000000   | 0.677500 ± 0.103081 |  126,465  |
|    GCN    | 0.500000 ± 0.000000 | 0.645000 ± 0.031667 |   0.500000 ± 0.000000   | 0.572500 ± 0.015833 |   35,585   |
| GraphSAGE | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |   0.500000 ± 0.000000   | 0.500000 ± 0.000000 |   93,511   |
|    GAT    | 0.500000 ± 0.000000 | 0.611667 ± 0.124733 |   0.500000 ± 0.000000   | 0.555833 ± 0.062367 |   36,097   |
|    GIN    | 0.500000 ± 0.000000 | 0.533333 ± 0.079232 |   0.500000 ± 0.000000   | 0.516667 ± 0.039616 |   68,865   |
|  SIR-GCN  | 0.500000 ± 0.000000 | 0.518333 ± 0.055000 |   0.500000 ± 0.000000   | 0.509167 ± 0.027500 |   97,665   |
|    PNA    | 0.500000 ± 0.000000 | 1.000000 ± 0.000000 |   0.500000 ± 0.000000   | 0.750000 ± 0.000000 |  214,091  |
|   EGC-S   | 0.500000 ± 0.000000 | 0.500000 ± 0.000000 |   0.500000 ± 0.000000   | 0.500000 ± 0.000000 |  123,817  |
|   EGC-M   | 0.500000 ± 0.000000 | 0.658333 ± 0.220637 |   0.500000 ± 0.000000   | 0.579167 ± 0.110318 |  127,481  |

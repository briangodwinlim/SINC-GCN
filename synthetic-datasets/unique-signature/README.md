# SINC-GCN/GCN/GraphSAGE/GATv2/GIN/SIR-GCN/PNA/EGC-S/EGC-M implementation on UniqueSignature

## Experiments

### SINC-GCN ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python train.py --nworkers 1 --model SINC --nhidden 16 --nlayers 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GCN ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python train.py --nworkers 1 --model GCN --nhidden 16 --nlayers 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GraphSAGE ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python train.py --nworkers 1 --model SAGE --nhidden 16 --nlayers 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GAT ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python train.py --nworkers 1 --model GAT --nhidden 16 --nlayers 1 --nheads 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### GIN ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python train.py --nworkers 1 --model GIN --nhidden 16 --nlayers 1 --nlayers-mlp 2 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### SIR-GCN ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python train.py --nworkers 1 --model SIR --nhidden 16 --nlayers 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### PNA ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python train.py --nworkers 1 --model PNA --nhidden 16 --nlayers 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### EGC-S ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python -u train.py --nworkers 1 --model EGC-S --nhidden 16 --nlayers 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

### EGC-M ($p_\text{edge} = \texttt{prob}$, $x_u \in \\{-\frac{\texttt{nfeat} - 1}{2}, \ldots, \frac{\texttt{nfeat} - 1}{2}\\}$)
```
python -u train.py --nworkers 1 --model EGC-M --nhidden 16 --nlayers 1 --min-nodes 30 --max-nodes 70 --prob-edge $prob --nfeat-range $nfeat --epochs 500 --batch-size 256 --lr 1e-3 --factor 0.5 --patience 10
```

## Summary

|   Model   | `prob` | `nfeat` |   Positive Class |  Test Balanced Accuracy  |   Parameters |
|:---------:|-------:|--------:|-----------------:|:------------------------:|-------------:|
| SINC-GCN  | 0.3000 |       3 |           0.3710 |   1.000000 ± 0.000000    |          352 |
| SINC-GCN  | 0.3000 |       5 |           0.3457 |   0.999245 ± 0.000836    |          352 |
| SINC-GCN  | 0.3000 |       7 |           0.3190 |   0.995926 ± 0.005433    |          352 |
| SINC-GCN  | 0.5000 |       3 |           0.2887 |   1.000000 ± 0.000000    |          352 |
| SINC-GCN  | 0.5000 |       5 |           0.2810 |   0.999061 ± 0.000204    |          352 |
| SINC-GCN  | 0.5000 |       7 |           0.2696 |   0.995695 ± 0.000303    |          352 |
| SINC-GCN  | 0.7000 |       3 |           0.2461 |   1.000000 ± 0.000000    |          352 |
| SINC-GCN  | 0.7000 |       5 |           0.2431 |   0.999974 ± 0.000000    |          352 |
| SINC-GCN  | 0.7000 |       7 |           0.2349 |   0.998690 ± 0.000019    |          352 |
|    GCN    | 0.3000 |       3 |           0.3710 |   0.500000 ± 0.000000    |           48 |
|    GCN    | 0.3000 |       5 |           0.3457 |   0.539572 ± 0.118715    |           48 |
|    GCN    | 0.3000 |       7 |           0.3190 |   0.538068 ± 0.114205    |           48 |
|    GCN    | 0.5000 |       3 |           0.2887 |   0.543475 ± 0.130424    |           48 |
|    GCN    | 0.5000 |       5 |           0.2810 |   0.626042 ± 0.192533    |           48 |
|    GCN    | 0.5000 |       7 |           0.2696 |   0.666073 ± 0.203397    |           48 |
|    GCN    | 0.7000 |       3 |           0.2461 |   0.589276 ± 0.178553    |           48 |
|    GCN    | 0.7000 |       5 |           0.2431 |   0.586235 ± 0.172469    |           48 |
|    GCN    | 0.7000 |       7 |           0.2349 |   0.671110 ± 0.209567    |           48 |
| GraphSAGE | 0.3000 |       3 |           0.3710 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.3000 |       5 |           0.3457 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.3000 |       7 |           0.3190 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.5000 |       3 |           0.2887 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.5000 |       5 |           0.2810 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.5000 |       7 |           0.2696 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.7000 |       3 |           0.2461 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.7000 |       5 |           0.2431 |   0.500000 ± 0.000000    |           66 |
| GraphSAGE | 0.7000 |       7 |           0.2349 |   0.500000 ± 0.000000    |           66 |
|    GAT    | 0.3000 |       3 |           0.3710 |   0.837841 ± 0.168924    |           64 |
|    GAT    | 0.3000 |       5 |           0.3457 |   0.621967 ± 0.186309    |           64 |
|    GAT    | 0.3000 |       7 |           0.3190 |   0.655412 ± 0.190341    |           64 |
|    GAT    | 0.5000 |       3 |           0.2887 |   0.810389 ± 0.203198    |           64 |
|    GAT    | 0.5000 |       5 |           0.2810 |   0.673882 ± 0.207529    |           64 |
|    GAT    | 0.5000 |       7 |           0.2696 |   0.639540 ± 0.190053    |           64 |
|    GAT    | 0.7000 |       3 |           0.2461 |   0.726848 ± 0.226848    |           64 |
|    GAT    | 0.7000 |       5 |           0.2431 |   0.543613 ± 0.130838    |           64 |
|    GAT    | 0.7000 |       7 |           0.2349 |   0.500000 ± 0.000000    |           64 |
|    GIN    | 0.3000 |       3 |           0.3710 |   0.835910 ± 0.000000    |          320 |
|    GIN    | 0.3000 |       5 |           0.3457 |   0.816901 ± 0.000000    |          320 |
|    GIN    | 0.3000 |       7 |           0.3190 |   0.803817 ± 0.000000    |          320 |
|    GIN    | 0.5000 |       3 |           0.2887 |   0.850106 ± 0.000000    |          320 |
|    GIN    | 0.5000 |       5 |           0.2810 |   0.836853 ± 0.000000    |          320 |
|    GIN    | 0.5000 |       7 |           0.2696 |   0.835118 ± 0.000000    |          320 |
|    GIN    | 0.7000 |       3 |           0.2461 |   0.855652 ± 0.000000    |          320 |
|    GIN    | 0.7000 |       5 |           0.2431 |   0.846908 ± 0.000000    |          320 |
|    GIN    | 0.7000 |       7 |           0.2349 |   0.840679 ± 0.000000    |          320 |
|  SIR-GCN  | 0.3000 |       3 |           0.3710 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.3000 |       5 |           0.3457 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.3000 |       7 |           0.3190 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.5000 |       3 |           0.2887 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.5000 |       5 |           0.2810 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.5000 |       7 |           0.2696 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.7000 |       3 |           0.2461 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.7000 |       5 |           0.2431 |   0.500000 ± 0.000000    |          336 |
|  SIR-GCN  | 0.7000 |       7 |           0.2349 |   0.500000 ± 0.000000    |          336 |
|    PNA    | 0.3000 |       3 |           0.3710 |   0.999560 ± 0.000263    |          403 |
|    PNA    | 0.3000 |       5 |           0.3457 |   0.991631 ± 0.000073    |          403 |
|    PNA    | 0.3000 |       7 |           0.3190 |   0.978051 ± 0.000151    |          403 |
|    PNA    | 0.5000 |       3 |           0.2887 |   1.000000 ± 0.000000    |          403 |
|    PNA    | 0.5000 |       5 |           0.2810 |   0.999062 ± 0.000017    |          403 |
|    PNA    | 0.5000 |       7 |           0.2696 |   0.995721 ± 0.000077    |          403 |
|    PNA    | 0.7000 |       3 |           0.2461 |   1.000000 ± 0.000000    |          403 |
|    PNA    | 0.7000 |       5 |           0.2431 |   0.999953 ± 0.000042    |          403 |
|    PNA    | 0.7000 |       7 |           0.2349 |   0.998744 ± 0.000005    |          403 |
|   EGC-S   | 0.3000 |       3 |           0.3700 |   0.500000 ± 0.000000    |          104 |
|   EGC-S   | 0.3000 |       5 |           0.3500 |   0.578651 ± 0.157303    |          104 |
|   EGC-S   | 0.3000 |       7 |           0.3200 |   0.504556 ± 0.013576    |          104 |
|   EGC-S   | 0.5000 |       3 |           0.2900 |   0.535192 ± 0.105576    |          104 |
|   EGC-S   | 0.5000 |       5 |           0.2800 |   0.542026 ± 0.126078    |          104 |
|   EGC-S   | 0.5000 |       7 |           0.2700 |   0.541498 ± 0.124493    |          104 |
|   EGC-S   | 0.7000 |       3 |           0.2500 |   0.500000 ± 0.000000    |          104 |
|   EGC-S   | 0.7000 |       5 |           0.2400 |   0.500000 ± 0.000000    |          104 |
|   EGC-S   | 0.7000 |       7 |           0.2300 |   0.500000 ± 0.000000    |          104 |
|   EGC-M   | 0.3000 |       3 |           0.3700 |   0.998889 ± 0.000133    |          120 |
|   EGC-M   | 0.3000 |       5 |           0.3500 |   0.972478 ± 0.059390    |          120 |
|   EGC-M   | 0.3000 |       7 |           0.3200 |   0.938930 ± 0.081844    |          120 |
|   EGC-M   | 0.5000 |       3 |           0.2900 |   1.000000 ± 0.000000    |          120 |
|   EGC-M   | 0.5000 |       5 |           0.2800 |   0.959000 ± 0.080091    |          120 |
|   EGC-M   | 0.5000 |       7 |           0.2700 |   0.932584 ± 0.096136    |          120 |
|   EGC-M   | 0.7000 |       3 |           0.2500 |   0.983285 ± 0.050146    |          120 |
|   EGC-M   | 0.7000 |       5 |           0.2400 |   0.960064 ± 0.079551    |          120 |
|   EGC-M   | 0.7000 |       7 |           0.2300 |   0.955736 ± 0.085183    |          120 |
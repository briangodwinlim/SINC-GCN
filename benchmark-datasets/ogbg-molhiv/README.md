# SINC-GCN/GIN implementation on ogbg-molhiv

## Preparations

### Molecular Fingerprints

Run the script below to generate molecular fingerprints (not used).

```
python fingerprint.py --morgan --maccs --rdkit --save
```

## Experiments

### SINC-GCN (100k)

```
python train.py --nworkers 1 --nhidden 75 --nlayers 4 --input-dropout 0.2 --norm bn --readout-pooling mean --residual --feat-dropout 0.4 --agg-type mean --neigh-agg-type sym --epochs 100 --batch-size 64 --lr 1e-3 --wd 1e-1 --factor 0.5 --patience 10
```

## Summary

|     Model     |     Test ROC-AUC     | Parameters |
| :------------: | :------------------: | :--------: |
| SINC-GCN (100k) | 0.785019 ± 0.012313 |  107,875  |

import torch
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.data import Subset
from dgl.dataloading import GraphDataLoader

from data import PlanarSATDataset
from model import SINCModel, GCNModel, SAGEModel, GATModel, GINModel, SIRModel, PNAModel, EGCSModel, EGCMModel, set_seed

loss_fn = nn.BCEWithLogitsLoss()
acc_fn = lambda logits, labels: torch.mean((logits > 0).float() == labels, dtype=torch.float32)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # PyTorch 2.4.0 deprecation warnings


def train(model, train_loader, device, optimizer, scaler):
    model.train()

    total_loss, total = 0, 0
    for graphs, labels in train_loader:
        graphs = graphs.to(device)
        labels = labels.to(device)
        feats = graphs.ndata.pop('feat')

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            logits = model(graphs, feats)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
    
    return total_loss / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    
    total_loss, total_acc, total = 0, 0, 0
    for graphs, labels in dataloader:
        graphs = graphs.to(device)
        labels = labels.to(device)
        feats = graphs.ndata.pop('feat')

        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            logits = model(graphs, feats)
            loss = loss_fn(logits, labels)
            acc = acc_fn(logits, labels)
        
        total = total + labels.shape[0]
        total_acc = total_acc + acc.item() * labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
        
    return total_loss / total, total_acc / total


def run(model, train_loader, val_loader, test_loader, test_lrn_loader, test_exp_loader, device, args):
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    
    for epoch in range(args.epochs):
        loss = train(model, train_loader, device, optimizer, scaler)
        loss, acc = evaluate(model, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        _, test_acc = evaluate(model, test_loader, device)
        _, test_lrn_acc = evaluate(model, test_lrn_loader, device)
        _, test_exp_acc = evaluate(model, test_exp_loader, device)
        scheduler.step(val_loss)

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | acc: {acc:.4f} | val_acc: {val_acc:.4f} | '
                  f'test_acc: {test_acc:.4f} | test_lrn_acc: {test_lrn_acc:.4f} | test_exp_acc: {test_exp_acc:.4f}')

    return acc, val_acc, test_acc, test_lrn_acc, test_exp_acc


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SINC-GCN/GCN/GraphSAGE/GATv2/GIN/PNA/SIR-GCN implementation on PlanarSAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    argparser.add_argument('--nworkers', type=int, default=0, help='number of workers')
    argparser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')
    
    argparser.add_argument('--model', type=str, default='SINC', help='model name', choices=['SINC', 'GCN', 'SAGE', 'GAT', 'GIN', 'SIR', 'PNA', 'EGC-S', 'EGC-M'])
    argparser.add_argument('--nhidden', type=int, default=64, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=8, help='number of graph convolution layers')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--nheads', type=int, default=1, help='number of attention heads for GAT')
    argparser.add_argument('--nlayers-mlp', type=int, default=2, help='number of MLP layers for GIN')
    
    argparser.add_argument('--dataset', type=str, default='EXP', help='name of dataset', choices=['EXP', 'CEXP'])
    
    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=20, help='batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--factor', type=float, default=0.7, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=5, help='patience for learning rate decay')
    argparser.add_argument('--min-lr', type=float, default=0, help='minimum learning rate')
    
    argparser.add_argument('--nruns', type=int, default=10, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()

    train_accs, val_accs, test_accs, test_lrn_accs, test_exp_accs = [], [], [], [], []
    for i in range(args.nruns):
        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset = PlanarSATDataset(dataset=args.dataset)
        
        # Seed
        set_seed(args.seed + i)
        
        train_mask, val_mask, test_mask, test_lrn_mask, test_exp_mask = dataset.splits[i]
        train_loader = GraphDataLoader(Subset(dataset, torch.nonzero(train_mask).squeeze()), shuffle=True, batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)
        val_loader = GraphDataLoader(Subset(dataset, torch.nonzero(val_mask).squeeze()), shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)
        test_loader = GraphDataLoader(Subset(dataset, torch.nonzero(test_mask).squeeze()), shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)
        test_lrn_loader = GraphDataLoader(Subset(dataset, torch.nonzero(test_lrn_mask).squeeze()), shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)
        test_exp_loader = GraphDataLoader(Subset(dataset, torch.nonzero(test_exp_mask).squeeze()), shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)

        # Load model
        Model = {'SINC': SINCModel, 'GCN': GCNModel, 'SAGE': SAGEModel, 'GAT': GATModel, 'GIN': GINModel, 'SIR': SIRModel, 'PNA': PNAModel, 'EGC-S': EGCSModel, 'EGC-M': EGCMModel}
        model = Model[args.model](args.nhidden, 1, args.nlayers, args.dropout, 
                                  num_heads=args.nheads, mlp_layers=args.nlayers_mlp).to(device)
        summary(model)

        # Training
        train_acc, val_acc, test_acc, test_lrn_acc, test_exp_acc = run(model, train_loader, val_loader, test_loader, test_lrn_loader, test_exp_loader, device, args)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        test_lrn_accs.append(test_lrn_acc)
        test_exp_accs.append(test_exp_acc)
        
    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Train accuracy: {train_accs}')
    print(f'Val accuracy: {val_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Test learn accuracy: {test_lrn_accs}')
    print(f'Test expressive accuracy: {test_exp_accs}')
    print(f'Average train accuracy: {np.mean(train_accs):.6f} ± {np.std(train_accs):.6f}')
    print(f'Average val accuracy: {np.mean(val_accs):.6f} ± {np.std(val_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')
    print(f'Average test learn accuracy: {np.mean(test_lrn_accs):.6f} ± {np.std(test_lrn_accs):.6f}')
    print(f'Average test expressive accuracy: {np.mean(test_exp_accs):.6f} ± {np.std(test_exp_accs):.6f}')

import torch
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data import UniqueSignatureDataset
from model import SINCModel, GCNModel, SAGEModel, GATModel, GINModel, SIRModel, PNAModel, EGCSModel, EGCMModel, set_seed

loss_fn = nn.BCEWithLogitsLoss()
acc_cnt_fn = lambda logits, labels: torch.sum((logits > 0).float() == labels, dtype=torch.float32)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # PyTorch 2.4.0 deprecation warnings


def train(model, train_loader, device, optimizer, scaler):
    model.train()

    total_loss, total = 0, 0
    for graphs in train_loader:
        graphs = graphs.to(device)
        feats = graphs.ndata.pop('feat')
        labels = graphs.ndata.pop('label')

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

    total_loss = 0
    total_acc, total = [0, 0], [0, 0]
    for graphs in dataloader:
        graphs = graphs.to(device)
        feats = graphs.ndata.pop('feat')
        labels = graphs.ndata.pop('label')

        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            logits = model(graphs, feats)
            loss = loss_fn(logits, labels)
            total_loss = total_loss + loss.item() * labels.shape[0]
            
            for i in range(2):
                if (labels == i).any():
                    total_acc[i] = total_acc[i] + acc_cnt_fn(logits[labels == i], labels[labels == i]).item()
                    total[i] = total[i] + (labels == i).sum().item()
        
    return total_loss / sum(total), sum(total_acc) / sum(total), 0.5 * (total_acc[0] / total[0] + total_acc[1] / total[1])


def run(model, train_loader, test_loader, device, args):
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    
    for epoch in range(args.epochs):
        loss = train(model, train_loader, device, optimizer, scaler)
        loss, acc, bal_acc = evaluate(model, train_loader, device)
        test_loss, test_acc, test_bal_acc = evaluate(model, test_loader, device)
        scheduler.step(loss)

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | acc: {acc:.4f} | bal_acc: {bal_acc:.4f} | '
                  f'test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | test_bal_acc: {test_bal_acc:.4f}')

    return acc, bal_acc, test_acc, test_bal_acc


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SINC-GCN/GCN/GraphSAGE/GATv2/GIN/PNA/SIR-GCN implementation on UniqueSignature',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    argparser.add_argument('--nworkers', type=int, default=0, help='number of workers')
    argparser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')
    
    argparser.add_argument('--model', type=str, default='SINC', help='model name', choices=['SINC', 'GCN', 'SAGE', 'GAT', 'GIN', 'SIR', 'PNA', 'EGC-S', 'EGC-M'])
    argparser.add_argument('--nhidden', type=int, default=16, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=1, help='number of graph convolution layers')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--nheads', type=int, default=1, help='number of attention heads for GAT')
    argparser.add_argument('--nlayers-mlp', type=int, default=2, help='number of MLP layers for GIN')
    
    argparser.add_argument('--min-nodes', type=int, default=30, help='minimum number of nodes in graphs')
    argparser.add_argument('--max-nodes', type=int, default=70, help='maximum number of nodes in graphs')
    argparser.add_argument('--prob-edge', type=float, default=0.5, help='probability of edges in graphs')
    argparser.add_argument('--nfeat-range', type=int, default=5, help='range of node features in graphs')
    argparser.add_argument('--samples', type=int, default=5000, help='number of random graphs to generate')
    argparser.add_argument('--train-size', type=float, default=0.8, help='fraction of samples for training')

    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=256, help='batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    argparser.add_argument('--min-lr', type=float, default=0, help='minimum learning rate')
    
    argparser.add_argument('--nruns', type=int, default=10, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()

    train_accs, train_bal_accs, test_accs, test_bal_accs = [], [], [], []
    for i in range(args.nruns):
        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset = UniqueSignatureDataset(min_nodes=args.min_nodes, max_nodes=args.max_nodes, prob_edge=args.prob_edge, 
                                         nfeat_range=args.nfeat_range, num_samples=args.samples)
        print(f'Percentage of positive nodes: {dataset.pos_nodes / dataset.num_nodes:.4f}')
        
        # Seed
        set_seed(args.seed + i)

        train_sampler = SubsetRandomSampler(torch.arange(int(args.train_size * len(dataset))))
        test_sampler = SubsetRandomSampler(torch.arange(int(args.train_size * len(dataset)), len(dataset)))
        train_loader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)
        test_loader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)

        # Load model
        Model = {'SINC': SINCModel, 'GCN': GCNModel, 'SAGE': SAGEModel, 'GAT': GATModel, 'GIN': GINModel, 'SIR': SIRModel, 'PNA': PNAModel, 'EGC-S': EGCSModel, 'EGC-M': EGCMModel}
        model = Model[args.model](args.nhidden, 1, args.nlayers, args.dropout, 
                                  num_heads=args.nheads, mlp_layers=args.nlayers_mlp).to(device)
        summary(model)

        # Training
        train_acc, train_bal_acc, test_acc, test_bal_acc = run(model, train_loader, test_loader, device, args)
        train_accs.append(train_acc)
        train_bal_accs.append(train_bal_acc)
        test_accs.append(test_acc)
        test_bal_accs.append(test_bal_acc)

    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Train accuracy: {train_accs}')
    print(f'Train balanced accuracy: {train_bal_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Test balanced accuracy: {test_bal_accs}')
    print(f'Average train accuracy: {np.mean(train_accs):.6f} ± {np.std(train_accs):.6f}')
    print(f'Average train balanced accuracy: {np.mean(train_bal_accs):.6f} ± {np.std(train_bal_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')
    print(f'Average test balanced accuracy: {np.mean(test_bal_accs):.6f} ± {np.std(test_bal_accs):.6f}')

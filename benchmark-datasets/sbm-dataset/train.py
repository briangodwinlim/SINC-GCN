import dgl
import torch
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.dataloading import GraphDataLoader
from dgl.data import PATTERNDataset, CLUSTERDataset

from model import SINCModel, GATModel, set_seed, warmup_lr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # PyTorch 2.4.0 deprecation warnings


def load_dataset(name, args):
    Dataset = PATTERNDataset if name == 'PATTERN' else CLUSTERDataset
    transform = dgl.transforms.AddSelfLoop() if args.add_self_loop else None
    dataset = Dataset(mode='train', transform=transform)
    dataset.input_dim = torch.unique(dataset[0].ndata['feat'], dim=0).shape[0]
    
    train_loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.nworkers)
    val_loader = GraphDataLoader(Dataset(mode='valid', transform=transform), batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.nworkers)
    test_loader = GraphDataLoader(Dataset(mode='test', transform=transform), batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.nworkers)
    
    return dataset, train_loader, val_loader, test_loader
    

def loss_fn(logits, labels):
    weight = torch.tensor([torch.sum(labels == c) for c in range(logits.shape[1])], device=labels.device)
    weight = (labels.shape[0] - weight) * (weight > 0) / labels.shape[0]
    loss_fn_ = nn.CrossEntropyLoss(weight=weight)
    return loss_fn_(logits, labels)

def acc_fn(logits, labels):
    preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
    classes = torch.unique(torch.cat([labels, preds], dim=-1))
    return torch.mean(torch.tensor([torch.mean(preds[labels == c] == c, dtype=torch.float64) if (labels == c).any() else 0 for c in classes]))


def train(model, train_loader, device, optimizer, scaler, args):
    model.train()

    total_loss, total = 0, 0
    for graphs in train_loader:
        graphs = graphs.to(device)
        feats = graphs.ndata.pop('feat')
        labels = graphs.ndata.pop('label').to(torch.int64)

        optimizer.zero_grad()
        
        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            logits = model(graphs, feats)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total = total + 1
        total_loss = total_loss + loss.item()
        
    return total_loss / total

@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()

    total_loss, total_acc, total = 0, 0, 0
    for graphs in dataloader:
        graphs = graphs.to(device)
        feats = graphs.ndata.pop('feat')
        labels = graphs.ndata.pop('label').to(torch.int64)
        
        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            logits = model(graphs, feats)
            loss = loss_fn(logits, labels)
            acc = acc_fn(logits, labels)

        total = total + 1
        total_loss = total_loss + loss.item()
        total_acc = total_acc + acc.item()

    return total_loss / total, total_acc / total


def run(model, train_loader, val_loader, test_loader, device, args, iter):
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    best_val_loss = 1e10

    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, train_loader, device, optimizer, scaler, args)
        loss, acc = evaluate(model, train_loader, device, args)
        val_loss, val_acc = evaluate(model, val_loader, device, args)
        test_loss, test_acc = evaluate(model, test_loader, device, args)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result = {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }
            
        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.4f} | acc: {acc:.4f} | '
                  f'val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | '
                  f'test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}')
        
    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'SINC-GCN/GATv2 implementation on SBMDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    argparser.add_argument('--nworkers', type=int, default=0, help='number of workers')
    argparser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')

    argparser.add_argument('--model', type=str, default='SINC', help='model name', choices=['SINC', 'GAT']) 
    argparser.add_argument('--nhidden', type=int, default=64, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=4, help='number of graph convolution layers')
    argparser.add_argument('--input-dropout', type=float, default=0, help='input dropout rate')
    argparser.add_argument('--edge-dropout', type=float, default=0, help='edge dropout rate')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--norm', type=str, default='none', help='type of normalization', choices=['cn', 'bn', 'ln', 'none'])
    argparser.add_argument('--readout-layers', type=int, default=1, help='number of MLP layers for node readout')
    argparser.add_argument('--readout-dropout', type=float, default=0, help='dropout rate for node readout')
    argparser.add_argument('--jumping-knowledge', action='store_true', help='use jumping knowledge for node readout')
    argparser.add_argument('--residual', action='store_true', help='add residual connections')
    argparser.add_argument('--resid-layers', type=int, default=0, help='number of MLP layers for SINC residual')
    argparser.add_argument('--resid-dropout', type=float, default=0, help='dropout rate for SINC residual')
    argparser.add_argument('--feat-dropout', type=float, default=0, help='dropout rate for SINC inner linear transformations')
    argparser.add_argument('--agg-type', type=str, default='mean', help='aggregation type for SINC', choices=['sum', 'max', 'mean', 'sym'])
    argparser.add_argument('--neigh-agg-type', type=str, default='mean', help='neighborhood aggregation type for SINC', choices=['sum', 'max', 'mean', 'sym'])
    argparser.add_argument('--nheads', type=int, default=1, help='number of attention heads for GAT')
    argparser.add_argument('--attn-dropout', type=float, default=0, help='dropout rate for GAT attention')
    
    argparser.add_argument('--dataset', type=str, default='PATTERN', help='name of dataset', choices=['PATTERN', 'CLUSTER'])
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graph')

    argparser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=32, help='batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    argparser.add_argument('--min-lr', type=float, default=0, help='minimum learning rate')
    
    argparser.add_argument('--nruns', type=int, default=5, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()

    val_accs, test_accs = [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset, train_loader, val_loader, test_loader = load_dataset(args.dataset, args)

        # Load model
        Model = {'SINC': SINCModel, 'GAT': GATModel}
        model = Model[args.model](dataset.input_dim, args.nhidden, dataset.num_classes, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm,
                                  args.readout_layers, args.readout_dropout, args.jumping_knowledge, args.residual,
                                  resid_layers=args.resid_layers, resid_dropout=args.resid_dropout, 
                                  feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                  neigh_agg_type=args.neigh_agg_type, num_heads=args.nheads, attn_dropout=args.attn_dropout).to(device)
        summary(model)

        # Training
        result = run(model, train_loader, val_loader, test_loader, device, args, i)
        val_accs.append(result['val_acc'])
        test_accs.append(result['test_acc'])
    
    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val accuracy: {val_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Average val accuracy: {np.mean(val_accs):.6f} ± {np.std(val_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')

# SINC-GCN (PATTERN)
# Namespace(cpu=False, gpu=0, seed=0, nworkers=1, use_amp=True, model='SINC', nhidden=70, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=1, readout_dropout=0.0, jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.0, feat_dropout=0.0, agg_type='sym', neigh_agg_type='mean', nheads=1, attn_dropout=0, dataset='PATTERN', add_self_loop=False, epochs=200, batch_size=128, lr=0.001, wd=0.1, factor=0.5, patience=10, min_lr=0, nruns=5, log_every=20)
# Runned 5 times
# Val accuracy: [0.8554906033385812, 0.8555633324389803, 0.8558332402586255, 0.8555958315962389, 0.8558967719445746]
# Test accuracy: [0.8579755034592029, 0.8582067429579555, 0.8577830446248599, 0.857892912454746, 0.8576713774470476]
# Average val accuracy: 0.855676 ± 0.000159
# Average test accuracy: 0.857906 ± 0.000182

# SINC-GCN (CLUSTER)
# Namespace(cpu=False, gpu=0, seed=0, nworkers=1, use_amp=True, model='SINC', nhidden=70, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=1, readout_dropout=0.0, jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.0, feat_dropout=0.0, agg_type='sym', neigh_agg_type='mean', nheads=1, attn_dropout=0, dataset='CLUSTER', add_self_loop=False, epochs=300, batch_size=128, lr=0.001, wd=0.1, factor=0.5, patience=10, min_lr=0, nruns=5, log_every=20)
# Runned 5 times
# Val accuracy: [0.6303736704151448, 0.6353368122880755, 0.6298288045480795, 0.6332247938794299, 0.6320791388138964]
# Test accuracy: [0.634243237571834, 0.6374247287719089, 0.6335175447543899, 0.6363632928918431, 0.633797027875442]
# Average val accuracy: 0.632169 ± 0.001993
# Average test accuracy: 0.635069 ± 0.001545

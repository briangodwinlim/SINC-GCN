import dgl
import torch
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data import MNISTSuperPixelDataset, CIFAR10SuperPixelDataset

from model import SINCModel, GINModel, set_seed, warmup_lr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # PyTorch 2.4.0 deprecation warnings

loss_fn = nn.CrossEntropyLoss()
acc_fn = lambda logits, labels: torch.mean(logits.argmax(dim=-1) == labels, dtype=torch.float32)


def load_dataset(name, args):
    Dataset = MNISTSuperPixelDataset if name == 'MNIST' else CIFAR10SuperPixelDataset
    
    transform = dgl.transforms.AddSelfLoop() if args.add_self_loop else None
    dataset = Dataset(split='train', use_feature=args.use_feature, transform=transform)
    dataset.input_dim = dataset[0][0].ndata['feat'].shape[1]
    dataset.edge_dim = dataset[0][0].edata['feat'].shape[1]
    dataset.output_dim = np.unique(dataset[:][1]).shape[0]
    test_dataset = Dataset(split='test', use_feature=args.use_feature, transform=transform)
    
    train_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(np.arange(len(dataset))[5000:]), batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)
    val_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(np.arange(len(dataset))[:5000]), batch_size=args.batch_size, drop_last=False, num_workers=args.nworkers)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.nworkers)
    
    return dataset, train_loader, val_loader, test_loader


def train(model, train_loader, device, optimizer, scaler, args):
    model.train()

    total_loss, total = 0, 0
    for graphs, labels in train_loader:
        graphs = graphs.to(device)
        labels = labels.to(device).to(torch.int64)
        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            logits = model(graphs, nfeats, efeats)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
        
    return total_loss / total

@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()

    total_loss, total_acc, total = 0, 0, 0
    for graphs, labels in dataloader:
        graphs = graphs.to(device)
        labels = labels.to(device).to(torch.int64)
        nfeats = graphs.ndata.pop('feat')
        efeats = graphs.edata.pop('feat')
        
        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            logits = model(graphs, nfeats, efeats)
            loss = loss_fn(logits, labels)
            acc = acc_fn(logits, labels)

        total = total + labels.shape[0]
        total_loss = total_loss + loss.item() * labels.shape[0]
        total_acc = total_acc + acc.item() * labels.shape[0]

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
        'SINC-GCN/GIN implementation on SuperPixelDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    argparser.add_argument('--nworkers', type=int, default=0, help='number of workers')
    argparser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')

    argparser.add_argument('--model', type=str, default='SINC', help='model name', choices=['SINC', 'GIN']) 
    argparser.add_argument('--nhidden', type=int, default=64, help='number of hidden units')
    argparser.add_argument('--nlayers', type=int, default=4, help='number of graph convolution layers')
    argparser.add_argument('--input-dropout', type=float, default=0, help='input dropout rate')
    argparser.add_argument('--edge-dropout', type=float, default=0, help='edge dropout rate')
    argparser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    argparser.add_argument('--norm', type=str, default='none', help='type of normalization', choices=['gn', 'cn', 'bn', 'ln', 'none'])
    argparser.add_argument('--readout-layers', type=int, default=1, help='number of MLP layers for graph readout')
    argparser.add_argument('--readout-dropout', type=float, default=0, help='dropout rate for graph readout')
    argparser.add_argument('--readout-pooling', type=str, default='sum', help='type of graph readout pooling', choices=['sum', 'mean'])
    argparser.add_argument('--jumping-knowledge', action='store_true', help='use jumping knowledge for graph readout')
    argparser.add_argument('--residual', action='store_true', help='add residual connections')
    argparser.add_argument('--resid-layers', type=int, default=0, help='number of MLP layers for residual connections')
    argparser.add_argument('--resid-dropout', type=float, default=0, help='dropout rate for residual connections')
    argparser.add_argument('--feat-dropout', type=float, default=0, help='dropout rate for SINC inner linear transformations')
    argparser.add_argument('--agg-type', type=str, default='sum', help='aggregation type', choices=['sum', 'max', 'mean', 'sym'])
    argparser.add_argument('--neigh-agg-type', type=str, default='sum', help='neighborhood aggregation type for SINC', choices=['sum', 'max', 'mean', 'sym'])
    argparser.add_argument('--nlayers-mlp', type=int, default=2, help='number of MLP layers for GIN')

    argparser.add_argument('--dataset', type=str, default='MNIST', help='name of dataset', choices=['MNIST', 'CIFAR10'])
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graphs')
    argparser.add_argument('--use-feature', action='store_true', help='use features in addition to super-pixel locations')
    
    argparser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    argparser.add_argument('--batch-size', type=int, default=32, help='batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    argparser.add_argument('--min-lr', type=float, default=0, help='minimum learning rate')
    
    argparser.add_argument('--nruns', type=int, default=5, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=20, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()
    
    if args.model == 'GIN' and args.agg_type == 'sym':
        raise ValueError('GIN cannot use agg_type == sym')

    val_accs, test_accs = [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset, train_loader, val_loader, test_loader = load_dataset(args.dataset, args)

        # Load model
        Model = {'SINC': SINCModel, 'GIN': GINModel}
        model = Model[args.model](dataset.input_dim, dataset.edge_dim, args.nhidden, dataset.output_dim, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm, 
                                  args.readout_layers, args.readout_dropout, args.readout_pooling, args.jumping_knowledge,
                                  args.residual, args.resid_layers, args.resid_dropout, feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                  neigh_agg_type=args.neigh_agg_type, mlp_layers=args.nlayers_mlp).to(device)
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

# SINC-GCN (MNIST)
# Namespace(cpu=False, gpu=0, seed=0, nworkers=1, use_amp=True, model='SINC', nhidden=75, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=3, readout_dropout=0.0, readout_pooling='mean', jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.2, feat_dropout=0.1, agg_type='max', neigh_agg_type='mean', nlayers_mlp=2, dataset='MNIST', add_self_loop=False, use_feature=False, epochs=200, batch_size=128, lr=0.001, wd=0.1, factor=0.5, patience=10, min_lr=0, nruns=5, log_every=20)
# Runned 5 times
# Val accuracy: [0.9824, 0.9834, 0.9848, 0.9844, 0.9822]
# Test accuracy: [0.9826, 0.9833, 0.9833, 0.9819, 0.9829]
# Average val accuracy: 0.983440 ± 0.001038
# Average test accuracy: 0.982800 ± 0.000522

# SINC-GCN (CIFAR10)
# Namespace(cpu=False, gpu=0, seed=0, nworkers=1, use_amp=True, model='SINC', nhidden=75, nlayers=4, input_dropout=0.0, edge_dropout=0.0, dropout=0.0, norm='bn', readout_layers=3, readout_dropout=0.0, readout_pooling='mean', jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.2, feat_dropout=0.1, agg_type='max', neigh_agg_type='mean', nlayers_mlp=2, dataset='CIFAR10', add_self_loop=False, use_feature=False, epochs=200, batch_size=128, lr=0.001, wd=0.1, factor=0.5, patience=10, min_lr=0, nruns=5, log_every=20)
# Runned 5 times
# Val accuracy: [0.7386, 0.7448, 0.7366, 0.7456, 0.7516]
# Test accuracy: [0.7269, 0.7355, 0.7317, 0.7387, 0.7358]
# Average val accuracy: 0.743440 ± 0.005354
# Average test accuracy: 0.733720 ± 0.004072

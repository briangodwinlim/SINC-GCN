import dgl
import torch
import argparse
import numpy as np
from torch import nn
from torchinfo import summary
from dgl.data import WikiCSDataset

from model import SINCModel, GATModel, set_seed, warmup_lr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # PyTorch 2.4.0 deprecation warnings

loss_fn = nn.CrossEntropyLoss()
acc_fn = lambda logits, labels: torch.mean(logits.argmax(dim=-1) == labels, dtype=torch.float32)


def load_dataset(device, args, iter):
    dataset = WikiCSDataset()
    graph = dataset[0]
    graph = dgl.to_bidirected(graph, copy_ndata=True) if args.add_reverse_edge else graph
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph)) if args.add_self_loop else graph
    graph = graph.to(device)
    labels = graph.ndata['label'].to(device)
    train_idx = graph.ndata['train_mask'][:, iter].to(device)
    val_idx = (graph.ndata['val_mask'] + graph.ndata['stopping_mask'])[:, iter].to(device)
    test_idx = graph.ndata['test_mask'].to(device)
    dataset.input_dim = graph.ndata['feat'].shape[1]
    
    return dataset, graph, labels, (train_idx.to(torch.bool), val_idx.to(torch.bool), test_idx.to(torch.bool))


def train(model, graph, labels, masks, device, args, optimizer, scaler):
    model.train()
    optimizer.zero_grad()
    
    with torch.autocast(device_type=device.type, enabled=args.use_amp):
        train_idx, val_idx, test_idx = masks
        feats = graph.ndata['feat']
        logits = model(graph, feats)
        loss = loss_fn(logits[train_idx], labels[train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, graph, labels, masks, device, args):
    model.eval()
    
    with torch.autocast(device_type=device.type, enabled=args.use_amp):
        train_idx, val_idx, test_idx = masks
        feats = graph.ndata['feat']
        logits = model(graph, feats)
        
        loss = loss_fn(logits[train_idx], labels[train_idx]).item()
        acc = acc_fn(logits[train_idx], labels[train_idx]).item()
        val_loss = loss_fn(logits[val_idx], labels[val_idx]).item()
        val_acc = acc_fn(logits[val_idx], labels[val_idx]).item()
        test_loss = loss_fn(logits[test_idx], labels[test_idx]).item()
        test_acc = acc_fn(logits[test_idx], labels[test_idx]).item()

        return loss, acc, val_loss, val_acc, test_loss, test_acc


def run(model, graph, labels, masks, device, args, iter):
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    best_val_loss = 1e10

    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, graph, labels, masks, device, args, optimizer, scaler)
        loss, acc, val_loss, val_acc, test_loss, test_acc = evaluate(model, graph, labels, masks, device, args)
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
        'SINC-GCN/GATv2 implementation on WikiCSDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
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
    
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to graph')
    argparser.add_argument('--add-reverse-edge', action='store_true', help='add reverse edge to graph')

    argparser.add_argument('--epochs', type=int, default=200, help='number of epochs')
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
        for idx in range(20):
            # Set seed
            set_seed(args.seed + i)

            # Load dataset
            device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
            dataset, graph, labels, masks = load_dataset(device, args, idx)

            # Load model
            Model = {'SINC': SINCModel, 'GAT': GATModel}
            model = Model[args.model](dataset.input_dim, args.nhidden, dataset.num_classes, args.nlayers, args.input_dropout, args.edge_dropout, args.dropout, args.norm,
                                      args.readout_layers, args.readout_dropout, args.jumping_knowledge, args.residual,
                                      resid_layers=args.resid_layers, resid_dropout=args.resid_dropout, 
                                      feat_dropout=args.feat_dropout, agg_type=args.agg_type, 
                                      neigh_agg_type=args.neigh_agg_type, num_heads=args.nheads, attn_dropout=args.attn_dropout).to(device)
            summary(model)

            # Training
            result = run(model, graph, labels, masks, device, args, i)
            val_accs.append(result['val_acc'])
            test_accs.append(result['test_acc'])
    
    print(args)
    print(f'Runned {args.nruns} times')
    print(f'Val accuracy: {val_accs}')
    print(f'Test accuracy: {test_accs}')
    print(f'Average val accuracy: {np.mean(val_accs):.6f} ± {np.std(val_accs):.6f}')
    print(f'Average test accuracy: {np.mean(test_accs):.6f} ± {np.std(test_accs):.6f}')

# SINC-GCN
# Namespace(cpu=False, gpu=0, seed=0, use_amp=True, model='SINC', nhidden=50, nlayers=4, input_dropout=0.3, edge_dropout=0.0, dropout=0.1, norm='bn', readout_layers=1, readout_dropout=0.0, jumping_knowledge=False, residual=True, resid_layers=1, resid_dropout=0.2, feat_dropout=0.0, agg_type='mean', neigh_agg_type='mean', nheads=1, attn_dropout=0, add_self_loop=False, add_reverse_edge=False, epochs=200, lr=0.001, wd=0.1, factor=0.5, patience=10, min_lr=0, nruns=5, log_every=20)
# Runned 5 times
# Val accuracy: [0.7885854840278625, 0.78574138879776, 0.809252917766571, 0.7923777103424072, 0.7775881886482239, 0.797307550907135, 0.778725802898407, 0.7952218651771545, 0.7942737936973572, 0.7961698770523071, 0.7940841913223267, 0.788016676902771, 0.7999621033668518, 0.7927569150924683, 0.7872582674026489, 0.796928346157074, 0.797307550907135, 0.8069776296615601, 0.7940841913223267, 0.797307550907135, 0.790291965007782, 0.778346598148346, 0.8073568344116211, 0.7912400364875793, 0.7794842720031738, 0.7927569150924683, 0.7866894006729126, 0.797307550907135, 0.788395881652832, 0.7959802746772766, 0.7918088436126709, 0.7925673127174377, 0.7901023626327515, 0.799203634262085, 0.7999621033668518, 0.7921881079673767, 0.7938945889472961, 0.8031854629516602, 0.7967386841773987, 0.7950322031974792, 0.7938945889472961, 0.7827076315879822, 0.8094425201416016, 0.7901023626327515, 0.7815699577331543, 0.7986348271369934, 0.7827076315879822, 0.7929465174674988, 0.796928346157074, 0.7952218651771545, 0.7823284268379211, 0.7891543507575989, 0.8009101152420044, 0.7956010699272156, 0.7908608317375183, 0.7933257222175598, 0.7986348271369934, 0.7946529984474182, 0.7971179485321045, 0.7925673127174377, 0.7940841913223267, 0.7832764387130737, 0.8069776296615601, 0.7946529984474182, 0.7825180292129517, 0.7921881079673767, 0.7781569957733154, 0.788016676902771, 0.7849829196929932, 0.7891543507575989, 0.7859309911727905, 0.7929465174674988, 0.7984452247619629, 0.7965490818023682, 0.7872582674026489, 0.7910504341125488, 0.7906712293624878, 0.7984452247619629, 0.7878270745277405, 0.7993932366371155, 0.7906712293624878, 0.7830868363380432, 0.8058399558067322, 0.7923777103424072, 0.7821387648582458, 0.796928346157074, 0.7882062792778015, 0.7904816269874573, 0.7938945889472961, 0.7959802746772766, 0.7832764387130737, 0.788016676902771, 0.7929465174674988, 0.7948426008224487, 0.7885854840278625, 0.7965490818023682, 0.7882062792778015, 0.7956010699272156, 0.7912400364875793, 0.7938945889472961]
# Test accuracy: [0.7798870801925659, 0.768941342830658, 0.7918590307235718, 0.7747562527656555, 0.7733880281448364, 0.7915170192718506, 0.7792029976844788, 0.7843338251113892, 0.7826235294342041, 0.7880964279174805, 0.7865571975708008, 0.7795450687408447, 0.7882674932479858, 0.7834786772727966, 0.7814263701438904, 0.7853599786758423, 0.7807422280311584, 0.7973319292068481, 0.7810842990875244, 0.7845048308372498, 0.7857020497322083, 0.7680861949920654, 0.7862151265144348, 0.768941342830658, 0.7727039456367493, 0.7865571975708008, 0.7798870801925659, 0.7877544164657593, 0.7797160744667053, 0.7915170192718506, 0.783307671546936, 0.7858730554580688, 0.7855310440063477, 0.7798870801925659, 0.7870702743530273, 0.7857020497322083, 0.7805712223052979, 0.7903198003768921, 0.7821104526519775, 0.781597375869751, 0.788951575756073, 0.7663758993148804, 0.7933983206748962, 0.7781768441200256, 0.7655207514762878, 0.7922011017799377, 0.7802291512489319, 0.7822815179824829, 0.7848469018936157, 0.7923721671104431, 0.7752693295478821, 0.783307671546936, 0.7870702743530273, 0.7851889729499817, 0.7826235294342041, 0.7904908061027527, 0.7817684412002563, 0.7875833511352539, 0.7831366062164307, 0.7805712223052979, 0.7863861918449402, 0.7667179703712463, 0.7904908061027527, 0.7672310471534729, 0.7692833542823792, 0.789293646812439, 0.773901104927063, 0.7744142413139343, 0.783307671546936, 0.7805712223052979, 0.7798870801925659, 0.7824525237083435, 0.7785188555717468, 0.7805712223052979, 0.7807422280311584, 0.7773216962814331, 0.7792029976844788, 0.7867282032966614, 0.7757824063301086, 0.7827945947647095, 0.781255304813385, 0.770309567451477, 0.7863861918449402, 0.7737300992012024, 0.770309567451477, 0.7865571975708008, 0.7809132933616638, 0.783307671546936, 0.7899777293205261, 0.7860441207885742, 0.7749273180961609, 0.7754403948783875, 0.7848469018936157, 0.7855310440063477, 0.7822815179824829, 0.7903198003768921, 0.7780057787895203, 0.7838207483291626, 0.7706515789031982, 0.7769796252250671]
# Average val accuracy: 0.792471 ± 0.006770
# Average test accuracy: 0.781705 ± 0.006764

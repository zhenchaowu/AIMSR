import argparse
import os
import torch as th
import numpy as np
import random
import sys
import pickle
import scipy.sparse

sys.path.append('..')
sys.path.append('../..')


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.enabled = True
    
seed_torch(123)


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if len(memory_available) == 0:
        return -1
    return int(np.argmax(memory_available))

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_freer_gpu())

print('The CUDA DEVICE is CUDA: {}'.format(str(get_freer_gpu())))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Tmall', help='the dataset directory')
parser.add_argument('--item-dim', type=int, default=256, help='the dimensionality of item embedding')
parser.add_argument('--pos-dim', type=int, default=256, help='the dimensionality of position embedding')
parser.add_argument('--num-layers', type=int, default=1, help='the number of layers')
parser.add_argument('--batch-size', type=int, default=256, help='the batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs')
parser.add_argument('--local-update', type=int, default=1, help='the number of local update')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument('--local-lr', type=float, default=1e-4, help='the local learning rate')
parser.add_argument('--ssl-batch', type=int, default=100, help='the batch size of SSL')
parser.add_argument('--meta-learning', type=bool, default=True, help='whether to use meta learning')
parser.add_argument('--reweight', type=bool, default=True, help='whether to use reweight')

parser.add_argument('--w', type=int, default=12, help='the normalized weight')
parser.add_argument('--alpha', type=float, default=0, help='the weight coefficient of meta_infoNCELosses')
parser.add_argument('--gamma', type=float, default=0.05, help='the weight coefficient of batch_infoNCELosses')
parser.add_argument('--beta', type=int, default=2, help='the dynamic controller of incremental learning')
parser.add_argument('--feat-drop', type=float, default=0.1, help='the dropout ratio for features')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='the parameter for L2 regularization')
parser.add_argument('--patience', type=int, default=3, help='the number of epochs that the performance does not improves after which the training stops')
parser.add_argument('--num-workers', type=int, default=4, help='the number of processes to load the input graphs')
parser.add_argument('--valid-split', type=float, default=None, help='the fraction for the validation set')
parser.add_argument('--log-interval', type=int, default=100, help='print the loss after this number of iterations')
parser.add_argument('--order', type=int, default=5, help='order of msg')
parser.add_argument('--reducer', type=str, default='mean', help='method for reducer')
parser.add_argument('--norm', type=bool, default=True, help='whether use l2 norm')
parser.add_argument('--IL-interval', type=int, default=10, help='the interval batch of incremental learning')

parser.add_argument('--granularity-type', type=str, default='hybrid', help='mean, gru, and hybrid')

parser.add_argument(
    '--extra',
    #action='store_true',
    default=True,
    help='whether use REnorm.',
)

parser.add_argument(
    '--fusion',
    #action='store_true',
    default=True,
    help='whether use IFR.',
)

parser.add_argument(
    '--incremental-learning',
    type=bool,
    default=True,
    help='whether use incremental learning'
)

parser.add_argument(
    '--data-aug',
    type=bool,
    default=True,
    help='whether use data augmentation'
)

args = parser.parse_args()
print(args)

######################################################################################################################

from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from src.utils.data.dataset import AugmentedDataset
from src.utils.data.collate import (
    seq_to_ccs_graph,
    collate_fn_factory_ccs
)
from src.utils.train import TrainRunner
from src.models import aimsr

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


print('reading dataset')
data_dir = '../datasets/'
train_sessions = pickle.load(open(data_dir + args.dataset +'/train.txt', 'rb'))
test_sessions = pickle.load(open(data_dir + args.dataset +'/test.txt', 'rb'))

with open(data_dir + args.dataset + '/num_items.txt', 'r') as f:
    num_items = int(f.readline())
    
#global_adj_coo = scipy.sparse.load_npz('datasets/' + args.dataset + '/adj_global120.npz')  #
#global_adj_coo = scipy.sparse.load_npz('datasets/' + args.dataset + '/adj_global320.npz')  #retailrocket
global_adj_coo = scipy.sparse.load_npz('../datasets/' + args.dataset + '/adj_global.npz')  #Tmall

def sparse2sparse(coo_matrix):
    v1 = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    i = th.LongTensor(indices)
    v = th.FloatTensor(v1)
    shape = coo_matrix.shape
    sparse_matrix = th.sparse.LongTensor(i, v, th.Size(shape))
    return sparse_matrix
sparse_global_adj = sparse2sparse(global_adj_coo).cuda()
        
num_pos = 0
for sample in train_sessions[0]:
    num_pos = len(sample) if len(sample) > num_pos else num_pos
for sample in test_sessions[0]:
    num_pos = len(sample) if len(sample) > num_pos else num_pos
num_pos += 1 

print('the number of all items: ', num_items)
print('the number of all position: ', num_pos)

train_set = AugmentedDataset(train_sessions)
test_set  = AugmentedDataset(test_sessions)
print('the number of samples in train set is {}'.format(len(train_set)))
print('the number of samples in test set is {}'.format(len(test_set)))


collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph,), num_pos, order=args.order)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True
)




model = aimsr.AIMSR(num_items, num_pos, args.item_dim, args.pos_dim, args.num_layers, sparse_global_adj, args.w, args.local_lr, args.local_update, args.ssl_batch, args.meta_learning, args.reweight, dropout=args.feat_drop, reducer=args.reducer, order=args.order, norm=args.norm, extra=args.extra, fusion=args.fusion, data_aug=args.data_aug, granularity_type=args.granularity_type, device=device)

model = model.to(device)

runner = TrainRunner(
    model,
    train_loader,
    test_loader,
    device=device,
    alpha=args.alpha,
    gamma=args.gamma,
    beta=args.beta,
    lr=args.lr,
    order=args.order,
    reweight=args.reweight,
    incremental_learning=args.incremental_learning,
    IL_interval=args.IL_interval,
    weight_decay=args.weight_decay,
    patience=args.patience,
)




print('start training')
mrr20, hit20, mrr10, hit10 = runner.train(args.epochs, args.log_interval)
print('MRR@20\tHR@20')
print(f'{mrr20 * 100:.3f}%\t{hit20 * 100:.3f}%')
print('MRR@10\tHR@10')
print(f'{mrr10 * 100:.3f}%\t{hit10 * 100:.3f}%')

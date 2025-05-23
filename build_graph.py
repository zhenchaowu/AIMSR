import pickle
import argparse
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm
import time


def build_adj(opt, num_node, verbose=True):
   
    sessions = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.pkl', 'rb'))


    if verbose: print("create seq adj matrix")
    adj = np.zeros((num_node + 1, num_node + 1), dtype=int)
    
    for sess in sessions:
        for i in range(len(sess) - 1):
            item_i = sess[i]
            item_j = sess[i + 1]
            adj[item_i, item_j] += 1
            
    
            
    
    if verbose: print("max. weight: ", adj.max())
    
    # filter out unreliable edges
    adj = np.where(adj <= opt.filter, 0, adj)

    np.fill_diagonal(adj, adj.diagonal() + 1)  # A + I

    if verbose: print(np.count_nonzero(adj) - np.count_nonzero(adj.diagonal()))
    if opt.spare:
        # reverse weight values
        rev_w_adj = adj.astype(np.float32)
        rev_mask = rev_w_adj.nonzero()
        max_v = max(rev_w_adj[rev_mask])

        rev_w_adj[rev_mask] = (max_v + 1) - rev_w_adj[rev_mask]

        """
        high weight between two nodes = high energy, lower energy connections are preferable
        find lowest energy connection between 2 nodes -> use connections with low energy (high weights in org adj)
        """
        if verbose: print("find shortest paths with lowest cost")
        adj = dijkstra(csgraph=rev_w_adj, directed=True, unweighted=False, limit=opt.limit)

        # re-reverse weight values
        np.fill_diagonal(adj, rev_w_adj.diagonal())  # refill diagonal
        adj[adj == np.inf] = 0
        mask = adj.nonzero()
        if (opt.limit - max_v) > 0:
            max_v = max(adj[mask])

        adj[mask] = (max_v + 1) - adj[mask]

    if verbose: print(np.count_nonzero(adj) - np.count_nonzero(adj.diagonal()))
    if verbose: print("normalize")

    # to csr sparse matrix
    adj_sparse = scipy.sparse.csr_matrix(adj)

    # symmetrical normalization -> D^{-1/2} A D^{-1/2} (GCN)
    rowsum = np.array(adj.sum(axis=1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_mat_inv = scipy.sparse.diags(r_inv)
    adj_sparse = r_mat_inv.dot(adj_sparse).dot(r_mat_inv)

    # to coo sparse matrix
    adj_sparse = adj_sparse.tocoo()

    return adj, adj_sparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='retailrocket', help='tmall/lastfm/retailrocket')
    parser.add_argument('--filter', type=int, default=0, help='Filter out unreliable edges below threshold.')
    parser.add_argument('--spare', type=int, default=1, help='Create adj based on shortest paths.')
    parser.add_argument('--limit', type=float, default=200, help='Max. search depth in dijsktra.')
    opt = parser.parse_args()

    if opt.dataset == 'tmall':
        opt.limit = 190 # max: 197
        num_node = 40727
    elif opt.dataset == 'retailrocket':
        opt.limit = 320 # max: 331
        num_node = 36968
    elif opt.dataset == 'diginetica':
        opt.limit = 120 #max: 134
        num_node = 43097

    

    print(opt)
    adj, adj_sparse = build_adj(opt, num_node)

    print("write to file...")
    scipy.sparse.save_npz('datasets/' + opt.dataset + '/adj_global', adj_sparse)

    print(adj_sparse.shape)
    print("Graph built.")

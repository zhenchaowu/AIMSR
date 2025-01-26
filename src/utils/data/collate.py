from collections import Counter
import numpy as np
import torch as th
import dgl
import pickle
import numba
from numba import jit


def label_last(g, last_nid):
    is_last = th.zeros(g.num_nodes(), dtype=th.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g

def label_last_ccs(g, last_nid):
    for i in range(len(last_nid)):
        is_last = th.zeros(g.num_nodes('s'+str(i+1)), dtype=th.int32)
        is_last[last_nid[i]] = 1
        g.nodes['s'+str(i+1)].data['last'] = is_last
    return g

def label_last_k(g, last_nids):
    is_last = th.zeros(g.number_of_nodes(), dtype=th.int32)
    is_last[last_nids] = 1
    g.nodes['s1'].data['last'] = is_last
    return g

def seq_to_eop_multigraph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
    else:
        src = th.LongTensor([])
        dst = th.LongTensor([])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])
    return g

def seq_to_shortcut_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    edges = counter.keys()
    src, dst = zip(*edges)

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g

def seq_to_session_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]
    )
    edges = counter.keys()
    if len(edges) > 0:
        src, dst = zip(*edges)
        weight = th.tensor(list(counter.values()))
    else:
        src, dst = [0], [0]
        weight = th.ones(1).long()

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    
    g.edata['w'] = weight
    # print(g.edata)
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])

    return g

def seq_to_ccs_graph(seq, order=1, coaDict=None):

    #print('the seq is {}'.format(seq))
    #print('the order is {}'.format(order))
    

    order1 = order
    order = min(order, len(seq))
    items = np.unique(seq)  #np.unique()有排序功能
    iid2nid = {iid: i for i, iid in enumerate(items)}  #将seq中的item重新编号
    
    #print('the iid2nid is {}'.format(iid2nid))
    #aaa
    
    
    num_nodes = len(items)
    seq_nid = [iid2nid[iid] for iid in seq]  #seq_nid为重新编号后的seq
    last_k = [iid2nid[seq[-1]]]  #对seq的最后一个元素重新编号
    
    #print(seq_nid[-1], last_k)
    #aaaa
    
    
    combine_seqs = []
    
    def com(i, order):
        item = str(seq[i:i+order])

        return item 
    
    class combine:
        def __init__(self):
            self.dict = {}
        
        def __call__(self, *input):
            return self.forward(*input)    
        
        def forward(self, i, order):
            if str(i) not in self.dict:
                self.dict[str(i)] = {}
            if order not in self.dict[str(i)]:
                self.dict[str(i)][order] = com(i, order)
            return self.dict[str(i)][order]
        
    combine = combine()  
    
    #print('iid2nid is {}'.format(iid2nid))
    
    item_dicts = [iid2nid]
    
    #print('item_dicts is {}'.format(item_dicts))
    
    #aaa
    
    cid2nid = {}
    item_g = []
    for i in range(1, order1):
        combine_seq = []
        item_dict = {}
        cnt = 0
        for j in range(len(seq_nid)-i):
            item = combine(j, i+1)
            if item not in item_dict:
                item_dict[item] = cnt
                cnt += 1
                combine_seq.append([seq[idx] for idx in range(j, j+i+1)])
    
        if len(item_dict) > 0:
            last_k.append(item_dict[item])
        else:
            last_k.append(0)
        combine_seqs.append(combine_seq)
                
        item_dicts.append(item_dict)
    
    #print('seq is {}'.format(seq))  #[11661, 38604, 40054, 40054, 5949, 20922, 10479, 40054]
    #print('seq_nid is {}'.format(seq_nid))   #[2, 4, 5, 5, 0, 3, 1, 5]
    #print('item_dicts is {}'.format(item_dicts))
    #item_dicts is [{5949: 0, 10479: 1, 11661: 2, 20922: 3, 38604: 4, 40054: 5}, {'[11661, 38604]': 0, '[38604, 40054]': 1, '[40054, 40054]': 2, '[40054, 5949]': 3, '[5949, 20922]': 4, '[20922, 10479]': 5, '[10479, 40054]': 6}, {'[11661, 38604, 40054]': 0, '[38604, 40054, 40054]': 1, '[40054, 40054, 5949]': 2, '[40054, 5949, 20922]': 3, '[5949, 20922, 10479]': 4, '[20922, 10479, 40054]': 5}]

    #print('combine_seqs is {}'.format(combine_seqs))
    #combine_seqs is [[[11661, 38604], [38604, 40054], [40054, 40054], [40054, 5949], [5949, 20922], [20922, 10479], [10479, 40054]], [[11661, 38604, 40054], [38604, 40054, 40054], [40054, 40054, 5949], [40054, 5949, 20922], [5949, 20922, 10479], [20922, 10479, 40054]]]
    
    #print('last_k is {}'.format(last_k))  #[5, 6, 5]

    #aaa
    
    graph_data = {}
    
    for k in range(order):
        #print('seq_nid is {}'.format(seq_nid))
        #k=1
        if k == 0:
            counter = Counter([(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]) ## original connect
        else:       
            counter = Counter([(item_dicts[k][combine(i, k+1)], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
            
        #print('counter is {}'.format(counter))
        
        #print('bbb')
        #print([(item_dicts[k][combine(i, k+1)], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        #print('ccc')
        
        edges = counter.keys()
        
        #print('edges is {}'.format(edges))
        #aaa
        
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = [], []
            weight = th.ones(1).long()
        
        graph_data[('s'+str(k+1), 'intra'+str(k+1), 's'+str(k+1))] = (th.tensor(src).long(), th.tensor(dst).long())

    for k in range(1, order): 
       
        counter = Counter([(seq_nid[i], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
        
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])
            weight = th.ones(1).long()

        ###Inter Here
        graph_data[('s1', 'inter', 's'+str(k+1))] = (src, dst)
        
        counter = Counter([(item_dicts[k][combine(i, k+1)], seq_nid[i+k+1]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
    
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])
            weight = th.ones(1).long()
        
        ###Inter Here
        graph_data[('s'+str(k+1), 'inter', 's1')] = (src, dst)
    
    if order < order1:
        for i in range(order, order1):
            graph_data[('s'+str(i+1), 'intra'+str(i+1), 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s'+str(i+1), 'inter', 's1')]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s1', 'inter', 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
    
    g = dgl.heterograph(graph_data)
    # print(g.num_nodes('s2'))
    #print(g.num_nodes('s1') == len(items))
    if g.num_nodes('s1') < len(items):  #当seq中只含有1个item
        g.add_nodes(len(items)-g.num_nodes('s1'), ntype='s1')
        
    g.nodes['s1'].data['iid'] = th.from_numpy(items)
    
    if order < order1:
        for i in range(order, order1):
            if 's'+str(i+1) not in g.ntypes or g.num_nodes('s'+str(i+1)) == 0:
                g.add_nodes(1, ntype='s'+str(i+1))
                g.nodes['s'+str(i+1)].data['iid'] = th.ones(1, i+1).long() * g.nodes['s1'].data['iid'][0]
                # print(g.nodes['s'+str(i+1)].data)
    for i in range(1, order):
        if g.num_nodes('s'+str(i+1)) == 0:
            g.add_nodes(1, ntype='s'+str(i+1))
        
        g.nodes['s'+str(i+1)].data['iid'] = th.from_numpy(np.array(combine_seqs[i-1]))
    
    label_last_ccs(g, last_k)

    return g
            
def collate_fn_factory(*seq_to_graph_fns):
    def collate_fn(samples):
        seqs, labels = zip(*samples)
        inputs = []
        for seq_to_graph in seq_to_graph_fns:
            graphs = list(map(seq_to_graph, seqs))        
            bg = dgl.batch(graphs)
            inputs.append(bg)
        labels = th.LongTensor(labels)
        return inputs, labels

    return collate_fn

def collate_fn_factory_ccs(seq_to_graph_fns, num_pos, order):
    def collate_fn(samples):
        seqs, labels = zip(*samples)
        
        #print('the size of seqs is {}'.format(len(seqs)))  #tuple, size:512
        #print('the size of labels is {}'.format(len(labels)))  #tuple, size:512
        #print('the size of seq_to_graph_fns is {}'.format(len(seq_to_graph_fns)))  #size:1
        
        inputs = []
        graphs = []

        cnt = 0
        for seq_to_graph in seq_to_graph_fns:
            batch = list(map(seq_to_graph, seqs, [order for _ in range(len(seqs))]))
            if cnt == 0:
                for idx, bh in enumerate(batch):
                    graph = bh
                    graphs.append(graph)
                bg = dgl.batch(graphs)
                cnt = 1
            else:
                bg = dgl.batch(batch)
            inputs.append(bg)
        
        query_seqs = np.zeros((len(seqs), num_pos-1))
        query_pos = np.zeros((len(seqs), num_pos))
        for i in range(len(seqs)):
            seq = seqs[i]
            length = len(seq)
            pos = [length-i+1 for i in range(length+1)]
            query_seqs[i][-length:] = seq
            query_pos[i][-length-1:] = pos

        query_seqs = th.Tensor(query_seqs).type(th.LongTensor)
        query_pos = th.Tensor(query_pos).type(th.LongTensor)
        
        labels = th.LongTensor(labels)
        
        return inputs, query_seqs, query_pos, labels

    return collate_fn

if __name__ == '__main__':
    
    seq = [3, 1, 3, 6, 2, 5, 1, 2, 4, 1, 2] # 2, 0, 2, 5, 1, 4, 0, 1, 3, 0, 1 
    seq0 = [250, 250, 250, 250, 3, 1, 2, 4, 1]
    # g1 = seq_to_ccs_graph(seq, order=4)
    # g2 = seq_to_ccs_graph(seq, order=2)
    collate_fn = collate_fn_factory_ccs(seq_to_ccs_graph, order=2)
    seqs = [[seq, 1], [seq0, 2]]
    print(collate_fn(seqs)[0][0].batch_num_nodes('s2'))

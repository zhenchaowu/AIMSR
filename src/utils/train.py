import time
import numpy as np
import torch as th
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm
from copy import deepcopy
import wandb


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):

    inputs, query_seqs, query_pos, labels = batch
    inputs_gpu  = [x.to(device) for x in inputs]
    query_seqs_gpu = query_seqs.to(device)
    query_pos_gpu = query_pos.to(device)
    labels_gpu  = labels.to(device)
   
    return inputs_gpu, query_seqs_gpu, query_pos_gpu, labels_gpu 



def evaluate(model, data_loader, device, order, reweight, cutoff=20):
    model.eval()
    mrr20 = 0
    hit20 = 0
    mrr10 = 0
    hit10 = 0
    num_samples = 0


    for batch in data_loader:
        inputs, query_seqs, query_pos, labels = prepare_batch(batch, device)
        if order > 1 and reweight:
            logits, _, _ = model(*inputs, query_seqs, query_pos, labels)
        else:
            logits = model(*inputs, query_seqs, query_pos, labels)

        batch_size   = logits.size(0)
        num_samples += batch_size
        labels       = labels.unsqueeze(-1)
        
        topk20         = logits.topk(k=cutoff)[1]  #返回topk的item下标
        hit_ranks20    = th.where(topk20 == labels)[1] + 1   #返回等于label的item下标
        hit20         += hit_ranks20.numel()
        mrr20         += hit_ranks20.float().reciprocal().sum().item()
            
        topk10         = logits.topk(k=10)[1]
        hit_ranks10    = th.where(topk10 == labels)[1] + 1
        hit10         += hit_ranks10.numel()
        mrr10         += hit_ranks10.float().reciprocal().sum().item()
  
    return mrr20 / num_samples, hit20 / num_samples, mrr10 / num_samples, hit10 / num_samples
    
    
class TrainRunner:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device,
        alpha,
        gamma,
        beta,
        lr,
        order,
        reweight,
        incremental_learning,
        IL_interval,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.device       = device
        self.alpha        = alpha
        self.gamma        = gamma
        self.beta         = beta
        self.epoch        = 0
        self.batch        = 0
        self.patience     = patience
        self.order        = order
        self.reweight     = reweight
        self.incremental_learning = incremental_learning
        self.IL_interval  = IL_interval


    def train(self, epochs, log_interval=100):
        max_mrr20 = 0
        max_hit20 = 0
        max_mrr10 = 0
        max_hit10 = 0
        bad_counter = 0
        t = time.time()
        mean_predict_loss = 0
        mean_losses = 0
        mean_meta_infoNCELosses = 0
        mean_batch_infoNCELosses = 0
 
        #mrr20, hit20, mrr10, hit10 = evaluate(self.model, self.test_loader, self.device, self.order, self.reweight)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            batch_num = len(self.train_loader)
            
            for batch in self.train_loader:   
                #print('the current batch is {}'.format(self.batch)) 
                if self.incremental_learning and self.batch % self.IL_interval == 0:        
                    previous_model = deepcopy(self.model)
                inputs, query_seqs, query_pos, labels = prepare_batch(batch, self.device)

                self.optimizer.zero_grad()
                if self.order > 1 and self.reweight:
                    scores, meta_infoNCELosses, batch_infoNCELosses = self.model(*inputs, query_seqs, query_pos, labels)
                else:
                    scores = self.model(*inputs, query_seqs, query_pos, labels)
                
                assert not th.isnan(scores).any()
                predict_loss = nn.functional.nll_loss(scores, labels)
                
                if self.order > 1 and self.reweight:
                    losses = predict_loss + self.alpha * meta_infoNCELosses + self.gamma * batch_infoNCELosses
                    losses.backward()            
                else:
                    predict_loss.backward()
         
                self.optimizer.step()
           
                if self.order > 1 and self.reweight:
                    mean_losses += losses.item() / log_interval
                    mean_predict_loss += predict_loss.item() / log_interval
                    mean_meta_infoNCELosses += self.alpha * meta_infoNCELosses.item() / log_interval
                    mean_batch_infoNCELosses += self.gamma * batch_infoNCELosses.item() / log_interval
                
                    if self.batch > 0 and self.batch % log_interval == 0:
                        print(f'Batch {self.batch}: Mean_Losses = {mean_losses:.4f}, Mean Predict Loss = {mean_predict_loss:.4f}, Mean_Meta_NCELoss = {mean_meta_infoNCELosses:.4f}, Mean_Batch_NCELoss = {mean_batch_infoNCELosses:.4f}, Time Elapsed = {time.time() - t:.2f}s')
                        t = time.time()
                        mean_losses = 0
                        mean_predict_loss = 0
                        mean_meta_infoNCELosses = 0
                        mean_batch_infoNCELosses = 0
                else:
                    mean_predict_loss += predict_loss.item() / log_interval
                    if self.batch > 0 and self.batch % log_interval == 0:
                        print(f'Batch {self.batch}:  Mean Predict Loss = {mean_predict_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s')
                        t = time.time()
                        mean_predict_loss = 0
                        
                
                self.batch += 1
            self.scheduler.step() #对学习率进行调整
            mrr20, hit20,  mrr10, hit10 = evaluate(self.model, self.test_loader, self.device, self.order, self.reweight)
            
            # wandb.log({"hit": hit, "mrr": mrr})

            print(f'Epoch {self.epoch}: MRR20 = {mrr20 * 100:.3f}%, Hit20 = {hit20 * 100:.3f}%')
            print(f'Epoch {self.epoch}: MRR10 = {mrr10 * 100:.3f}%, Hit10 = {hit10 * 100:.3f}%')

            if mrr20 < max_mrr20 and hit20 < max_hit20 and mrr10 < max_mrr10 and hit10 < max_hit10:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr20 = max(max_mrr20, mrr20)
            max_hit20 = max(max_hit20, hit20)
            max_mrr10 = max(max_mrr10, mrr10)
            max_hit10 = max(max_hit10, hit10)
            self.epoch += 1
            
            if self.incremental_learning and self.batch % self.IL_interval == 0:
                for i, (p, q) in enumerate(zip(self.model.parameters(), previous_model.parameters())):
                    eta = np.exp(-self.beta*((1.0 * (self.batch + 1)) / batch_num)) 
                    p.data = p.data * eta + (1 - eta) * q.data
                        
            self.model.store_parameters()  
            
            
            #self.model.store_parameters()
        return max_mrr20, max_hit20, max_mrr10, max_hit10

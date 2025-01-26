import math
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init
from scipy import sparse
from copy import deepcopy
from entmax import entmax_bisect
from .gnn_models import GATConv


class GlobalItemConv(nn.Module):
    def __init__(self, spare=True, layers=1, feat_drop=0.0):
        super(GlobalItemConv, self).__init__()
        self.spare = spare
        self.layers = layers
        self.feat_drop = nn.Dropout(feat_drop)

    def forward(self, x, adj):
        h = x.cuda()
        final = [x]
        for i in range(self.layers):
            h = th.sparse.mm(adj, h)
            h = self.feat_drop(h)
            final.append(h)
        if self.layers > 1:
            h = th.sum(th.stack(final), dim=0) / (self.layers + 1)
        return h




class SemanticExpander(nn.Module):
    def __init__(self, input_dim, reducer, order):
        super(SemanticExpander, self).__init__()
        self.input_dim = input_dim
        self.order = order
        self.reducer = reducer
        self.GRUs = nn.ModuleList()
        for i in range(self.order):
            self.GRUs.append(nn.GRU(self.input_dim, self.input_dim, 1, True, True))  #torch.nn.GRU(input_size, hidden_size, num_layers, bias, batch_first)
    
        if self.reducer == 'concat':
            self.Ws = nn.ModuleList()
            for i in range(1, self.order):
                self.Ws.append(nn.Linear(self.input_dim * (i+1), self.input_dim))
        
        
    def forward(self, feat):  
        
        if len(feat.shape) < 3:
            return feat, feat
        if self.reducer == 'mean':
            invar = th.mean(feat, dim=1)
        elif self.reducer == 'max':
            invar =  th.max(feat, dim=1)[0]
        elif self.reducer == 'concat':
            invar =  self.Ws[feat.size(1)-2](feat.view(feat.size(0), -1))
        var = self.GRUs[feat.size(1)-2](feat)[1].permute(1, 0, 2).squeeze()
        
        return invar, var
        
###########################################################################################################################################################
      
class MSHGNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0.0, activation=None, order=1, granularity_type='hybrid', reducer='mean'):
        super(MSHGNN, self).__init__()    
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        self.activation = activation
        self.order = order
        self.granularity_type = granularity_type
        
        self.lamda1 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda2 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda3 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda4 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda1.data.fill_(0.5)
        self.lamda2.data.fill_(0.5)
        self.lamda3.data.fill_(0.5)
        self.lamda4.data.fill_(0.5)
        
        conv1_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True) for i in range(self.order)}
        conv1_modules.update({'inter'     : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True)})
        self.conv1 = dglnn.HeteroGraphConv(conv1_modules, aggregate='sum')
        
        conv2_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True) for i in range(self.order)}
        conv2_modules.update({'inter'     : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True)})
        self.conv2 = dglnn.HeteroGraphConv(conv2_modules, aggregate='sum')
        
        conv3_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True) for i in range(self.order)}
        conv3_modules.update({'inter'     : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True)})
        self.conv3 = dglnn.HeteroGraphConv(conv3_modules, aggregate='sum')
        
        conv4_modules = {'intra'+str(i+1) : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True) for i in range(self.order)}
        conv4_modules.update({'inter'     : GATConv(input_dim, output_dim, 1, dropout, dropout, residual=True)})
        self.conv4 = dglnn.HeteroGraphConv(conv4_modules, aggregate='sum')
        
        self.lint = nn.Linear(output_dim, 1, bias=False)

    def forward(self, g, feat_src, feat_dst):
        
        with g.local_scope():
            
            h10 = self.conv1(g, (feat_src, feat_dst))
            h11 = self.conv2(g, (feat_dst, feat_dst))
            h20 = self.conv3(g.reverse(copy_edata=True), (feat_src, feat_dst))
            h21 = self.conv4(g.reverse(copy_edata=True), (feat_dst, feat_dst))
            
            h = {}
            for i in range(self.order):
                
                hl, hr = th.zeros(1, self.output_dim).to(self.lint.weight.device), th.zeros(1, self.output_dim).to(self.lint.weight.device)
                
                if 's'+str(i+1) in h10:
                    if self.granularity_type == 'hybrid':
                        hl = self.lamda1 * h10['s'+str(i+1)] + self.lamda2 * h11['s'+str(i+1)]
                    elif self.granularity_type == 'mean' or self.granularity_type == 'gru':
                        hl = self.lamda1 * h11['s'+str(i+1)]
                    
                if 's'+str(i+1) in h11:
                    if self.granularity_type == 'hybrid':
                        hr = self.lamda3 * h20['s'+str(i+1)] + self.lamda4 * h21['s'+str(i+1)]
                    elif self.granularity_type == 'mean' or self.granularity_type == 'gru':
                        hr = self.lamda2 * h21['s'+str(i+1)]
                
                h['s'+str(i+1)] = hl + hr
                
                if len(h['s'+str(i+1)].shape) > 2:
                    h['s'+str(i+1)] = h['s'+str(i+1)].max(1)[0]

                       
        return h
        
        
################################################################################################################################################
                
    
class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        feat_drop=0.0,
        activation=None,
        order=1,
        granularity_type='hybrid',
        device=th.device('cpu')
    ):
        super(AttnReadout, self).__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.order = order
        self.granularity_type = granularity_type
        self.device = device
        self.fc_u = nn.ModuleList()
        self.fc_v = nn.ModuleList()
        self.fc_e = nn.ModuleList()
        
        for i in range(self.order):
            self.fc_u.append(nn.Linear(input_dim, hidden_dim, bias=True))
            self.fc_v.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.fc_e.append(nn.Linear(hidden_dim, 1, bias=False))
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation
        
    def forward(self, g, feats_invar, feats_var, last_nodess):   
        
        rsts_invar = []
        rsts_var = []
      
        nfeats_invar = []
        nfeats_var = []
        for i in range(self.order): 
            feat_invar = feats_invar['s'+str(i+1)]
            feat_var = feats_var['s'+str(i+1)]
            
            feat_invar = th.split(feat_invar, g.batch_num_nodes('s'+str(i+1)).tolist())
            feat_var = th.split(feat_var, g.batch_num_nodes('s'+str(i+1)).tolist())
            
            feats_invar['s'+str(i+1)] = th.cat(feat_invar, dim=0)
            feats_var['s'+str(i+1)] = th.cat(feat_var, dim=0)
                       
            nfeats_invar.append(feat_invar)
            nfeats_var.append(feat_var)
            
        feat_vs_invar = th.cat(tuple(feats_invar['s'+str(i+1)][last_nodess[i]].unsqueeze(1) for i in range(self.order)), dim=1)
        feat_vs_var = th.cat(tuple(feats_var['s'+str(i+1)][last_nodess[i]].unsqueeze(1) for i in range(self.order)), dim=1)
        
        if self.granularity_type == 'hybrid':
            feats = th.cat([th.cat((th.cat(tuple(nfeats_invar[j][i] for j in range(self.order)), dim=0), th.cat(tuple(nfeats_var[j][i] for j in range(self.order)), dim=0)), 0) for i in range(len(g.batch_num_nodes('s1')))], dim=0) 
        elif self.granularity_type == 'mean' or self.granularity_type == 'gru':
            feats_invar = th.cat([th.cat(tuple(nfeats_invar[j][i] for j in range(self.order)), dim=0) for i in range(len(g.batch_num_nodes('s1')))], dim=0)
            feats_var = th.cat([th.cat(tuple(nfeats_var[j][i] for j in range(self.order)), dim=0) for i in range(len(g.batch_num_nodes('s1')))], dim=0)
                       
        batch_num_nodes = th.cat(tuple(g.batch_num_nodes('s'+str(i+1)).unsqueeze(1) for i in range(self.order)), dim=1).sum(1)
        
        if self.granularity_type == 'hybrid':
            batch_num_nodes = batch_num_nodes * 2
        
        idx = th.cat(tuple(th.ones(batch_num_nodes[j])*j for j in range(len(batch_num_nodes)))).long()
        
        for i in range(self.order):
            if self.granularity_type == 'hybrid':
                feat_u = self.fc_u[i](feats)   
            elif self.granularity_type == 'mean' or self.granularity_type == 'gru':
                feat_u_invar = self.fc_u[i](feats_invar)   
                feat_u_var = self.fc_u[i](feats_var)   
             
            feat_v_invar = self.fc_v[i](feat_vs_invar[:, i])[idx]   
            feat_v_var = self.fc_v[i](feat_vs_var[:, i])[idx]  
                 
            if self.granularity_type == 'hybrid':  
                e_invar = self.fc_e[i](th.sigmoid(feat_u + feat_v_invar))   
                e_var = self.fc_e[i](th.sigmoid(feat_u + feat_v_var))   
            elif self.granularity_type == 'mean' or self.granularity_type == 'gru':
                e_invar = self.fc_e[i](th.sigmoid(feat_u_invar + feat_v_invar))   
                e_var = self.fc_e[i](th.sigmoid(feat_u_var + feat_v_var))   
                         
            alpha_invar = dgl.ops.segment.segment_softmax(batch_num_nodes, e_invar)
            alpha_var = dgl.ops.segment.segment_softmax(batch_num_nodes, e_var)
            
            if self.granularity_type == 'hybrid': 
                feat_norm_invar = feats * alpha_invar
            elif self.granularity_type == 'mean' or self.granularity_type == 'gru':
                feat_norm_invar = feats_invar * alpha_invar
            rst_invar = dgl.ops.segment.segment_reduce(batch_num_nodes, feat_norm_invar, 'sum')
            
            if self.granularity_type == 'hybrid': 
                feat_norm_var = feats * alpha_var
            elif self.granularity_type == 'mean' or self.granularity_type == 'gru':
                feat_norm_var = feats_var * alpha_var
            rst_var = dgl.ops.segment.segment_reduce(batch_num_nodes, feat_norm_var, 'sum')
            
            rsts_invar.append(rst_invar.unsqueeze(1))
            rsts_var.append(rst_var.unsqueeze(1))
        
            if self.fc_out is not None:  
                rst_invar = self.fc_out(rst_invar)
                rst_var = self.fc_out(rst_var)
            if self.activation is not None:  
                rst_invar = self.activation(rst_invar)
                rst_var = self.activation(rst_var)
        rst_invar = th.cat(rsts_invar, dim=1)
        rst_var = th.cat(rsts_var, dim=1)

        return rst_invar, rst_var


######################################################################################################################################################################

class DualAttention(nn.Module):

    def __init__(self, item_embeddings, pos_embeddings, pos_num, norm, data_aug, adj, dropout=0.2, activate='relu'):
        super(DualAttention, self).__init__()
        
        self.item_embeddings = item_embeddings
        self.pos_embeddings = pos_embeddings
        self.item_dim = self.item_embeddings.weight.size(1)
        self.pos_dim = pos_embeddings.weight.size(1)
        self.dim = self.item_dim + self.pos_dim
        self.pos_num = pos_num
        self.norm = norm
        self.data_aug = data_aug
        self.adj = adj

        self.dropout = nn.Dropout(dropout)
        self.self_atten_q = nn.Linear(self.dim, self.item_dim)
        self.self_atten_k = nn.Linear(self.dim, self.item_dim)
        self.self_atten_v = nn.Linear(self.dim, self.item_dim)
        self.final_pos_embedding_layer = nn.Linear(self.item_dim*2,  self.item_dim)
        
        self.is_dropout = False
        self.attention_mlp = nn.Linear(self.dim, self.dim)
        self.alpha_w = nn.Linear(self.dim, 1)
        
        self.lamda1 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda2 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda1.data.fill_(0.5)
        self.lamda2.data.fill_(0.5)
        
        self.item_conv = GlobalItemConv()
        
        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu

        self.initial_()

    def initial_(self):
        init.constant_(self.attention_mlp.bias, 0)
        
        
    def forward(self, x, pos, adj=None):
    
        if self.data_aug:
            graph_item_embs = self.item_conv(self.item_embeddings.weight, self.adj)

        if self.data_aug:
            x_embeddings = graph_item_embs[x]
        else:
            x_embeddings = self.item_embeddings(x)  # B,seq,dim

        pos_embeddings = self.pos_embeddings(pos)  # B, seq, dim
        mask = (x != 0).float()  # B,seq
     
        final_embeddings = self.get_final_embedding(x_embeddings, mask)
        final_x_embeddings = th.cat((x_embeddings, final_embeddings.unsqueeze(1)), 1)  
        
        x_ = th.cat((final_x_embeddings, pos_embeddings), 2)  # B seq, 2*dim
        alpha_ent = self.get_alpha(x = x_[:, -1, :], number= 0)
        
        mask = th.cat((mask, th.ones(x.size(0)).unsqueeze(1).cuda()), 1)
        m_s = self.self_attention(x_, x_, x_, mask, alpha_ent)

        return m_s    
        
        
    def get_final_embedding(self, x_embeddings, mask):
    
        embeddings_mask = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), x_embeddings.size(2)) 
        real_x_embeddings = x_embeddings * embeddings_mask    
        mean_embeddings = th.mean(real_x_embeddings, dim=1)         
   
        return mean_embeddings

        

    def get_alpha(self, x=None, number=None):   #x:[512, 200]
        if number == 0:
            alpha_ent = th.sigmoid(self.alpha_w(x)) + 1
            alpha_ent = self.add_value(alpha_ent).unsqueeze(1)
            alpha_ent = alpha_ent.expand(-1, self.pos_num, -1)                
            return alpha_ent
        if number == 1:
            alpha_global = th.sigmoid(self.alpha_w(x)) + 1
            alpha_global = self.add_value(alpha_global)
            return alpha_global

    def add_value(self, value):
        mask_value = (value ==1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value
        
    def self_attention(self, q, k, v, mask=None, alpha_ent = 1):

        if self.is_dropout:   #True
            q_ = self.dropout(self.activate(self.self_atten_q(q)))
            k_ = self.dropout(self.activate(self.self_atten_k(k)))
            v_ = self.dropout(self.activate(self.self_atten_v(v)))
        else:
            q_ = self.activate(self.self_atten_q(q))
            k_ = self.activate(self.self_atten_k(k))
            v_ = self.activate(self.self_atten_v(v))
            
  
        scores = th.matmul(q_, k_.transpose(1, 2)) / math.sqrt(self.item_dim)
        
        if mask is not None:    
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            scores = scores.masked_fill(mask == 0, -np.inf)  
                
        alpha = entmax_bisect(scores, alpha_ent, dim=-1)

        att_v = th.matmul(alpha, v_)  # B, seq, dim

        if self.is_dropout:
            att_v = self.dropout(self.activate(att_v))
        else:
            att_v = self.activate(att_v)
            
        if self.norm:
            att_v = nn.functional.normalize(att_v, dim=-1)
         
        c = att_v[:, -1, :]

        return c



#########################################################################################################################################################################
class MetaWeightNet(nn.Module):
    def __init__(self, order, hidden_dim, drop_rate=0.8):
        super(MetaWeightNet, self).__init__()
        self.order = order
        self.hidden_dim = hidden_dim
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
          
        # diginetica
        #self.SSL_layer1 = nn.Linear(self.hidden_dim*3, self.hidden_dim*2)
        #self.SSL_layer2 = nn.Linear(self.hidden_dim*2, 1)
        #self.SSL_layer3 = nn.Linear(self.hidden_dim*2, 1)
        
        # retailrocket and tmall
        self.SSL_layer1 = nn.Linear(self.hidden_dim*3, int((self.hidden_dim*3)/2))
        self.SSL_layer2 = nn.Linear(int((self.hidden_dim*3)/2), 1)
        self.SSL_layer3 = nn.Linear(self.hidden_dim*2, 1)
        
        
    def compute_loss(self, query_embedding, session_embedding):
           
        losses = F.mse_loss(query_embedding.repeat(self.order, 1), session_embedding, reduction='none')
        losses = losses.sum(1).cuda()

        return losses
        

    def forward(self, query_embedding, session_embedding, vars_dict=None): 
    
        session_con_losses = self.compute_loss(query_embedding, session_embedding)  
        
        SSL_input = th.cat((session_con_losses.unsqueeze(1).repeat(1, self.hidden_dim), session_embedding), 1)  
        SSL_input2 = th.cat((SSL_input, query_embedding.repeat(self.order, 1)), 1)
        SSL_input3 = session_con_losses.unsqueeze(1).repeat(1, 2* self.hidden_dim) * th.cat((session_embedding, query_embedding.repeat(self.order,1)), 1)   #size:[3, 200]
      
        if vars_dict is not None:
            session_con_loss_weights = self.dropout(F.linear(SSL_input2, vars_dict['SSL_layer1.weight'], vars_dict['SSL_layer1.bias']))  
            session_con_loss_weights = np.sqrt(SSL_input2.shape[1])*self.dropout(F.linear(session_con_loss_weights, vars_dict['SSL_layer2.weight'], vars_dict['SSL_layer2.bias']))
            SSL_weight3 = self.dropout(F.linear(SSL_input3, vars_dict['SSL_layer3.weight'], vars_dict['SSL_layer3.bias']))
            
        else:
            session_con_loss_weights = self.dropout(self.SSL_layer1(SSL_input2))
            session_con_loss_weights = np.sqrt(SSL_input2.shape[1])*self.dropout(self.SSL_layer2(session_con_loss_weights))
            SSL_weight3 = self.dropout(self.SSL_layer3(SSL_input3))
     
        
        session_con_loss_weights = self.sigmoid(session_con_loss_weights)
        SSL_weight3 = self.sigmoid(SSL_weight3)
         
        session_con_loss_weights = (session_con_loss_weights + SSL_weight3)/2   
        session_con_loss_weights = F.softmax(session_con_loss_weights, dim=0)
        meta_infoNCELoss = th.mm(session_con_losses.unsqueeze(0), session_con_loss_weights)  
        
        return meta_infoNCELoss, session_con_loss_weights

##########################################################################################################################################################################

class AIMSR(nn.Module):
    
    def __init__(self, num_items, num_pos, item_dim, pos_dim, num_layers, adj, w, local_lr, num_local_update, SSL_batch, use_meta_learning, reweight, dropout=0.3, reducer='mean', order=3, norm=True, extra=True, fusion=True, data_aug=True, granularity_type='hybrid', device=th.device('cpu')):
        super().__init__()
        
        
        self.lamda1 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda2 = th.nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.lamda1.data.fill_(0.5)
        self.lamda2.data.fill_(0.5)
        
        self.num_items = num_items
        self.num_pos = num_pos
        self.item_dim = item_dim
        self.pos_dim = pos_dim
        self.w = w
        self.num_layers = num_layers
        self.layers   = nn.ModuleList()
        self.SSL_batch = SSL_batch
        
        self.reducer  = reducer
        self.order    = order
        self.norm     = norm
        self.local_lr = local_lr
        self.num_local_update = num_local_update
        self.use_meta_learning = use_meta_learning
        self.reweight = reweight
        self.adj      = adj
        self.data_aug = data_aug
        self.granularity_type = granularity_type
        
        self.item_embeddings = nn.Embedding(self.num_items, self.item_dim, padding_idx=0, max_norm=1.0)  
        self.pos_embeddings = nn.Embedding(self.num_pos+1, self.pos_dim, padding_idx=0, max_norm=1.0)
        
        self.item_conv = GlobalItemConv()
        self.query = DualAttention(self.item_embeddings, self.pos_embeddings, self.num_pos, self.norm, self.data_aug, self.adj)
        self.meta_weight_net = MetaWeightNet(self.order, self.item_dim)
        

        self.register_buffer('indices', th.arange(self.num_items, dtype=th.long))
        
        input_dim     = self.item_dim
        self.expander = SemanticExpander(input_dim, self.reducer, self.order)    #生成Consecutive Intent Unit的表示
        
        self.device = device
        for i in range(self.num_layers):
            layer = MSHGNN(
                input_dim,
                item_dim,
                dropout=dropout,
                order=self.order,
                granularity_type=self.granularity_type,
                activation=nn.PReLU(item_dim)
            )
            self.layers.append(layer)
            
        self.readout = AttnReadout(
            input_dim,
            item_dim,
            item_dim,
            feat_drop=dropout,
            activation=None,
            order=self.order,
            granularity_type=self.granularity_type,
            device=self.device
        )
        input_dim += item_dim    #input_dim=2*item_dim
        self.feat_drop = nn.Dropout(dropout)

        self.fc_sr = nn.ModuleList()
        for i in range(self.order):
            self.fc_sr.append(nn.Linear(input_dim, item_dim, bias=False))  #将global和local级联起来生成session表示
        
        self.sc_sr = nn.ModuleList()
        for i in range(self.order):
            self.sc_sr.append(nn.Sequential(nn.Linear(item_dim, item_dim, bias=True),  nn.ReLU(), nn.Linear(item_dim, 2, bias=False), nn.Softmax(dim=-1)))  #ReNorm中的beta_r和beta_o

        self.reset_parameters()
        self.fusion = fusion
        self.extra = extra
        self.store_parameters()

  
    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.item_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    
    def store_parameters(self):
        
        self.keep_weight = deepcopy(self.meta_weight_net.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        #print('the weight_name is {}'.format(self.weight_name))
        self.weight_len = len(self.keep_weight)   
        for i in range(self.weight_len):
            self.keep_weight[self.weight_name[i]] = self.keep_weight[self.weight_name[i]].cuda()  
            
  
    def forward(self, mg, query_seqs, query_pos, labels):
        
        graph_item_embs = self.item_conv(self.item_embeddings.weight, self.adj)
        
        if self.data_aug:
            target = graph_item_embs[self.indices]
        else:
            target = self.item_embeddings(self.indices)
            
        if self.norm:
            target = nn.functional.normalize(target, dim=-1)
        
        if self.order > 1:
            query_embeddings = self.query(query_seqs, query_pos)  
            label_embeddings = target[labels]
        
        feats_invar = {}
        feats_var = {}
        for i in range(self.order):

            iid = mg.nodes['s' + str(i+1)].data['iid']  

            if self.data_aug:
                feat = graph_item_embs[iid]
            else:
                feat = self.item_embeddings(iid)   
       
            feat = self.feat_drop(feat)   
            feat_invar, feat_var = self.expander(feat) 

            if th.isnan(feat_invar).any():
                feat_invar = feat_invar.masked_fill(feat_invar != feat_invar, 0)
            if th.isnan(feat_var).any():
                feat_var = feat_var.masked_fill(feat_var != feat_var, 0)
                  
            if self.norm:
                feat_invar = nn.functional.normalize(feat_invar, dim=-1)
                feat_var = nn.functional.normalize(feat_var, dim=-1)
            feats_invar['s' + str(i+1)] = feat_invar
            feats_var['s' + str(i+1)] = feat_var
   
        h_invar = feats_invar
        h_var = feats_var

        for idx, layer in enumerate(self.layers):    # MSHGNN

            h_invar = layer(mg, h_var, h_invar)
            h_var = layer(mg, h_invar, h_var)
                
   
        last_nodes = []
        for i in range(self.order):
            if self.norm:
                h_invar['s'+str(i+1)] = nn.functional.normalize(h_invar['s'+str(i+1)], dim=-1)
                h_var['s'+str(i+1)] = nn.functional.normalize(h_var['s'+str(i+1)], dim=-1)      
            last_nodes.append(mg.filter_nodes(lambda nodes: nodes.data['last'] == 1, ntype='s'+str(i+1)))
            
            
            
        feat_invar = h_invar
        feat_var = h_var
        
        sr_g_invar, sr_g_var = self.readout(mg, feat_invar, feat_var, last_nodes)
        
        sr_l_invar = th.cat([feat_invar['s'+str(i+1)][last_nodes[i]].unsqueeze(1) for i in range(self.order)], dim=1)
        sr_invar = th.cat([sr_l_invar, sr_g_invar], dim=-1)
        sr_invar = th.cat([self.fc_sr[0](sr).unsqueeze(1) for i, sr in enumerate(th.unbind(sr_invar, dim=1))], dim=1)  
        
        sr_l_var = th.cat([feat_var['s'+str(i+1)][last_nodes[i]].unsqueeze(1) for i in range(self.order)], dim=1)
        sr_var = th.cat([sr_l_var, sr_g_var], dim=-1)
        sr_var = th.cat([self.fc_sr[0](sr).unsqueeze(1) for i, sr in enumerate(th.unbind(sr_var, dim=1))], dim=1)  
         
        if self.granularity_type == 'hybrid':
            sr = self.lamda1 * sr_invar + self.lamda2 * sr_var  
        elif self.granularity_type == 'mean':
            sr = sr_invar
        elif self.granularity_type == 'gru':
            sr = sr_var
       
        if self.norm:
            sr = nn.functional.normalize(sr, dim=-1)

            
        ####################################################################################################
        
        if self.order > 1:
                
            meta_infoNCELosses = 0
            meta_infoNCELoss_weights = None
            for sess_id in range(sr.size(0)):
                query_embedding = query_embeddings[sess_id].unsqueeze(0)      
                session_embedding = sr[sess_id, :, :]        
                
                if self.reweight:
                    meta_infoNCELoss, session_con_loss_weights = self.meta_weight_net(query_embedding, session_embedding)
                
                if self.use_meta_learning:
                    self.meta_weight_net.zero_grad()
                    grad = th.autograd.grad(meta_infoNCELoss, self.meta_weight_net.parameters(), create_graph=True) 
                        
                    f_fast_weights = deepcopy(self.keep_weight)
            
                    meta_infoNCELoss = 0
                    session_con_loss_weights = None
                    # local update
                    for idx in range(self.num_local_update):  
                        for i in range(self.weight_len):
                            f_fast_weights[self.weight_name[i]] = f_fast_weights[self.weight_name[i]] - self.local_lr * grad[i] 
                
                        meta_infoNCELoss, session_con_loss_weights = self.meta_weight_net(query_embedding, session_embedding, f_fast_weights)
                
                        if idx < self.num_local_update-1:
                            grad = th.autograd.grad(meta_infoNCELoss, f_fast_weights.values(), create_graph=True) 

                  
                if self.reweight:
                    meta_infoNCELosses += meta_infoNCELoss
                    try:
                        meta_infoNCELoss_weights = th.cat((meta_infoNCELoss_weights, session_con_loss_weights.view(1, self.order)), 0)
                    except:
                        meta_infoNCELoss_weights = session_con_loss_weights.view(1, self.order)
                
          
            if self.reweight:
                query_con_losses = self.SSL(query_embeddings, sr)   
                label_con_losses = self.SSL(label_embeddings, sr)
                batch_query_infoNCELosses = (meta_infoNCELoss_weights *  query_con_losses).sum(1).sum()  
                batch_label_infoNCELosses = (meta_infoNCELoss_weights *  label_con_losses).sum(1).sum()  
                mean_infoNCELosses1 = (batch_query_infoNCELosses + batch_label_infoNCELosses) / meta_infoNCELoss_weights.shape[0]
                
                label_query_con_losses = self.single_infoNCE_loss_one_by_one(label_embeddings, query_embeddings).sum() 
                query_label_con_losses = self.single_infoNCE_loss_one_by_one(query_embeddings, label_embeddings).sum() 
                mean_infoNCELosses2 = (label_query_con_losses + query_label_con_losses) / meta_infoNCELoss_weights.shape[0]
                
                mean_infoNCELosses = mean_infoNCELosses1 + mean_infoNCELosses2
                

        ###################################################################################################################################
        
        
        
               
        if self.extra:
            logits = sr @ target.t()
            phi = self.sc_sr[0](sr).unsqueeze(-1)

            mask = th.zeros(phi.size(0), self.num_items).to(self.device)   
            iids = th.split(mg.nodes['s1'].data['iid'], mg.batch_num_nodes('s1').tolist())
            
            for i in range(len(mask)):
                mask[i, iids[i]] = 1

            logits_in = logits.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
            logits_ex = logits.masked_fill(mask.bool().unsqueeze(1), float('-inf'))
            
            score     = th.softmax(self.w * logits_in.squeeze(), dim=-1)
            score_ex  = th.softmax(self.w * logits_ex.squeeze(), dim=-1) 

          
            if th.isnan(score).any():
                score = feat.masked_fill(score != score, 0)
            if th.isnan(score_ex).any():
                score_ex = score_ex.masked_fill(score_ex != score_ex, 0)
          
            assert not th.isnan(score).any()
            assert not th.isnan(score_ex).any()
 
            if self.order == 1:
                phi = phi.squeeze(1)  
                score = (th.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
            else:
                score = (th.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)

        else:
            logits = sr.squeeze() @ target.t()
            score  = th.softmax(self.w * logits, dim=-1)
            
        if self.order > 1 and self.fusion:
            if self.reweight:
                score = (score * meta_infoNCELoss_weights.unsqueeze(2)).sum(1)
            else:
                score = score.sum(1)     
        elif self.order > 1:
            score = score[:, 0]  
            
        score = th.log(score)
        
        if self.order > 1:
            if self.reweight:
                return score, meta_infoNCELosses, mean_infoNCELosses
            else:
                return score
        else:
            return score
        
        
    def SSL(self, query_embeddings, session_embeddings):
            
        session_con_losses = None
        for i in range(self.order):
            session_con_loss = self.single_infoNCE_loss_one_by_one(session_embeddings[:, i, :], query_embeddings)
   
            try:
                session_con_losses = th.cat((session_con_losses, session_con_loss.unsqueeze(1)), 1)
            except:
                session_con_losses = session_con_loss.unsqueeze(1)

        return session_con_losses
        
        
    
   

    def single_infoNCE_loss_one_by_one(self, session_embedding, query_embedding):  

        def multi_neg_sample_pair_index(batch_index, step_index, query_embedding, session_embedding):  

            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set                         
            neg2_index = th.as_tensor(np.array(list(neg2_index_set))).long().cuda()  
            neg2_index = th.unsqueeze(neg2_index, 0)                             
            neg2_index = neg2_index.repeat(len(batch_index), 1)              
            neg2_index = th.reshape(neg2_index, (1, -1))                          
            neg2_index = th.squeeze(neg2_index)                                   
                                                                              
            neg1_index = batch_index.long().cuda()                              
            neg1_index = th.unsqueeze(neg1_index, 1)                          
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))               
            neg1_index = th.reshape(neg1_index, (1, -1))                                   
            neg1_index = th.squeeze(neg1_index)                                   

            neg_score_pre = th.sum(compute(session_embedding, query_embedding, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)  
                
            return neg_score_pre  


        def compute(x1, x2, neg1_index=None, neg2_index=None):  

            if neg1_index!=None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]  
            D = x1.shape[1]

            x1 = x1
            x2 = x2

            scores = th.exp(th.div(th.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1)+1e-8))  
            
            return scores
            
        N = query_embedding.shape[0]  
        D = query_embedding.shape[1]  
            
        step_index = th.Tensor([i for i in range(N)]).type(th.IntTensor)

        pos_score = compute(query_embedding, session_embedding).squeeze()  
        neg_score = th.zeros((N,), dtype=th.float64).cuda()  
            

        #-------------------------------------------------multi version-----------------------------------------------------
        steps = int(np.ceil(N / self.SSL_batch))  #separate the batch to smaller one  SSL_batch:30
        for i in range(steps):
            st = i * self.SSL_batch
            ed = min((i+1) * self.SSL_batch, N)
            batch_index = step_index[st: ed]

            neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, query_embedding, session_embedding)
            if i ==0:
                neg_score = neg_score_pre
            else:
                neg_score = th.cat((neg_score, neg_score_pre), 0)
        #-------------------------------------------------multi version-----------------------------------------------------

        con_loss = -th.log(1e-8 +th.div(pos_score, neg_score+1e-8))  

        assert not th.any(th.isnan(con_loss))
        assert not th.any(th.isinf(con_loss))

        return th.where(th.isnan(con_loss), th.full_like(con_loss, 0+1e-8), con_loss)

        
        

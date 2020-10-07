import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import layers   # for model establishment
from utils import *

class GraphSAINT(nn.Module):
    """
        Trainer model
        Transfer data to GPU:   A  init:  1) feat_full   2) label_full   3) label_full_cat
    """
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Inputs:
            arch_gcn            parsed arch of GCN
            train_params        parameters for training
            cpu_eval (bool)  :   whether use CPU side for evalution
        """
        super(GraphSAINT,self).__init__()
        
        self.use_cuda = torch.cuda.is_available()
        if cpu_eval:
            self.use_cuda=False
        
        # whether to use gated_attention, attention or highorder  aggregator
        if "attention" in arch_gcn:
            if "gated_attention" in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls = layers.GatedAttentionAggregator
                    self.mulhead = int(arch_gcn['attention'])
            else:
                self.aggregator_cls = layers.AttentionAggregator
                self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls=layers.HighOrderAggregator
            self.mulhead=1
        
        # each layer in arch_gcn['arch']:  is a string separated by '-'
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        
        # check if the task is a multi-label task
        # sigmoid: means this is a multi-label task
        self.sigmoid_loss = (arch_gcn['loss']=='sigmoid')   # use sigmoid for multi-label loss function
        
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))
        
        
        
        self.num_classes = num_classes
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
                        = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()
        self.set_dims(_dims)

        self.loss = 0
        self.opt_op = None

        # build the model below        
        self.aggregators = self.get_aggregators()
        self.conv_layers = nn.Sequential(*self.aggregators)
        # calculate the final embeddings:
        self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.num_classes,\
                            act='I', order=0, dropout=self.dropout, bias='bias')
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)

    def set_dims(self,dims):
        """
            dims_feat: Obtain the dimension of each embedding layer
            dims_weight:  Obtain the dimension of each weight
        """
        self.dims_feat = [dims[0]] + [( (self.aggr_layer[l]=='concat') * self.order_layer[l] + 1) * dims[l+1] for l in range(len(dims)-1)]
        self.dims_weight = [(self.dims_feat[l], dims[l+1]) for l in range(len(dims)-1)]

    def set_idx_conv(self):
        idx_conv = np.where(np.array(self.order_layer)>=1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer)-1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer)==1)[0])


    def forward(self, adj_subgraph, feat_subg):
        
        _, emb_subg = self.conv_layers((adj_subgraph, feat_subg))
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        
        # obtain the prediction
        pred_subg = self.classifier((None, emb_subg_norm))[1]
        return pred_subg


    def _loss(self, preds, labels, norm_loss):
        """
            use the norm_loss as the weight factor
        """
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight = norm_loss,reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss * _ls).sum()


    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        aggregators = []
        for l in range(self.num_layers):
            aggrr = self.aggregator_cls(*self.dims_weight[l], dropout=self.dropout,\
                    act=self.act_layer[l], order=self.order_layer[l], \
                    aggr=self.aggr_layer[l], bias=self.bias_layer[l], mulhead=self.mulhead)
            aggregators.append(aggrr)
        return aggregators

    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)
        
        
    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph, feat_subg, label_subg_converted):
        """
        Purpose:  only count the time for the training process, including forward and backward propogation
        Forward and backward propagation
        norm_loss_subgraph : is the key to rescale the current batch/subgraph
        """
        self.train()
        # =============== start of the training process for one step: =================
        self.optimizer.zero_grad()
        # here call the forward propagation
        preds = self(adj_subgraph, feat_subg)    # will call the forward function
        loss = self._loss(preds, label_subg_converted, norm_loss_subgraph) # labels.squeeze()?
        
        # call the back propagation
        loss.backward()
        # any clipin ggradient optimization?
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)  #
#         Clips gradient norm of an iterable of parameters.
#         The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
        
        self.optimizer.step()
        # ending of the training process
        
        # also return the total training time and uploading time in seconds
        return loss, self.predict(preds)

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Purpose: evaluation only on the CPU side
        Forward propagation only
        No backpropagation and thus no need for gradients
        """
        self.eval()
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(self.label_full.numpy().argmax(axis=1).astype(np.int64))
                
        label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        
        with torch.no_grad():
            # only call the forward propagation
            preds = self(adj_subgraph, feat_subg)
            loss = self._loss(preds, label_subg_converted, norm_loss_subgraph)
            
        return loss, self.predict(preds), label_subg

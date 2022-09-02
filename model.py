
import torch.nn as nn
import torch
import math
import random

import torch.nn.functional as F

from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from torch.utils.data import DataLoader
import dgl.nn.pytorch as dglnn
class MyGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MyGCN, self).__init__()
        self.out1 = dglnn.GraphConv(nfeat, nhid)
        self.out2 = dglnn.GraphConv(nhid, nhid)
        self.l1 = nn.Linear(nhid, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 2)
    def forward(self, x1, x2, fea1, fea2):
        fea1 = F.relu(self.out1(x1, fea1))
        fea2 = F.relu(self.out1(x2, fea2))
        fea1 = F.relu(self.out2(x1, fea1))
        fea2 = F.relu(self.out2(x2, fea2))
        x1.ndata['fea'] = fea1
        x2.ndata['fea'] = fea2
        hg1 = dgl.mean_nodes(x1, 'fea')
        hg2 = dgl.mean_nodes(x2, 'fea')
        hg = torch.mul(hg1, hg2)
        l1 = self.l1(hg)
        l2 = self.l2(l1)
        l3 = F.softmax(self.l3(l2))
        return l3


from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

import data_utils
import dgl.nn.pytorch as dglnn
import dgl
import random
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, getopt

opts, args = getopt.getopt(sys.argv[1:],"he:l:o:",["epochs=","lr=","outfile="])
for opt, arg in opts:
    if opt == '-h':
        print('train_model.py -e <epochs> -l <learning_rate>')
        sys.exit()
    elif opt in ("-e", "--epochs"):
        EPOCHS = arg
    elif opt in ("-l", "--lr"):
        LR = arg
    elif opt in ("-o", "--outfile"):
        OUTFILE = arg


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
        l3 = self.l3(l2)
        return l1, l3

use_gpu = torch.cuda.is_available()
use_gpu
torch.cuda.set_device(0)

with open("allprotein_adj_C3_all_10A_patch_surface.pkl",'rb') as infile:
    dict_adj = pkl.load(infile)
with open("allprotein_fea_dssp_C3_all_patch_surface.pkl",'rb') as infile:
    dict_fea = pkl.load(infile)

train_ppi = []
train_label = []
test_ppi = []
test_label = []
with open("HuRI example dataset/0.train.pos") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj.keys() and linea[1] in dict_adj.keys():
            train_ppi.append((linea[0],linea[1]))
            train_label.append(1)
with open("HuRI example dataset/0.train.neg") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj.keys() and linea[1] in dict_adj.keys():
            train_ppi.append((linea[0],linea[1]))
            train_label.append(0)
with open("HuRI example dataset/0.test.pos") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj.keys() and linea[1] in dict_adj.keys():
            test_ppi.append((linea[0],linea[1]))
            test_label.append(1)
with open("HuRI example dataset/0.test.neg") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj.keys() and linea[1] in dict_adj.keys():
            test_ppi.append((linea[0],linea[1]))
            test_label.append(0)

train_samples = []
test_samples = []
for i in range(len(train_ppi)):
    try:
        g1,g2 = getData_GCN(train_ppi[i][0],train_ppi[i][1],dict_adj,dict_fea)
        train_samples.append((g1,g2,train_label[i]))
    except:
        pass
for i in range(len(test_ppi)):
    try:
        g1,g2 = getData_GCN(test_ppi[i][0],test_ppi[i][1],dict_adj_y,dict_fea_y)
        test_samples.append((g1,g2,test_label[i]))
    except:
        pass



seed = 867482
setup_seed(seed)
pos_index = []
neg_index = []
for i in range(len(train_samples)):
    if train_samples[i][2]==1:
        pos_index.append(i)
    else:
        neg_index.append(i)
pos_index_add = random.choices(pos_index,k=len(neg_index)-len(pos_index))
train_samples_new = []
for i in pos_index_add:
    train_samples_new.append(train_samples[i])
for i in neg_index:
    train_samples_new.append(train_samples[i])
for i in pos_index:
    train_samples_new.append(train_samples[i])
random.shuffle(train_samples_new)

epochs = EPOCHS #20
lr = LR #0.0005
hidden = 512
dropout = 0.2
nfeat = 33
model = MyGCN(nfeat,hidden,dropout)
model = model.cuda()
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()
optimizer = optim.Adam(model.parameters(),lr=lr)
data_loader = DataLoader(train_samples_new, batch_size=16, shuffle=False,
                     collate_fn=collate_GCN)
model.train()
for epoch in range(epochs):
    for iter,(batch_g1,batch_g2,batch_label) in enumerate(data_loader):
        fea1 = batch_g1.ndata['fea']
        fea2 = batch_g2.ndata['fea']
        batch_g1,batch_g2,fea1,fea2 = batch_g1.to('cuda:3'),batch_g2.to('cuda:3'),fea1.cuda(),fea2.cuda()
        batch_label = batch_label.cuda()
        trainfc,prediction = model(batch_g1,batch_g2,fea1,fea2)
        tt = prediction.detach().cpu().numpy()
        loss = loss_func(prediction, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loader = DataLoader(test_samples, batch_size=512, shuffle=False,
                         collate_fn=collate_GCN)
    model.eval()
    test_ped, t_label, test_pred= [], [],[]
    with torch.no_grad():
        for it, (batch_g1,batch_g2,batch_label) in enumerate(test_loader):
            fea1 = batch_g1.ndata['fea']
            fea2 = batch_g2.ndata['fea']
            batch_g1,batch_g2,fea1,fea2 = batch_g1.to('cuda:3'),batch_g2.to('cuda:3'),fea1.cuda(),fea2.cuda()
            batch_label = batch_label.cuda()
            GCN_tensor,pred = model(batch_g1,batch_g2,fea1,fea2)
            pred = torch.softmax(pred, 1)
            tt = pred.detach().cpu().numpy()
            test_pred += list(tt[:,1])
            t_label += batch_label.cpu().numpy().tolist()
    precision,recall,thresholds = precision_recall_curve(t_label,test_pred)
    auprc = metrics.auc(recall,precision)
    print("auprc:",auprc)
torch.save(model,OUTFILE+str(seed)+".pt")

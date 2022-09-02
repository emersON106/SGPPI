import torch
import random

best_train_fc_GCN = []
best_test_fc_GCN = []
torch.cuda.set_device(1)
max_aupr = 0
best_pre_GCN_train = []
best_pre_GCN_test = []
best_thr = []
best_precision = []
best_recall = []


epochs = 50
lr = 0.001
weight_decay = 5e-4
hidden = 512
dropout = 0.2
nfeat = 33
model = MyGCN(nfeat, hidden, dropout)
model = model.cuda()
loss_func = nn.BCELoss()
loss_func = loss_func.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

data_loader = DataLoader(train_samples, batch_size=16, shuffle=False,
                         collate_fn=collate_GCN)
model.train()
for epoch in range(epochs):
    for iter, (batch_g1, batch_g2, batch_label) in enumerate(data_loader):
        fea1 = batch_g1.ndata['fea']
        fea2 = batch_g2.ndata['fea']
        batch_g1, batch_g2, fea1, fea2 = batch_g1.to('cuda:1'), batch_g2.to('cuda:1'), fea1.cuda(), fea2.cuda()
        batch_label = batch_label.cuda()
        loss_label = torch.stack((torch.abs(batch_label.cuda() - 1), batch_label.cuda()), dim=1).float()

        prediction = model(batch_g1, batch_g2, fea1, fea2)
        loss = loss_func(prediction, loss_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

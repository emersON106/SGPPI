import numpy as np
import torch
from sklearn import preprocessing
import scipy.sparse as sp
import dgl
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def getData_GCN(protein1, protein2, dict_adj, dict_fea):
    idx_features1 = dict_fea[protein1].astype(np.float32)
    idx_features2 = dict_fea[protein2].astype(np.float32)
    feature1 = sp.csr_matrix(np.concatenate(
        (idx_features1[:, 1:2], preprocessing.scale(idx_features1[:, 2:6]), preprocessing.scale(idx_features1[:, 6:])),
        axis=1), dtype=np.float32)
    feature2 = sp.csr_matrix(np.concatenate(
        (idx_features2[:, 1:2], preprocessing.scale(idx_features2[:, 2:6]), preprocessing.scale(idx_features2[:, 6:])),
        axis=1), dtype=np.float32)
    idx1 = np.array(idx_features1[:, 0], dtype=np.int32)
    idx2 = np.array(idx_features2[:, 0], dtype=np.int32)
    idx_map1 = {j: i for i, j in enumerate(idx1)}
    idx_map2 = {j: i for i, j in enumerate(idx2)}
    edges_unordered1 = dict_adj[protein1]
    edges_unordered1 = edges_unordered1.astype('int32')
    edges_unordered2 = dict_adj[protein2]
    edges_unordered2 = edges_unordered2.astype('int32')
    edges1 = np.array(list(map(idx_map1.get, edges_unordered1.flatten())), dtype=np.int32).reshape(
        edges_unordered1.shape)
    edges2 = np.array(list(map(idx_map2.get, edges_unordered2.flatten())), dtype=np.int32).reshape(
        edges_unordered2.shape)
    adj1 = sp.coo_matrix((np.ones(edges1.shape[0]), (edges1[:, 0], edges1[:, 1])),
                         shape=(idx_features1.shape[0], idx_features1.shape[0]), dtype=np.float32)
    adj2 = sp.coo_matrix((np.ones(edges2.shape[0]), (edges2[:, 0], edges2[:, 1])),
                         shape=(idx_features2.shape[0], idx_features2.shape[0]), dtype=np.float32)
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    g1 = dgl.DGLGraph(adj1)
    g1 = dgl.add_self_loop(g1)
    g2 = dgl.DGLGraph(adj2)
    g2 = dgl.add_self_loop(g2)
    feature1 = torch.FloatTensor(np.array(feature1.todense()))
    feature2 = torch.FloatTensor(np.array(feature2.todense()))
    g1.ndata['fea'] = feature1
    g2.ndata['fea'] = feature2
    return g1, g2
def collate_GCN(samples):
    g1s,g2s,labels= map(list,zip(*samples))
    return dgl.batch(g1s),dgl.batch(g2s),torch.tensor(labels, dtype=torch.long)


train_ppi = []
train_label = []
test_ppi = []
test_label = []
with open("train_pos.txt") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj.keys() and linea[1] in dict_adj.keys():
            train_ppi.append((linea[0],linea[1]))
            train_label.append(1)
with open("train_neg.txt") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj.keys() and linea[1] in dict_adj.keys():
            train_ppi.append((linea[0],linea[1]))
            train_label.append(0)
with open("test_pos.txt") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj_y.keys() and linea[1] in dict_adj_y.keys():
            test_ppi.append((linea[0],linea[1]))
            test_label.append(1)
with open("test_neg.txt") as infile:
    for line in infile:
        if len(line)<10:
            continue
        linea = line.strip().split('\t')
        if linea[0] in dict_adj_y.keys() and linea[1] in dict_adj_y.keys():
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
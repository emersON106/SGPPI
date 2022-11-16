import os
import torch
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
import dgl

def setup_seed(seed=867482):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def returnTrainAndTest(x,y,k,randomstate):
    train = []
    test = []
    skf = StratifiedKFold(k, random_state=randomstate, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(x, y), start=1):
        train.append(list(train_index))
        test.append(list(test_index))
    return train,test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def getData_GCN(protein1, protein2, dict_adj, dict_fea):
    idx_features1 = dict_fea[protein1].astype(np.float32)
    idx_features2 = dict_fea[protein2].astype(np.float32)
    feature1 = sp.csr_matrix(
        np.concatenate((idx_features1[:, 1:2], idx_features1[:, 2:6], idx_features1[:, 6:]), axis=1), dtype=np.float32)
    feature2 = sp.csr_matrix(
        np.concatenate((idx_features2[:, 1:2], idx_features2[:, 2:6], idx_features2[:, 6:]), axis=1), dtype=np.float32)
    # feature1 = sp.csr_matrix(preprocessing.scale(idx_features1[:, 6:26]), dtype=np.float32)
    # feature2 = sp.csr_matrix(preprocessing.scale(idx_features2[:, 6:26]), dtype=np.float32)
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
import os
import numpy as np
import pickle as pkl
rootdir = "SamplePrepare/"
sampleList = [] # This is a List contains all the protein name in your dataset
listdir_adj = os.listdir(rootdir+"adj/")
listdir_features = os.listdir(rootdir+"features/")
dict_adj = {}
dict_fea = {}
for filename in sampleList:
    try:
    #proteiname = filename.split('-')[1]
        tmpadj = np.loadtxt(rootdir +"adj10/AF-"+ filename + "-F1-path.adj", dtype=np.dtype(str))
        tmpfea = np.loadtxt(rootdir +"features/AF-"+ filename + "-F1-patch.features", dtype=np.dtype(str))
        dict_adj[filename] = tmpadj
        dict_fea[filename] = tmpfea
    except:
        pass
with open(rootdir+"sample_adj.pkl",'wb') as outfile:
    pkl.dump(dict_adj,outfile)
with open(rootdir+"sample_fea.pkl",'wb') as outfile:
    pkl.dump(dict_fea,outfile)

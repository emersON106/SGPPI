# SGPPI
SGPPI: structure-aware prediction of protein-protein interactions in rigorous conditions with graph convolutional network
SGPPI, a structure-based deep learning framework for predicting PPIs using graph convolutional networks (GCN). In particular, SGPPI focuses on protein patches on protein-protein binding interfaces and extracts structural, geometric and evolutionary features from the residue contact map to predict PPIs. We demonstrate that our model outperforms traditional machine learning
methods and state-of-the-art deep learning-based methods using a non-representation-bias benchmark data set. Moreover, our model trained on human data can be reliably transferred to predict yeast PPIs, indicating that SGPPI can capture converging
structural features of protein interactions across various species.
# Requirements
torch (==1.5.0)  
scipy (==1.5.2)  
scikit-learn (==0.24.2)  
dgl (0.7.2)  
numpy (==1.19.1)  
# Graph convolutional neural network
In SGPPI, we used GCN to cpature the hidden features of protein structures. The graph used here is the residue contact map with the threshold of 10Å. Features of the node including the pssm profiles, second structure and Jet2 features. To use SGPPI, users should prepare the  adjacency matrix of the graph and the feature list of the residues. 
# Running
## Get Features
SGPPI regard the protein as collection of protein interface patches and integrated the global and local structural features of each residue in these patches. Besides, a comprehensive set of protein sequence and structural features are considered: a) evolutionary information of the residue through position-specific scoring matrices (PSSMs); b) location in the underlying protein secondary structure; c) global and local geometrical descriptors.
To use SGPPI, you should first calculate all the needed features of proteins. We have published the calculated features of both human and yeast proteins, you can find them at https://figshare.com/articles/dataset/PDB_files/20353353. The features mainly include the following files: *.atomAxs, *.axs, *.clusters, *.cv, *.cvlocal and *.pssm. Use feature_extract.py to generate the features of the corresponding protein.

```shell
feature_extract.py –i protein_name –o protein_features
```

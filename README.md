# SGPPI
SGPPI: structure-aware prediction of protein-protein interactions in rigorous conditions with graph convolutional network
SGPPI, a structure-based deep learning framework for predicting PPIs using graph convolutional networks (GCN). In particular, SGPPI focuses on protein patches on protein-protein binding interfaces and extracts structural, geometric and evolutionary features from the residue contact map to predict PPIs. We demonstrate that our model outperforms traditional machine learning
methods and state-of-the-art deep learning-based methods using a non-representation-bias benchmark data set. Moreover, our model trained on human data can be reliably transferred to predict yeast PPIs, indicating that SGPPI can capture converging
structural features of protein interactions across various species.
# Graph convolutional neural network
In SGPPI, we used GCN to cpature the hidden features of protein structures. The graph used here is the residue contact map with the threshold of 10Å. Features of the node including the pssm profiles, second structure and Jet2 features. To use SGPPI, users should prepare the  adjacency matrix of the graph and the feature list of the residues. 
# USAGE
## Requirements
torch (==1.5.0)  
scipy (==1.5.2)  
scikit-learn (==0.24.2)  
dgl (0.7.2)  
numpy (==1.19.1)  
## Get Features
SGPPI regard the protein as collection of protein interface patches and integrated the global and local structural features of each residue in these patches. Besides, a comprehensive set of protein sequence and structural features are considered: a) evolutionary information of the residue through position-specific scoring matrices (PSSMs); b) location in the underlying protein secondary structure; c) global and local geometrical descriptors.
To use SGPPI, you should first calculate all the needed features of proteins. We have published the calculated features of both human and yeast proteins, you can find them at [figshare](https://figshare.com/articles/dataset/PDB_files/20353353). The features mainly include the following files: *.atomAxs, *.axs, *.clusters, *.cv, *.cvlocal and *.pssm. Use feature_extract.py to generate the features of the corresponding protein.

```python
python feature_extract.py –i protein_name –o protein_features
```
## Get Adjacent Matrix
SGPPI consider a contact if the geometrical distance of any two residues’ Cα atoms is less than a certain threshold (default 10 Å), allowing us to represent a protein structure by an undirected graph of the included surface/patch residues. Use adjmatrix_extract.py to generate the adjacent matrix of the corresponding protein structure.

```python
python adjmatrix_extract.py –i pdb_file –o adjacent_matrix
```
## Store the features and adjacency matrix of all sample proteins
Before starting, users should prepare two files: sample_adj.pkl and sample_fea.pkl corresponding to the dictionary of sample adjacency matrix and sample features. Use SaveToDict.py generate these two files. Before running the script, you need to prepare a list of all the samples’ names, and then modify the sampleList in the script. If your sample contains two proteins: P14859 and Q5SXM2, please modify the list in the script to sampleList = ['P14859','Q5SXM2'] and then run the script:

```python
python SaveToDict.py
```
# Acknowledgments
We would like to thank the [DGL](https://github.com/dmlc/dgl) team for the source code of GCN part.

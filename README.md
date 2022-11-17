# SGPPI
SGPPI: structure-aware prediction of protein-protein interactions in rigorous conditions with graph convolutional network
SGPPI, a structure-based deep learning framework for predicting PPIs using graph convolutional networks (GCN). In particular, SGPPI focuses on protein patches on protein-protein binding interfaces and extracts structural, geometric and evolutionary features from the residue contact map to predict PPIs. We demonstrate that our model outperforms traditional machine learning
methods and state-of-the-art deep learning-based methods using a non-representation-bias benchmark data set. Moreover, our model trained on human data can be reliably transferred to predict yeast PPIs, indicating that SGPPI can capture converging
structural features of protein interactions across various species.
# Graph convolutional neural network
In SGPPI, we used GCN to cpature the hidden features of protein structures. The graph used here is the residue contact map with the threshold of 10Å. Features of the node including the pssm profiles, second structure and Jet2 features. To use SGPPI, users should prepare the  adjacency matrix of the graph and the feature list of the residues. 
# USAGE
## Dataset
We provided the positive and negative samples for three baseline datasets: Profkernelppi human dataset, HuRI dataset and filtered Pan’s dataset. You can find all the datasets used in SGPPI from three datasets folders and the form of the dataset is as follows:
| Protein A | Protein B |
| ------ | ------  | 
|O15015|P53582|
|Q9NZC7|Q8IY17|
|Q9UK11|O43795|


Each data set contains two columns, which are the input two proteins.  In the SGPPI model, we set the label of the positive sample to 1 and the label of the negative sample to 0.
## Requirements
torch (==1.5.0)  
scipy (==1.5.2)  
scikit-learn (==0.24.2)  
dgl (0.7.2)  
numpy (==1.19.1)  
## Get Features
SGPPI regard the protein as collection of protein interface patches and integrated the global and local structural features of each residue in these patches. Besides, a comprehensive set of protein sequence and structural features are considered: a) evolutionary information of the residue through position-specific scoring matrices (PSSMs); b) location in the underlying protein secondary structure; c) global and local geometrical descriptors.<br />
To use SGPPI, you should first calculate all the needed features of proteins. We have published the calculated features of both human and yeast proteins, you can find them at [figshare](https://figshare.com/articles/dataset/PDB_files/20353353). The features mainly include the following files: `.atomAxs`, `.axs`, `.clusters`, `.cv` , `.cvlocal` and `.pssm`.<br /> 
| Features | Description |
| ------ | ------  | 
|`.atomAxs`|accessibility at atomic level|
|`.axs`|accessibility at residue level|
|`.clusters`|potential protein interaction interface|
|`.cv`|global circular variances|
|`.cvlocal`|local circular variances|
|`.pssm`|position-specific scoring matrices|


Use feature_extract.py to generate the features of the corresponding protein. **Please note: `dssp.txt` and all the feature files should be in the same file directory as "feature_extract.py".**

```python
python feature_extract.py –i protein_name –o protein_features
```
## Get Adjacent Matrix
SGPPI consider a contact if the geometrical distance of any two residues’ Cα atoms is less than a certain threshold (default 10 Å), allowing us to represent a protein structure by an undirected graph of the included surface/patch residues. Use `adjmatrix_extract.py` to generate the adjacent matrix of the corresponding protein structure. **Please note: `Human_RSA0.2.pkl` or `Yeast_RSA0.2.pkl` should be in the same file directory as "adjmatrix_extract.py". Use the parameter -s to select the species**

```python
python adjmatrix_extract.py –i pdb_file –o adjacent_matrix -s human
```
## Store the features and adjacency matrix of all sample proteins
Before starting, users should prepare two files: `sample_adj.pkl` and `sample_fea.pkl` corresponding to the dictionary of sample adjacency matrix and sample features. Use SaveToDict.py generate these two files. Before running the script, you need to prepare a list of all the samples’ names, and then modify the `sampleList` and `rootdir` in the script. If your sample contains two proteins: P14859 and Q5SXM2, please modify the list in the script to sampleList = ['P14859','Q5SXM2'] and then run the script:

```python
python SaveToDict.py
```
After running, two `.pkl` files (`sample_adj.pkl` and `sample_fea.pkl`) will be obtained, which will be used for the training and prediction of the model in the training step.
## Train the model
After all the above documents are ready, the training of the model can be started.
Use `train_model.py` to train the SGPPI model. 
```python
python train_model.py -e 20 -l 0.0005 -o model0
```
In the source code, we provide the seed for users to repeat our results. The default dataset is the first  cross validation data of HuRI dataset. If you want to modify the data set, please change to the target dataset path at the corresponding position of the source code. <br /><br />
At the same time, we also uploaded all the models used in the paper, which you can find in three `models` folders or at [figshare](https://figshare.com/articles/software/SGPPI_models_rar/21564036). To use these models directly, please use pytorch load as follows:

```python
torch.load("model.pt")
```

# Acknowledgments
We would like to thank the [DGL](https://github.com/dmlc/dgl) team for the source code of GCN part.

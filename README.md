# SGPPI
SGPPI: structure-aware prediction of protein-protein interactions in rigorous conditions with graph convolutional network
SGPPI, a structure-based deep learning framework for predicting PPIs using graph convolutional networks (GCN). In particular, SGPPI focuses on protein patches on protein-protein binding interfaces and extracts structural, geometric and evolutionary features from the residue contact map to predict PPIs. We demonstrate that our model outperforms traditional machine learning
methods and state-of-the-art deep learning-based methods using a non-representation-bias benchmark data set. Moreover, our model trained on human data can be reliably transferred to predict yeast PPIs, indicating that SGPPI can capture converging
structural features of protein interactions across various species.
## Requirements
torch (==1.5.0)
scipy (==1.5.2)
scikit-learn (==0.24.2)
dgl (0.7.2)
numpy (==1.19.1)
## Dataset
The dataset and PDB files used in our research are available at 10.6084/m9.figshare.20353353

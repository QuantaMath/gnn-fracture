# gnn-fracture
Prediction and control of fracture paths in disordered architected materials using graph neural networks

## Pipeline
- To generate the Voronoi networks and their corresponding finite element meshes, run **generation/generateVoronoiLattices.py**. Adding the flag **--plot=1** plots each Voronoi network as it is generated. Their mechanical response under mode-I loading is then evaluated using the open-source finite element code ae108 [link](https://www.ae108.ethz.ch). The training dataset can be found in [link](https://www.research-collection.ethz.ch/handle/...).
- To train the GNN, run **learning/train_gnn.py**
- To optimize the fracture length, run **optimization/optimize.py**. This generates optimal Voronoi networks and stores them.


## Requirements

- Python (tested on version 3.9.1)
- Python packages:
  - Pandas
  - Numpy
  - Scipy
  - NetworkX
  - Pytorch 
  - Matplotlib


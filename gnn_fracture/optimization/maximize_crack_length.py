import sys
import time
import torch
import random
import argparse
import numpy as np
from .voronoi import Voronoi
from joblib import Parallel, delayed

sys.path.append('../generation')
from ..generation.voronoiUtilities import *
sys.path.append('../learning')
from ..learning.model import *
from ..learning.utilities_gru import getNotchEdges
from ..learning.utilities_gru import updateQuantitiesForNextStepPrediction

def run(
    Nx : int = 13,
    Ny : int = 13,
    shape : str = 'hex',
    notch_width : int = 4,
    num_nodes : int = 162,
    normalize_input : int = 1,
    encoder : str = 'GraphConv',
    decoder : str = 'InnerProduct',
    coor_dim : int = 2 ,
    edge_dim : int = 3 ,
    node_dim : int = 9,
    hidden_dim : int = 256,
    gnn_layers : int = 6,
    latent_dim : int = 32,
    dropout : float = 0.0,
    device : str = 'cpu' ,
    model_name : str = 'gru',
    seed : int = 1,
    dt : float = 4.0,
    lamdaZ0 : float = 40.,
    lamdaGeom0 : float = 100.,
    opt_iter0 : int = 0,
    opt_iters : int = 10,
    n_cores : int = 40,
    n_samples : int = 2000,
    opt_name : str = 'glob-0',
):
 

    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # The trained model we use for evaluation
    loadName = 'output/' + model_name

    # Read data normalizers 
    if normalize_input:
        rootDir = 'data/'
        dataDirGDual = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-G-dual/'
        x_bounds = np.loadtxt(dataDirGDual + 'x_normalization.dat') 
        r_bounds = np.loadtxt(dataDirGDual + 'r_normalization.dat')

    # Where to save the optimized architecture
    saveName = 'output/' + opt_name

    # Batch vector which shows which holds the batch id for each node
    batch = np.zeros(num_nodes)
    batch = torch.tensor(batch, device=device, dtype=torch.int64)
        
    # Define encoder
    if encoder == 'GraphConvGRU':
        encoder = GraphConvGRU(node_dim, hidden_dim, latent_dim, edge_dim, gnn_layers, dropout, batch)
    else:
        sys.exit('Unknown encoder type')

    # Define decoder
    if decoder == 'InnerProduct':
        decoder = InnerProductDecoder(batch_size=1)
    else:
        sys.exit('Unknown decoder type')

    # Define autoencoder 
    graph_model = GAEGRU(encoder, decoder)

    # Load trained model
    graph_model.load_state_dict(torch.load(loadName + '-model', map_location=torch.device(device)))

    # Move to device
    graph_model.to(device)
    graph_model.eval()

    # The broken edges forming the initial notch
    broken_edges_init = getNotchEdges(Nx, Ny, shape, notch_width)

    # Initialize ensemble parameters
    lamda = np.array([lamdaZ0,lamdaGeom0])
    lamda_all = np.copy(lamda)
    hexSpacing = 2
    hexZ = 6

    for iter in range(opt_iter0, opt_iter0 + opt_iters):

        # Define the sampling function evaluated in parallel
        def sample(idx):

            start = time.time()

            # Equilibrate the Voronoi graph
            voronoi = Voronoi(Nx, Ny, shape, notch_width, lamda, hexSpacing, hexZ, seed*n_samples + idx)
            voronoi.equilibrate()
            voronoi.computeVoronoiTesselation()
            # Compute input quantities for predictive model 
            rightBdNodes = getRightBoundaryNucleiIndices(Nx,Ny)
            A0, x0, r, M = voronoi.getDualGraphAndFeatures(notch_width)

            # Normalize
            if normalize_input:
                # Unpack        
                x_min,x_max = x_bounds
                r_x_min,r_x_max = r_bounds[0]
                r_y_min,r_y_max = r_bounds[1]
                r_l_min,r_l_max = r_bounds[2]
                # Normalize
                x0 = (x0-x_min)/x_max
                r[:,0] = (r[:,0]-r_x_min)/r_x_max
                r[:,1] = (r[:,1]-r_y_min)/r_y_max
                r[:,2] = (r[:,2]-r_l_min)/r_l_max

            # Move to torch tensors
            A0 = torch.tensor(A0, dtype=torch.float32)        
            x0 = torch.tensor(x0, dtype=torch.float32)
            M = torch.tensor(M, dtype=torch.int16)
            r = torch.tensor(r, dtype=torch.float32)
            A = A0.clone().detach()
            x = x0.clone().detach()

            # Initialize list of broken edges
            edges_pred = broken_edges_init.copy()

            # Incrementally predict crack advance until boundary is reached
            reachedBoundary = False
            step = 0
            while not reachedBoundary:

                # No hidden state for first step prediction
                if step == 0: 
                    h = None

                # Predict the edge to break
                edge_index = torch.nonzero(A).t().contiguous()
                if use_gru:
                    h, F_pred = graph_model(x, edge_index, r, M, h)
                else:
                    F_pred = graph_model(x, edge_index, r, M)
                edge_idx_pred = torch.argmax(F_pred.view(1,-1),axis=1)
                edge_idx_pred = int(edge_idx_pred.detach().numpy())
                edge_pred = list(divmod(edge_idx_pred,num_nodes))
                edges_pred.append(edge_pred)

                # Update the quantities for next prediction (adjacency matrix, etc)
                A, x, r, M = updateQuantitiesForNextStepPrediction(A0, A, x0, r, 
                    x_bounds, edges_pred, device)

                # Stop if complete failure (i.e. crack reached boundary)
                if edge_pred[0] in rightBdNodes or edge_pred[1] in rightBdNodes:
                    reachedBoundary = True
                step += 1

            # Return the energies and the crack length
            return voronoi.HZ, voronoi.HGeom, len(edges_pred)

        # Sample in parallel  
        start = time.time()
        HC = Parallel(n_jobs=n_cores)(delayed(sample)(s) for s in range(n_samples))
        HC = np.array(HC)      
        print('Sampling time (s):', time.time()-start)

        # Compute covariance
        covHC = np.cov(HC.T)
        print('covHC=')
        print(covHC)

        # Timestep
        # Note: in case of extremely topological disorder, the full covariance matrix is not invertible. 
        # In that case, only need to update geometrical lamda, and add small Gaussian noise to avoid num. problems
        if np.std(HC[:,0]) < 3e-2:
            covHInv = 1./covHC[1,1]
            dlamdaGeom = -dt*covHInv*covHC[-1,1]
            dlamda = np.array([0.,dlamdaGeom])
            HC[:,0] = HC[:,0] + np.random.normal(0,4e-2,n_samples)
            print('Warning - Very small HZ')
        else:
            # Check invertibility
            if np.linalg.cond(covHC[:-1,:-1]) < 1./sys.float_info.epsilon:
                covHInv = np.linalg.inv(covHC[:-1,:-1])
                dlamda = -dt*covHInv.dot(covHC[-1,:-1])
            else:
                break

        lamda += dlamda
        lamda_all = np.vstack((lamda_all, lamda))
        print('<HZ> =', np.mean(HC[:,0]), ' -- <HGeom> =', np.mean(HC[:,1]))

        # Save evolution of H, crack length
        np.savetxt(saveName + '-HC-step' + str(iter) + '.dat', HC, fmt='%.4f %.4f %lu')

        # Save evolution of lamda
        filenameLamda = saveName + '-lamda-evolution.dat'
        if iter == 0:
            f = open(filenameLamda, "w")
            f.write('lamdaZ, ') 
            f.write('lamdaGeom\n') 
            f.write(str(round(lamda[0],4)) + ', ')
            f.write(str(round(lamda[1],4)) + '\n')
        else:
            f = open(filenameLamda, "a")
            f.write(str(round(lamda[0],4)) + ', ')
            f.write(str(round(lamda[1],4)) + '\n')
        f.close() 

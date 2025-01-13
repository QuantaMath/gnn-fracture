import argparse
import numpy as np
import pandas as pd
from .voronoi import Voronoi

def run(
		Nx: int = 13, # Nx x Ny cells
		Ny: int = 13, # Nx x Ny cells
		shape: str = 'hex', # 'hex'
		notchWidth: int = 4, # number of missing cells in notch
		meshIdx0:int = 0, # starting mesh index (when splitting the computation in batches)
		nMeshes: int = 10000, # how many meshes to compute (when splitting the computation in batches)
		dataDir: str = "data/",
		plot : int = 0):

	# Read the lambda parameters controlling the disorder for the sampled Voronoi lattices
	dataDir = dataDir + shape +'-' + str(Nx) +'x' + str(Ny) + '-G/'
	fname = dataDir + 'lamda-sampling.dat'
	lamdaPairs = pd.read_csv(fname, skipinitialspace=True,skiprows=0)
	lamdaPairs = lamdaPairs.iloc[:].to_numpy()
	nRepeat = int(np.round(nMeshes/len(lamdaPairs)))+1
	lamdaPairs = np.concatenate([lamdaPairs]*nRepeat,axis=0)
	lamdaPairs = lamdaPairs[:nMeshes]

	meshIdx = meshIdx0-1
	for lamda in lamdaPairs:
		meshIdx += 1
		print('meshIdx = ',meshIdx)
		voronoi = Voronoi(Nx, Ny, shape, notchWidth, lamda, meshIdx)
		voronoi.sampleNuclei()
		voronoi.saveGraph(meshIdx)
		voronoi.saveDualGraph(meshIdx)
		if plot:
			voronoi.plotLattice()
			voronoi.plotDualLattice()

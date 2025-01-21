import sys
import time
import torch
import random
import numpy as np
from .model import *
from .utilities_gru import *
from torch_geometric.data import Data, DataLoader
from tqdm.notebook import tqdm

def run(
    Nx : int = 13,
    Ny : int = 13,
    shape : str = 'hex',
    Ndata : int = 60000,
    num_nodes : int = 162,
    notch_width : int = 4,
    epochs : int = 20,
    batch_size : int = 20,
    data_augment : int = 1,
    normalize : int = 1, 
    test_size : float = 0.1,
    encoder : str = 'GraphConvGRU',
    decoder : str = 'InnerProduct',
    coor_dim : int = 2, 
    edge_dim : int = 3, 
    node_dim : int = 9,
    hidden_dim : int = 256,
    gnn_layers : int = 6,
    latent_dim : int = 32,
    min_disorder : float = 0.1,
    max_disorder : float = 1.2,
    dropout : float = 0.4,
    learning_rate : float = 5e-5,
    patience : int = 6, 
    weight_decay : float = 1e-5,
    device : str = 'cuda',
    scheduler : str = 'StepLR', 
    optimizer : str = 'Adam',
    model_name :str = 'gru',
):
    # Split to train/test data
    random.seed(0)

    np.random.seed(0)
    torch.manual_seed(0)

    # Split to train, test data
    train_mesh_idx, test_mesh_idx = splitTrainTestMeshes(Nx, Ny, shape, Ndata, min_disorder,
        max_disorder, test_size)

    # Find the maximum fracture sequence length 
    max_sequence_length = getMaxSequenceLength(Nx, Ny, shape, train_mesh_idx)

    # Output directory
    saveName = 'output/' + model_name
    np.savetxt(saveName + '-train_mesh_idx.dat', train_mesh_idx, fmt='%d')
    np.savetxt(saveName + '-test_mesh_idx.dat', test_mesh_idx, fmt='%d')

    # Get train and test data loaders
    train_loaders,_ = getDataLoaders(Nx, Ny, shape, notch_width, train_mesh_idx, normalize,
        data_augment, max_sequence_length, batch_size, shuffle=True)
    test_loaders,_ = getDataLoaders(Nx, Ny, shape, notch_width, test_mesh_idx, normalize,
        0, max_sequence_length, batch_size, shuffle=True)

    data_sets_train = len(train_loaders) * len(train_loaders[0].dataset)
    data_sets_test = max(1,len(test_loaders) * len(test_loaders[0].dataset))

    # Batch vector which shows which holds the batch id for each node
    batch = np.zeros(num_nodes*batch_size)
    for b in range(batch_size):
        batch[b*num_nodes:(b+1)*num_nodes] = b
    batch = torch.tensor(batch, device=device, dtype=torch.int64)
    num_batches = len(train_loaders[0].dataset)/batch_size
    
    # Define encoder
    if encoder == 'GraphConvGRU':
        encoder = GraphConvGRU(node_dim, hidden_dim, latent_dim, edge_dim, gnn_layers, dropout, batch)
    else:
        sys.exit('Unknown encoder type')

    # Define decoder
    if decoder == 'InnerProduct':
        decoder = InnerProductDecoder(batch_size)
    else:
        sys.exit('Unknown decoder type')

    # Define autoencoder 
    graph_model = GAEGRU(encoder, decoder)

    # Move to device
    graph_model.to(device)

    # Define optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(graph_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    torch.autograd.set_detect_anomaly(True)

    # Define scheduler
    if scheduler == 'ConstantLR':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs) 
    elif scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) 
    else:
        sys.exit('Unknown scheduler option')

    loss_curve = []
    test_loss_curve = []
    startTotal = time.time()

    # Early stopping
    last_loss = 1e6
    trigger_times = 0

    # Define testing
    def test(graph_model, test_loaders):

        # Switch to evaluation mode
        graph_model.eval()
        total_loss = 0.
        data_sets_test = 0

        with torch.no_grad():
            # For each batch of meshes
            for data_all_steps in zip(*test_loaders):

                # Forward pass for each timestep
                for step in range(max_sequence_length):

                    # No hidden state for first step prediction
                    if step == 0: 
                        h = None

                    data = data_all_steps[step].to(device)
                    mask = torch.block_diag(*torch.split(data.M,num_nodes)) 
                    data_F = torch.zeros_like(mask, dtype=torch.float32)
                    for b in range(batch_size):
                        data_F[data.F_idx[2*b]+b*num_nodes,
                               data.F_idx[2*b+1]+b*num_nodes] = 1
                        data_F[data.F_idx[2*b+1]+b*num_nodes,
                               data.F_idx[2*b]+b*num_nodes] = 1
                    h, F_pred = graph_model(data.x, data.edge_index, 
                        data.edge_attr, mask, h)
                    pad_mask = torch.ones_like(mask, dtype=torch.float32)
                    for b in range(batch_size):
                        if data.pad[b]: 
                            pad_mask[b*num_nodes:(b+1)*num_nodes,
                                     b*num_nodes:(b+1)*num_nodes] = 0
                    loss = BCELoss(2*F_pred, data_F, pad_mask)
                    total_loss += loss.item()

                    for b in range(batch_size):
                        if data.pad[b] == 0:
                            data_sets_test += 1

        data_sets_test = max(data_sets_test, 1)
        return total_loss/data_sets_test

    # Main training loop
    for epoch in tqdm(range(epochs)):

        start = time.time()

        # Switch to training mode
        graph_model.train()

        # Initialize accumulated loss
        total_loss = 0.
        data_sets_train = 0

        # For each batch of meshes
        batch_idx = 0
        for data_all_steps in tqdm(zip(*train_loaders)):

            # loss = 0
            batch_idx += 1

            # Forward pass for each timestep
            for step in range(max_sequence_length):

                # No hidden state for first step prediction
                if step == 0: 
                    h = None

                data = data_all_steps[step].to(device)
                mask = torch.block_diag(*torch.split(data.M,num_nodes))  
                data_F = torch.zeros_like(mask, dtype=torch.float32)
                for b in range(batch_size):
                    data_F[data.F_idx[2*b]+b*num_nodes,
                           data.F_idx[2*b+1]+b*num_nodes] = 1
                    data_F[data.F_idx[2*b+1]+b*num_nodes,
                           data.F_idx[2*b]+b*num_nodes] = 1
                if step == 0:
                    h, F_pred = graph_model(data.x, data.edge_index, 
                        data.edge_attr, mask, h)
                else:
                    h, F_pred = graph_model(data.x, data.edge_index, 
                        data.edge_attr, mask, h.detach())
                pad_mask = torch.ones_like(mask, dtype=torch.float32)
                for b in range(batch_size):
                    if data.pad[b]: 
                        pad_mask[b*num_nodes:(b+1)*num_nodes,
                                 b*num_nodes:(b+1)*num_nodes] = 0
                loss = BCELoss(2*F_pred, data_F, pad_mask)
                
                for b in range(batch_size):
                    if data.pad[b] == 0:
                        data_sets_train += 1

                # Compute gradient after accumulating loss from the whole sequence 
                loss.backward()
                        
                # Update the weights using the gradient
                optimizer.step()
                optimizer.zero_grad()

                # Keep accumulated loss during epoch
                total_loss += loss.item()  

        total_loss /= data_sets_train

        # Test loss
        test_loss = test(graph_model, test_loaders)
        loss_curve.append(total_loss)
        test_loss_curve.append(test_loss)

        # Early stopping
        if test_loss > last_loss:
            trigger_times += 1
            if trigger_times > patience:
                print('Trigger Times:', trigger_times)
                print('Overfitting.. Stopping the training')
                break
        else:
            trigger_times = 0

        # Stop if things blow up
        if total_loss > 1000:
            print('Blow up - Total loss = ', total_loss)
            break
            
        last_loss = test_loss

        # Update scheduler
        scheduler.step()

        # Print statistics
        print("Epoch:", '%04d' % (epoch + 1) \
          ,", loss = ", "{:.8f}".format(total_loss)\
          ,", test loss = ", "{:.2f}".format(test_loss)\
         )
        sys.stdout.flush()

        # Write to file
        str_tmp = 'Epoch: ' + str(epoch + 1) \
          + ', loss = ' + str(round(total_loss, 4)) \
          + ', test loss = ' + str(round(test_loss , 4)) \
          + ', time = ' + str(time.time() - start) \
          + '\n'

        with open(saveName + '-loss.txt', 'a') as the_file:
            the_file.write(str_tmp)
            
    # Save the trained model
    torch.save(graph_model.state_dict(),saveName + '-model')
    torch.save(optimizer.state_dict(),saveName + '-optimizer')

    # Total time
    print('Total time (s):', time.time()-startTotal)
    the_file.close()
    graph_model.eval()

    # Save loss
    np.savetxt(saveName + '-loss.csv', np.c_[loss_curve, test_loss_curve], 
        delimiter=",", header="Train loss, Test loss")

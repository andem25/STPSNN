import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from net_definition import Net

#   _______        _       _             
#  |__   __|      (_)     (_)            
#     | |_ __ __ _ _ _ __  _ _ __   __ _ 
#     | | '__/ _` | | '_ \| | '_ \ / _` |
#     | | | | (_| | | | | | | | | | (_| |
#     |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
#                                   __/ |
#                                  |___/ 
def training_routine(net, train_dataloader, valid_dataloader, num_epochs, trained_folder, device, loss, optimizer, lr_decay, patience, epoch_start=0, f=None, lr_steps=6):
    """
    Train the spiking neural network with validation and early stopping.
    
    Args:
        net: The neural network model to train
        train_dataloader: DataLoader for training data
        valid_dataloader: DataLoader for validation data
        num_epochs: Total number of training epochs
        trained_folder: Directory path to save trained models
        device: Device to run training on (CPU/GPU)
        loss: Loss function
        optimizer: Optimizer for weight updates
        lr_decay: Learning rate decay factor
        patience: Number of epochs to wait before reducing learning rate
        epoch_start: Starting epoch number (for resuming training)
        f: File object for logging training progress
        lr_steps: Maximum number of learning rate reduction steps
    
    Returns:
        tuple: (loss_hist, valid_loss_hist, acc_hist, valid_acc, best_loss) - 
               Training and validation metrics history
    """
    valid_loss_hist = []
    loss_hist = []
    acc_hist = []
    valid_acc_hist = []
    best_loss = None
    
    # Outer training loop
    for epoch in range(epoch_start,num_epochs):
        #iter_counter = 0
        train_loss = 0
        train_samples = 0
        train_acc = 0
        valid_loss = 0
        valid_samples = 0
        valid_acc = 0
        for i, (data, targets) in enumerate(train_dataloader):
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec, mem_rec = net(data)
            loss_val = loss(spk_rec, targets)
            train_loss += loss_val*targets.numel()
            train_samples += targets.numel()
            _, idx = spk_rec.sum(dim=0).max(1)
            train_acc += np.sum((targets == idx).detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            f.write(f'\nEpoch: {epoch}, Step: {i}/{len(train_dataloader)} Train acc: {train_acc/train_samples}, Train loss: {train_loss/train_samples}')
            print(f'\rEpoch: {epoch}, Step: {i}/{len(train_dataloader)} Train acc: {train_acc/train_samples}, Train loss: {train_loss/train_samples}',end='')
        print("\n")

        train_loss = train_loss/train_samples
        train_acc = train_acc/train_samples
        loss_hist.append(train_loss.item())
        acc_hist.append(train_acc)
        torch.save(net.state_dict(), trained_folder + '/last_network.pt') # .state_dict() # save current model

        # Test set
        with torch.no_grad():
            net.eval()
            for i,(valid_data, valid_targets) in  enumerate(valid_dataloader):
                valid_data = valid_data.to(device)
                valid_targets = valid_targets.to(device)

                valid_spk, valid_mem = net(valid_data)

                batch_loss = loss(valid_spk, valid_targets)
                valid_loss += batch_loss*valid_targets.numel()
                valid_samples += valid_targets.numel()
                _, idx = valid_spk.sum(dim=0).max(1)
                valid_acc += np.sum((valid_targets == idx).detach().cpu().numpy())
                f.write(f'\nEpoch: {epoch}, Step: {i}/{len(valid_dataloader)} Valid acc: {valid_acc/valid_samples}, Valid loss: {valid_loss/valid_samples}')
                print(f'\rEpoch: {epoch}, Step: {i}/{len(valid_dataloader)} Valid acc: {valid_acc/valid_samples}, Valid loss: {valid_loss/valid_samples}',end='')

            # Store loss history for future plotting
            valid_loss = valid_loss/valid_samples
            valid_acc = valid_acc/valid_samples
            valid_loss_hist.append(valid_loss.item())
            valid_acc_hist.append(valid_acc)

            if best_loss is None:
                  best_loss = valid_loss
                  last_save = epoch
                  torch.save(net.state_dict(), trained_folder + f'/network.pt') # .state_dict() # {epoch:2d} #save best model
                  print("\nUpdated best model, saving in:",trained_folder + f'/network.pt\n')
            if valid_loss < best_loss:
                  best_loss = valid_loss
                  torch.save(net.state_dict(), trained_folder + f'/network.pt') # .state_dict() # {epoch:2d} #save best model
                  print("\nUpdated best model, saving in:",trained_folder + f'/network.pt\n')
                  last_save = epoch


            f.write(f'\rEpoch: {epoch}, Step: {i}/{len(valid_dataloader)} Valid acc: {valid_acc/valid_samples}, Valid loss: {valid_loss/valid_samples}')
            print(f'\nSummary Epoch: {epoch}, Train acc: {train_acc}, Train loss: {train_loss}, Valid acc: {valid_acc}, Valid loss: {valid_loss}\n', end='')

            if last_save + patience == epoch:
               if lr_steps>0:
                 optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * lr_decay
                 lr_steps = lr_steps-1
                 f.write(f'\n Updated lr {optimizer.param_groups[0]["lr"]} \n')
               else:
                 print("Early exit")
                 break
      
    return loss_hist, valid_loss_hist, acc_hist, valid_acc, best_loss
import torch

#   _______        _   
#  |__   __|      | |  
#     | | ___  ___| |_ 
#     | |/ _ \/ __| __|
#     | |  __/\__ \ |_ 
#     |_|\___||___/\__|

def test(net, test_dataloader, num_steps, device, what_to_print=[]):
    """
    Test the trained spiking neural network on test data.
    
    Args:
        net: The neural network model to test
        test_dataloader: DataLoader containing test data
        num_steps: Number of time steps for simulation
        device: Device to run inference on (CPU/GPU)
        what_to_print: Optional list of items to print during testing
    
    Returns:
        tuple: (test_spikes, layer1_spikes) - Output spikes and first layer spikes
    """
    with torch.no_grad():
      net.eval()
      for data, targets in test_dataloader:
        data = data.to(device)
        targets = targets.to(device)
        test_spk, mem_rec, spk_1_rec = net(data, test = True)
    return test_spk, spk_1_rec
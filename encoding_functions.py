# from imports import *
from torch.utils.data import Dataset
import numpy as np
import torch



def level_define_16(val, massimo, minimo):
    """
    Encode a single value into 16 discrete levels using thermometer encoding.
    
    Args:
        val: Value to encode
        massimo: Maximum boundary for encoding
        minimo: Minimum boundary for encoding
    
    Returns:
        list: A 16-element binary list representing the encoded value
    """
    lunghezza = massimo - minimo
    step = lunghezza / 12
    base = minimo - 2*step

    encoded = []
    for i in range(1,17):
        if i == 16:
            if val>(base+(i-1)*step):
                encoded.append(1)
            # print(base+(i-1)*step)
        else:
            if val<(base+i*step):
                encoded.append(1)
                break
            else:
                encoded.append(0)
            # print(base+i*step)
    for k in range(16-len(encoded)):
        encoded.append(0)
    return encoded
  
  

def return_max_min(data, min_boundaries):
  """
  Calculate maximum and minimum values for each channel from the initial data windows.
  
  Args:
      data: Input data array with shape (windows, channels, samples)
      min_boundaries: Minimum time boundary in minutes for calculating max/min
  
  Returns:
      tuple: (massimo, minimo) - Lists of maximum and minimum values for each channel
  """
  print("RETURN MAX MIN")
  print(np.array(data).shape)
  massimo = [None, None, None, None]
  minimo = [None, None, None, None]
  win_for_max_min = int((int(min_boundaries)*60)/8) + 1
  for win in data[:win_for_max_min]:
    for chan in range(data.shape[1]):
    #print(win[chan].shape)
      if massimo[chan] is None:
        massimo[chan] = max(abs(max(win[chan])),abs(min(win[chan])))
      else:
        massimo[chan] = max(massimo[chan],max(abs(max(win[chan])),abs(min(win[chan]))))
      minimo[chan] = -massimo[chan]
      
  print("massimo",massimo)
  print("minimo",minimo)
  return massimo, minimo

def encode(data, massimo, minimo):
    """
    Encode the entire dataset using level_define_16 for each value across all channels.
    
    Args:
        data: Input data array containing windows of EEG signals
        massimo: List of maximum values for each channel
        minimo: List of minimum values for each channel
    
    Returns:
        list: Encoded data with each value converted to 16-level representation
    """
    print("massimo",massimo)
    print("minimo",minimo)
    encoded = []
    for window in data:
        encoded_window = [[],[],[],[]]
        for nw, channel in enumerate(window):
            for val in channel:
                  encoded_window[nw].append(level_define_16(val, massimo[nw], minimo[nw]))
        encoded.append(encoded_window)
    return encoded

      
      
class EEG_Dataset(Dataset):
  """
  Custom PyTorch Dataset class for EEG data.
  
  Args:
      x: Input EEG data
      y: Target labels
  """
  def __init__(self, x, y):
    super(EEG_Dataset, self).__init__()
    self.input = x
    self.target = y

  def __getitem__(self, idx):
    """
    Get a single sample from the dataset.
    
    Args:
        idx: Index of the sample
    
    Returns:
        tuple: (input_tensor, target_tensor) for the specified index
    """
    y_out = torch.tensor(self.target[idx]).long() #.astype(np.float32)
    return (
      torch.tensor(self.input[idx].astype(np.float32)), #.astype(np.float32),
      y_out
    )

  def __len__(self):
    """
    Get the total number of samples in the dataset.
    
    Returns:
        int: Number of samples
    """
    return len(self.target) # just one sample for this problem
  
  
def return_start_end(y_test):
  """
  Identify the start and end indices of seizure events in the test labels.
  
  Args:
      y_test: Array of binary labels (1 for seizure, 0 for non-seizure)
  
  Returns:
      tuple: (start_seizure, end_seizure) - Lists of start and end indices of seizures
  """
  test_frame_shift_sec = 8
  y_pred = np.empty(y_test.shape)
  hopsize = test_frame_shift_sec

  start_seizure=[]
  start_seiz = 0
  end_seizure = []
  for i in range(y_test.shape[0]):
    if y_test[i]==1:
       if start_seiz==0:
         start_seizure.append(i)
       start_seiz=1
    if start_seiz == 1 and y_test[i]==0:
       end_seizure.append(i-1)
       start_seiz = 0

  print("START:",start_seizure)
  print("END:",end_seizure)
  
  return start_seizure, end_seizure
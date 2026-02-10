import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
from collections import deque

#  _   _      _                      _      _______                _                    
# | \ | |    | |                    | |    |__   __|              | |                   
# |  \| | ___| |___      _____  _ __| | __    | | ___  _ __   ___ | | ___   __ _ _   _  
# | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /    | |/ _ \| '_ \ / _ \| |/ _ \ / _` | | | | 
# | |\  |  __/ |_ \ V  V / (_) | |  |   <     | | (_) | |_) | (_) | | (_) | (_| | |_| | 
# |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\    |_|\___/| .__/ \___/|_|\___/ \__, |\__, | 
#                                                     | |                   __/ | __/ | 
#                                                     |_|                  |___/ |___/  
class Net(nn.Module):
    """
    Spiking Neural Network with optional Short-Term Plasticity (STP) for seizure detection.
    
    Args:
        num_inputs: Number of input features
        num_outputs: Number of output classes
        num_steps: Number of time steps for simulation
        weight_start: Initial weights for STP (optional)
        STP_enable: Enable Short-Term Plasticity mechanism
        STP_pot: Potentiation parameters for STP
        STP_dep: Depression parameters for STP
        sf: Sampling frequency (default: 1/256)
        beta: Decay rate for leaky integrate-and-fire neurons
        learn_th: Enable learnable threshold
        learn_b: Enable learnable beta
        threshold: Firing threshold for neurons
        fr: Baseline firing rate
        fr_seiz: Seizure firing rate
        deactivation_rate: Initial deactivation rate for STP
    """
    def __init__(self, num_inputs, num_outputs, num_steps, weight_start=None, STP_enable=False, 
                 STP_pot=None, STP_dep=None, sf=1/256, beta=0.9, learn_th=False, 
                 learn_b=False, threshold=1.0, fr=None, fr_seiz=None, deactivation_rate=None):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.STP_ENABLE = STP_enable
        self.fr = fr
        self.deactivation_rate = deactivation_rate
        self.threc = []

        # Layer SNN
        self.fc1 = nn.Linear(num_inputs, 8)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=learn_th, learn_beta=learn_b)
        
        self.fc2 = nn.Linear(8, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=learn_th, learn_beta=learn_b)

        if self.STP_ENABLE:
            from STP_func import STP
            self.STP2 = STP(STP_pot, STP_dep, weight_start, fr, fr_seiz)

    def forward(self, x, test=False):
        """
        Forward pass through the spiking neural network.
        
        Args:
            x: Input tensor (batch, levels, steps, channels)
            test: If True, also return first layer spikes for analysis
        
        Returns:
            tuple: (spike_recordings, membrane_potentials) or 
                   (spike_recordings, membrane_potentials, layer1_spikes) if test=True
        """
        device = x.device
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec, mem_rec = [], []
        
        # Pre-processing: (batch, levels, steps, channels) -> (batch, steps, inputs)
        x = x.transpose(2, 1)
        
        # Variabili di stato (mantenute sul device corretto)
        rate = 0.0
        buff = deque([0.0], maxlen=4096)
        counter = 0
        spk_c = torch.zeros(8, device=device)
        
        init_weight = self.fc2.weight.detach().clone()
        init_deact = self.deactivation_rate.item() if torch.is_tensor(self.deactivation_rate) else self.deactivation_rate
        # Somma pesi positivi iniziali (riga 0)
        suminit = torch.sum(init_weight[0][init_weight[0] > 0]).item()
        spk_1_rec = []

        for step in range(self.num_steps):
            # Input processing
            x_timestep = x[:, step, :].reshape(x.size(0), -1)
            
            # Layer 1
            cur1 = self.fc1(x_timestep)
            spk1, mem1 = self.lif1(cur1, mem1)

            # STP Logic & Weight Update
            rate_val = rate / 4096
            if self.STP_ENABLE and (rate_val < self.deactivation_rate):
                if counter == 4096:
                    with torch.no_grad():
                        # Passiamo spk_c come tensore/numpy a seconda di cosa si aspetta STP2
                        the_w, _ = self.STP2.step_try(self.fc2.weight, spk_c.cpu().numpy(), self.fr)
                        self.fc2.weight.copy_(the_w.float())
            if self.STP_ENABLE:
                # Update Dynamic Deactivation Rate (eseguito ad ogni step)
                with torch.no_grad():
                    w_row0 = self.fc2.weight[0]
                    pos_w_sum = torch.sum(w_row0[w_row0 > 0])
                    self.deactivation_rate = (0.35 + 0.65 * pos_w_sum / suminit) * init_deact
                    self.threc.append(self.deactivation_rate)

            # counter 4096 step
            if counter == 4096:
                counter = 0
                spk_c.zero_()

            counter += 1
            if self.STP_ENABLE:
                spk_c += spk1.detach().squeeze(0)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            # Recording
            spk_rec.append(spk2)
            mem_rec.append(mem2) 
            # Aggiornamento Sliding Window Rate
            current_spk2_sum = spk2.sum().item()
            rate += current_spk2_sum
            if len(buff) == 4096:
                rate -= buff.popleft()
            
            buff.append(current_spk2_sum)
            
            if test:
                spk_1_rec.append(spk1.detach().cpu().numpy())
                
            
        if not test:
            return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
        else:
            return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0), spk_1_rec
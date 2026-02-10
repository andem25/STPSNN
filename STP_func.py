import torch
import numpy as np




def create_empty_list(n):
    """
    Create a list of n empty lists.
    
    Args:
        n: Number of empty lists to create
    
    Returns:
        list: A list containing n empty lists
    """
    return [[] for _ in range(n)]

#    _____ _______ _____  
#   / ____|__   __|  __ \ 
#  | (___    | |  | |__) |
#   \___ \   | |  |  ___/ 
#   ____) |  | |  | |     
#  |_____/   |_|  |_|    

 
class STP:
    """
    Short-Term Plasticity (STP) mechanism for dynamic weight modulation.
    
    Implements weight adjustments based on firing rates with potentiation and depression.
    
    Args:
        pot: Potentiation rate
        dep: Depression rate
        w: Initial weights
        fr: Baseline firing rate
        fr_seiz: Seizure firing rate
    """
    def __init__(self, pot, dep, w, fr, fr_seiz):
        self.pot = pot
        self.dep = dep
        with torch.no_grad():
            w = torch.FloatTensor(w)
            self.minimo_pesi =  [0.001 for i in range(len(pot))]
            self.corrente_tipica =  w*fr
            self.corrente_tipica_seiz =  w*fr_seiz
            print("minimo_pesi", self.minimo_pesi)
        self.massimo_pesi = [2 for i in range(len(pot))]
    def step_try(self, w, rate, fr):
        """
        Perform one STP weight update step.
        
        Args:
            w: Current weights
            rate: Firing rates for each neuron
            fr: Baseline firing rate
        
        Returns:
            tuple: (updated_weights, debug_string) - Updated weights and debug information
        """
        rate = np.array(rate)   
        abs_weights = abs(w) - ((self.dep) * rate * ((rate/4096 * np.array(w)) > np.array(self.corrente_tipica)))  + (self.pot) * (4096-rate) * ((rate/4096 * np.array(w)) < np.array(self.corrente_tipica))
        check_string = "rate/4096 * np.array(w) > np.array(self.corrente_tipica)\n"
        check_string = check_string + str(rate) + "/4096 "
        check_string = check_string + str(w) + " >"
        check_string = check_string + str(self.corrente_tipica) + "\n"
        # print(check_string)
        pesi = torch.copysign(np.maximum(abs_weights, self.minimo_pesi), w)
        pesi = torch.where(w > 0, pesi, w)
        return pesi, check_string
    


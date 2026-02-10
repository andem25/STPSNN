import torch
import torch.nn.functional as F
#  _                   
# | |                  
# | |     ___  ___ ___ 
# | |    / _ \/ __/ __|
# | |___| (_) \__ \__ \
# |______\___/|___/___/                                            


class SpikeRate(torch.nn.Module):
    """
    Custom loss function based on spike rate matching.
    
    Computes MSE loss between actual spike rates and target rates based on labels.
    
    Args:
        true_rate: Target spike rate for positive class (0-1)
        false_rate: Target spike rate for negative class (0-1)
        reduction: Reduction method for loss ('sum' or 'mean')
    """
    def __init__(
        self, true_rate, false_rate, reduction='sum'):
        super(SpikeRate, self).__init__()
        if not (true_rate >= 0 and true_rate <= 1):
            raise AssertionError(
                f'Expected true rate to be between 0 and 1. Found {true_rate=}'
            )
        if not (false_rate >= 0 and false_rate <= 1):
            raise AssertionError(
                f'Expected false rate to be between 0 and 1. '
                f'Found {false_rate=}'
            )
        self.true_rate = true_rate
        self.false_rate = false_rate
        self.reduction = reduction
        
        print("Spikerate loss 1 out")

    def forward(self, input, label):
        """
        Forward computation of spike rate loss.
        
        Args:
            input: Spike recordings tensor (time_steps, batch, outputs)
            label: Target labels
        
        Returns:
            torch.Tensor: Computed MSE loss between spike rates and target rates
        """
        input = input.transpose(1,0)
        input = input.transpose(1,2)

        spike_rate = input.mean(dim=-1)
        target_rate = self.true_rate * label + self.false_rate * (1 - label)
        return F.mse_loss(
            spike_rate.flatten(),
            target_rate.flatten(),
            reduction=self.reduction
        )
# STPSNN — Short-Term Plasticity for Spiking Neural Networks

Spiking Neural Network (SNN) with **Short-Term Plasticity (STP)** for EEG-based epileptic seizure detection. This work implements a biologically-inspired weight modulation mechanism that dynamically adapts synaptic weights during inference, improving seizure classification through adaptive thresholding.

> **Based on:** P. Busia, G. Leone, A. Matticola, L. Raffo and P. Meloni, "Wearable Epilepsy Seizure Detection on FPGA With Spiking Neural Networks," in *IEEE Transactions on Biomedical Circuits and Systems*, vol. 19, no. 6, pp. 1175-1186, Dec. 2025, doi: 10.1109/TBCAS.2025.3575327.

---

## Overview

The pipeline consists of two main phases:

1. **Training** — Train a 2-layer SNN with Leaky Integrate-and-Fire (LIF) neurons on encoded EEG data for binary seizure vs. non-seizure classification.
2. **Testing with STP** — Evaluate the trained network and apply Short-Term Plasticity to dynamically modulate synaptic weights based on firing rate statistics, enabling adaptive threshold-based seizure detection.

### Key Features

- **Level Encoding**: raw EEG values are encoded into 16-level binary spike trains (4 channels × 16 levels = 64 inputs)
- **Leaky Integrate-and-Fire Neurons**: 2-layer SNN architecture (64 → 8 → 1) with configurable decay rate and threshold
- **Spike Rate Loss**: custom loss function matching output firing rates to target rates (0.35 for seizure, 0.03 for non-seizure)
- **Short-Term Plasticity (STP)**: biologically-inspired potentiation/depression mechanism that adapts weights at inference time
- **Adaptive Thresholding**: dynamic deactivation rate that adjusts based on current weight distribution

---

## Repository Structure

```
STPSNN/
├── train_net.ipynb          # Training notebook (run this first)
├── test_net_stp.ipynb       # Testing & STP evaluation notebook
├── net_definition.py        # SNN architecture (Net class with optional STP)
├── STP_func.py              # Short-Term Plasticity mechanism
├── encoding_functions.py    # Thermometer encoding & EEG Dataset class
├── loss.py                  # Custom SpikeRate loss function
├── training_routine.py      # Training loop with validation & early stopping
├── test_routine.py          # Inference routine
├── trained_folder/          # Pre-trained model & normalization params
│   ├── network.pt           # Trained model weights
│   └── training_results.json # Max/min normalization values
├── LICENSE
└── README.md
```

---

## How to Run

> ⚠️ **Important**: Both notebooks require **Google Colab with High RAM runtime** due to the large EEG data encoding and long temporal sequences (2048 timesteps per window). Standard RAM will likely cause the session to crash.

### Step 1: Training the Network

1. Open [`train_net.ipynb`](train_net.ipynb) in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andem25/STPSNN/blob/main/train_net.ipynb)

2. **Change runtime to High RAM**:
   - Go to `Runtime` → `Change runtime type`
   - Enable **High-RAM** option
   - Click `Save`

3. Run all cells sequentially. The notebook will:
   - Install dependencies (`snntorch`, `mne`)
   - Clone this repository
   - Download the EEG dataset from Google Drive
   - Encode EEG signals into spike trains (thermometer encoding)
   - Train the SNN for up to 500 epochs with early stopping (patience=20)
   - Save the best model to `trained_folder/network.pt`

### Step 2: Testing with Short-Term Plasticity

1. Open [`test_net_stp.ipynb`](test_net_stp.ipynb) in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andem25/STPSNN/blob/main/test_net_stp.ipynb)

2. **Change runtime to High RAM** (same steps as above — this is critical for the test notebook as it processes the entire recording as a single continuous sequence)

3. Run all cells sequentially. The notebook will:
   - Load the pre-trained model from `trained_folder/`
   - Encode train and test EEG data into spike trains
   - Run baseline inference (without STP)
   - Compute STP parameters (potentiation/depression rates) from hidden layer firing statistics
   - Run inference with STP enabled (adaptive weight modulation)
   - Compare performance: **without STP** (static threshold) vs **with STP** (adaptive threshold)
   - Display confusion matrices and metrics (accuracy, sensitivity, specificity, precision)

> **Note**: A pre-trained model is already included in `trained_folder/`, so you can run the test notebook directly without training first.

---

## Dependencies

| Package    | Purpose                                |
|------------|----------------------------------------|
| `torch`    | Deep learning framework                |
| `snntorch` | Spiking Neural Network layers (LIF)    |
| `numpy`    | Numerical operations                   |
| `mne`      | EEG data handling                      |
| `gdown`    | Google Drive dataset download          |
| `matplotlib` | Visualization                        |
| `scikit-learn` | Classification metrics             |

All dependencies are automatically installed when running the notebooks in Google Colab.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

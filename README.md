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

## Dataset

This implementation uses the **CHB-MIT Scalp EEG Database**, a publicly available dataset from PhysioNet containing long-term EEG recordings from pediatric patients with intractable seizures.

**Dataset Details:**
- **Source**: [CHB-MIT Scalp EEG Database (PhysioNet)](https://physionet.org/content/chbmit/1.0.0/)
- **Subject**: Patient chb01 (22-channel recordings sampled at 256 Hz)
- **Selected Channels**: F7-T7, T7-P7, F8-T8, T8-P8 (4 channels)

**Training Files** (7 recordings):
- `chb01_02.edf`, `chb01_03.edf`, `chb01_04.edf`, `chb01_15.edf`, `chb01_16.edf`, `chb01_18.edf`, `chb01_21.edf`
- Segmented into **2048-sample windows** (8 seconds at 256 Hz)
- **80-20 train/validation split** with random shuffling for generalization

**Test Files** (5 recordings):
- `chb01_22.edf`, `chb01_23.edf`, `chb01_24.edf`, `chb01_25.edf`, `chb01_26.edf`
- Segmented into **2048-sample windows** preserving **chronological order** to simulate real-time detection
- No shuffling applied to maintain temporal continuity

**Labeling Strategy:**
A window is labeled as **SEIZURE (1)** if ≥50% of its samples overlap with annotated seizure periods; otherwise, it is labeled as **NON-SEIZURE (0)**. This threshold balances sensitivity while reducing false positives from brief seizure transitions.

> **Note**: A chronologically-ordered version of the training data is also available for experiments requiring temporal structure preservation.

### Preprocessed Data Structure

The notebooks automatically download preprocessed data organized in [this Google folder](https://drive.google.com/drive/folders/1EARnrSSj1DeHf0OiBmQ6_wcCJjKc8a2m?usp=drive_link) in pickle files as follows:

```
eeg_data_share/
├── test_routine/                  # Data for testing with chronological order
│   ├── test/
│   │   ├── label_window.pkl       # Test labels (seizure/non-seizure)
│   │   └── training_window.pkl    # Encoded test EEG windows
│   └── train/
│       ├── label_window.pkl       # Train labels for STP parameter computation
│       └── training_window.pkl    # Encoded train EEG windows (chronological)
└── train_routine/                 # Data for training with shuffling
    ├── train_data.pkl             # Encoded training windows (80% split, shuffled)
    ├── valid_data.pkl             # Encoded validation windows (20% split, shuffled)
    ├── y_train.pkl                # Training labels
    └── y_valid.pkl                # Validation labels
```


## Repository Structure

```
STPSNN/
├── train_net.ipynb          # Training notebook (run this first)
├── test_net_stp.ipynb       # Testing & STP evaluation notebook
├── net_definition.py        # SNN architecture (Net class with optional STP)
├── STP_func.py              # Short-Term Plasticity mechanism
├── encoding_functions.py    # Level encoding & EEG Dataset class
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

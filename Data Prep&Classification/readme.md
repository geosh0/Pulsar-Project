# 🧠 Data Prep & Classification Engine

This folder contains the Deep Learning pipeline used to classify the pulsar candidates. It moves from raw image loading to Model Training and Inference.

## 🏗️ Architecture: The "Hydra" CNN

We utilize a custom **Multi-Stream Convolutional Neural Network** (inspired by the PICS architecture).

Instead of treating the candidate plot as one large image, the model treats it as four distinct data streams. It has **four "Heads"** (separate CNN branches), each specialized in analyzing one specific visual feature:
1.  **Head A:** Analyzes the **Profile** (Shape).
2.  **Head B:** Analyzes the **Time Plot** (Continuity).
3.  **Head C:** Analyzes the **Freq Plot** (Broadband nature).
4.  **Head D:** Analyzes the **DM Curve** (Dispersion peaks).

The features extracted by these four heads are concatenated into a dense layer to make the final probability decision (Pulsar vs. Noise).

## 📂 File Description

*   **`main_notebook.ipynb`:** The orchestrator. Loads data, initializes the model, runs the training loop, plots loss/accuracy curves, and evaluates the Test Set.
*   **`pulsar_augmentation.py`:** Contains the `PulsarDataset` class and `torchvision` transforms.
    *   *Preprocessing:* Converts images to Grayscale tensors.
    *   *Normalization:* Fixes the "Domain Gap" (brightness difference) between PALFA and CSIRO images using Z-score normalization.
    *   *Augmentation:* Applies random shifts and "Cutouts" (simulating missing data) to prevent overfitting.
*   **`pulsar_train.py`:** Defines the PyTorch `nn.Module` class for the Multi-Stream CNN.

## 📊 Training Strategy
*   **Optimizer:** Adam (`lr=0.001`).
*   **Loss Function:** BCEWithLogitsLoss.
*   **Train/Test Split:** Performed using `GroupShuffleSplit` on the `group_id` column to ensure zero data leakage.
*   **Results:** The model typically achieves >99% Recall on the held-out test set, successfully distinguishing complex RFI from true pulsar signals.

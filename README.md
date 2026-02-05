# Platonic Representation Hypothesis: Convergence Analysis (MNIST-1D vs 2D)

This project investigates the **Platonic Representation Hypothesis**  by analyzing how different neural network architectures (MLPs vs. CNNs) and input modalities (1D synthetic signals vs. 2D real images) converge toward a shared representation geometry.

## Project Overview

The objective is to test whether independent models, trained to solve the same conceptual task (digit classification), learn similar feature spaces regardless of initialization (random seed) or data format.

### Key Features:

* **MNIST-1D Dataset**: A synthetic, low-dimensional (40 channels) version of MNIST designed to measure spatial inductive biases.


* **Centered Kernel Alignment (CKA)**: A similarity metric used to compare feature spaces, ensuring invariance to orthogonal transformation and isotropic scaling.


* **Platonic Validation**: A cross-domain comparison between the "ideal geometry" learned by a 2D CNN and multiple 1D MLPs.

---

## Model Architectures

The project compares two distinct neural network types, both projecting to a **128-dimensional** bottleneck to allow for direct CKA comparison:

| Model | Type | Input | Feature Latent Space |
| --- | --- | --- | --- |
| **Mnist1DNet** | Multi-Layer Perceptron (MLP) | 1D Vector (40) | 128 Dimensions |
| **Mnist2DNet** | Convolutional Neural Network (CNN) | 2D Image (28x28) | 128 Dimensions |

---

## Methodology & CKA

To quantify the alignment between feature matrices  and , we implement **Linear CKA**:

1. **Centering**: Features are centered by subtracting the mean using a centering matrix .


2. **HSIC Calculation**: We compute the Hilbert-Schmidt Independence Criterion to measure statistical dependence.


3. **Normalization**: The score is normalized to the range `[0, 1]`, where 1 indicates identical representations up to rotation and scaling.

---

## Experimental Results

The findings demonstrate strong internal convergence but partial cross-domain invariance:

* **1D Convergence (Self-CKA) ≈ 0.96**: 1D models trained with different seeds reach nearly identical internal geometries.


* **Platonic Score (1D vs 2D) ≈ 0.25**: There is a measurable structural correlation between 1D and 2D representations, though it is influenced by architectural differences.

---

## Setup & Installation

This project was developed using a **Conda environment**.

### 1. Create the Environment

```bash
conda create --name platonic_vision python=3.10
conda activate platonic_vision

```

### 2. Install Dependencies

Install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

```

*Note: The environment includes `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, and the specific `mnist1d` library.*

### 3. Run the Experiment

Launch the Jupyter notebook to replicate the training and analysis:

```bash
jupyter notebook representation_convergence.ipynb

```

---

## References

* Kornblith et al. (2019): *Similarity of Neural Network Representations Revisited* (CKA methodology).


* **Greydanus (2020)**: *MNIST-1D Dataset*.


* Isola et al. (2024): *The Platonic Representation Hypothesis*.
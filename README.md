# Platonic Representation Hypothesis: Symmetric Convergence Analysis

This project investigates the limits of the **Platonic Representation Hypothesis** in a multimodal context.
We analyze whether neural networks trained on radically different input topologies (**1D synthetic signals** vs. **2D spatial images**) converge toward a shared representation geometry.

## Research Question
Does the learned geometric structure depend more on the **Input Topology** (Architecture) or on the **Semantic Concepts** (Classes)?

---

## Methodology: Symmetric Setup & Alignment
This experiment utilizes a **symmetric 5x5 validation setup**. We train 5 independent versions of both 1D and 2D models to distinguish between chance alignment and true convergence.

### Critical Implementation Detail: Class-Conditional Alignment
To ensure the mathematical validity of CKA between disjoint datasets (MNIST-1D and MNIST-2D), we apply a strict **alignment protocol**:
1.  **Feature Extraction**: Latent vectors are extracted from the bottleneck layer.
2.  **Sorting**: Samples in the feature matrices X and Y are **sorted by class label (0-9)**.
3.  **Correspondence**: This ensures that row *i* in both matrices corresponds to the same semantic concept, allowing CKA to measure structural similarity despite different input modalities.

---

## Model Architectures
Both models project their internal representations to a shared **128-dimensional** bottleneck to allow for direct comparison:

| Model | Type | Input | Inductive Bias | Feature Space |
| :--- | :--- | :--- | :--- | :--- |
| **Mnist1DNet** | MLP | 1D Vector (40) | Global/Sequential | 128 Dimensions |
| **Mnist2DNet** | CNN | 2D Image (28x28) | Local/Spatial | 128 Dimensions |

---

## Experimental Results
The results reveal an interesting dichotomy between semantic clustering and geometric topology:

* **1D Convergence (Self-CKA) ≈ 0.96**: High stability. Models with the same architecture learn nearly identical representations.
* **2D Convergence (Self-CKA) ≈ 0.95**: CNNs also exhibit consistent internal geometry across seeds.
* **Platonic Score (1D vs 2D) ≈ 0.24**: **Partial Convergence**.
    * *Interpretation:* The score is significantly above random, indicating a **Shared Semantic Structure** (classes are clustered similarly). However, the low value compared to self-convergence highlights a significant **Modality Gap**, suggesting that the "shape" of the representation remains heavily influenced by the input topology (1D vs 2D).

---

## Mathematical Core: Linear CKA
To quantify the alignment, we use **Centered Kernel Alignment (CKA)**:

$$CKA(X, Y) = \frac{HSIC(X, Y)}{\sqrt{HSIC(X, X) \cdot HSIC(Y, Y)}}$$

Where HSIC measures statistical dependence between the centered Gram matrices of the sorted features.

---

## Setup & Installation

This project was developed using a **Conda environment**.

### 1. Create the Environment
```bash
conda create --name platonic_vision python=3.10
conda activate platonic_vision

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
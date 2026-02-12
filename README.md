# Platonic Representation Hypothesis: Symmetric Convergence Analysis

This project investigates the **Platonic Representation Hypothesis** by analyzing how different neural network architectures (MLPs vs. CNNs) and input modalities (1D synthetic signals vs. 2D real images) converge toward a shared representation geometry.

## Project Evolution: Symmetric Setup
Originally designed to compare multiple 1D models against a single 2D anchor, the experiment has been upgraded to a **symmetric 5x5 configuration**. We now train 5 independent versions of both the 1D and 2D models to ensure that our "Platonic Score" is robust and not biased by a single random initialization.

### Key Features:
* **MNIST-1D & MNIST-2D**: A direct comparison between synthetic 1D signals and standard 2D spatial images.
* **Centered Kernel Alignment (CKA)**: A similarity metric used to compare feature spaces, ensuring invariance to orthogonal transformation and isotropic scaling.
* **Robust Statistical Validation**: The final metrics are derived from the average of **25 cross-domain comparisons** (5x5 matrix).

---

## Model Architectures
Both models project their internal representations to a **128-dimensional** bottleneck to allow for direct CKA comparison:

| Model | Type | Input | Feature Latent Space |
| :--- | :--- | :--- | :--- |
| **Mnist1DNet** | MLP | 1D Vector (40) | 128 Dimensions |
| **Mnist2DNet** | CNN | 2D Image (28x28) | 128 Dimensions |

---

## Methodology & CKA
To quantify the alignment between feature matrices $X$ and $Y$, we implement **Linear CKA**:

1.  **Centering**: Features are centered using a centering matrix $H$.
2.  **HSIC Calculation**: We compute the Hilbert-Schmidt Independence Criterion to measure statistical dependence.
3.  **Normalization**: The score is normalized to the range $[0, 1]$.

$$CKA(X, Y) = \frac{HSIC(X, Y)}{\sqrt{HSIC(X, X) \cdot HSIC(Y, Y)}}$$

---

## Experimental Results
The findings demonstrate strong internal convergence within each domain and a measurable, though partial, cross-domain invariance:

* **1D Consistency (Self-CKA) ≈ 0.96**: 1D models consistently reach nearly identical internal geometries regardless of the seed.
* **2D Consistency (Self-CKA) ≈ 0.98**: CNNs show even higher stability in their learned representations.
* **Platonic Match (1D vs 2D) ≈ 0.25**: A stable structural correlation exists across domains, supporting the hypothesis that different modalities can converge toward shared conceptual representations.

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
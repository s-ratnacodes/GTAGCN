# GTAGCN

# Scaling Node Classification with Generalized Topology Adaptive Networks

**[S Ratna¹, Sukhdeep Singh², Anuj Sharma¹*]** 
[1 Department of Computer Science and Applications, Panjab University, Chandigarh, India
sratna@pu.ac.in, anujs@pu.ac.in
2 Department of Computer Science, D.M. College (affiliated to Panjab University, Chandigarh), Moga, Punjab, India
sukha13@ymail.com]    
> 📦 Framework: PyTorch Geometric  
> 🧪 Task: Node Classification

---

## Overview

This repository contains the official implementation of **GTAGCN**, evaluated on **9 benchmark datasets** from PyTorch Geometric for the node classification task.


## Datasets

All 9 datasets are loaded directly from [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/):

| # | Dataset | PyG Source | Category |
|---|---------|-----------|----------|
| 1 | Cora | `Planetoid` | Citation Network |
| 2 | CiteSeer | `Planetoid` | Citation Network |
| 3 | PubMed | `Planetoid` | Citation Network |
| 4 | Amazon Photo | `Amazon` | Co-purchase Graph |
| 5 | Amazon Computers | `Amazon` | Co-purchase Graph |
| 6 | Cornell | `WebKB` | Webpage Graph |
| 7 | Texas | `WebKB` | Webpage Graph |
| 8 | Wisconsin | `WebKB` | Webpage Graph |
| 8 | Cora_ML | `CitationFull` | Citation Graph |

> **Note:** All datasets are automatically downloaded by PyG on first run — no manual download needed.


---

## File Structure

```
.
├── gtagcn1.ipynb         # Main model v1 — run on Google Colab
├── gtagcn2.ipynb         # Main model v2 — run on Google Colab
├── gtagcn3.ipynb         # Main model v3 — run on Google Colab
│
├── ab1.ipynb             # Ablation study 1 — run on GPU
├── ab2.ipynb             # Ablation study 2 — run on GPU
├── ab3.ipynb             # Ablation study 3 — run on GPU
├── ab4.ipynb             # Ablation study 4 — run on GPU
│
└── README.md
```

> 3 main code files (`gtagcn1/2/3`) · 4 ablation files (`ab1/2/3/4`) · 7 files total  
> 🟡 Main files: Google Colab &nbsp;|&nbsp; 🔵 Ablation files: GPU server

---

## Running the Code

### Main Experiments (Google Colab)

Open any of the three notebooks in [Google Colab](https://colab.research.google.com/):

```
gtagcn1.ipynb   ← Model variant 1
gtagcn2.ipynb   ← Model variant 2
gtagcn3.ipynb   ← Model variant 3
```

Install dependencies by running the first cell:

```python
!pip install torch_geometric
```

### Ablation Studies (GPU)

Run on a machine with a CUDA-enabled GPU:

```bash
jupyter notebook ab1.ipynb
jupyter notebook ab2.ipynb
jupyter notebook ab3.ipynb
jupyter notebook ab4.ipynb
```

---


## Installation

```bash
pip install torch torch_geometric scikit-learn matplotlib
```

---

---

## Results

Results are reported as **Mean ± Std** over **10 runs** with random 80/20 node splits.

| Dataset | Accuracy (%) |
|---------|-------------|
| Cora | XX.XX ± X.XX |
| CiteSeer | XX.XX ± X.XX |
| PubMed | XX.XX ± X.XX |
| Amazon Photo | XX.XX ± X.XX |
| Amazon Computers | XX.XX ± X.XX |
| Cornell | XX.XX ± X.XX |
| Texas | XX.XX ± X.XX |
| Wisconsin | XX.XX ± X.XX |

---


## System Requirements

> ⚠️ Please fill in your actual system specs below.

| Component | Specification |
|-----------|--------------|
| OS |Windows 10 (Version 10.0.20348) |
| GPU | 2 × NVIDIA L4 |
| CUDA | 11.8 |
| RAM | 137.18 GB |
| Python |3.11.7 |
| PyTorch | 2.7.1+cu118 |
| PyG (torch_geometric) | 2.6.1 |
| VRAM | 23.91 GB (each) |


---




## Datasets

All 9 datasets are loaded directly from [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/):

| # | Dataset | PyG Source | Category |
|---|---------|-----------|----------|
| 1 | Cora | `Planetoid` | Citation Network |
| 2 | CiteSeer | `Planetoid` | Citation Network |
| 3 | PubMed | `Planetoid` | Citation Network |
| 4 | Amazon Photo | `Amazon` | Co-purchase Graph |
| 5 | Amazon Computers | `Amazon` | Co-purchase Graph |
| 6 | Cornell | `WebKB` | Webpage Graph |
| 7 | Texas | `WebKB` | Webpage Graph |
| 8 | Wisconsin | `WebKB` | Webpage Graph |

> **Note:** All datasets are automatically downloaded by PyG on first run — no manual download needed.

---

## File Structure

```
.
├── gtagcn1.ipynb         # Main model v1 — run on Google Colab
├── gtagcn2.ipynb         # Main model v2 — run on Google Colab
├── gtagcn3.ipynb         # Main model v3 — run on Google Colab
│
├── ab1.ipynb             # Ablation study 1 — run on GPU
├── ab2.ipynb             # Ablation study 2 — run on GPU
├── ab3.ipynb             # Ablation study 3 — run on GPU
├── ab4.ipynb             # Ablation study 4 — run on GPU
│
└── README.md
```

> 3 main code files (`gtagcn1/2/3`) · 4 ablation files (`ab1/2/3/4`) · 7 files total  
> 🟡 Main files: Google Colab &nbsp;|&nbsp; 🔵 Ablation files: GPU server
> [Google Colab](https://colab.research.google.com/):
---


## Installation

```bash
pip install torch torch_geometric scikit-learn matplotlib
```

---

## Results

Results are reported as **Mean ± Std** over **10 runs** with random 80/20 node splits.

| Dataset | Accuracy (%) |
|---------|-------------|
| Cora | XX.XX ± X.XX |
| CiteSeer | XX.XX ± X.XX |
| PubMed | XX.XX ± X.XX |
| Amazon Photo | XX.XX ± X.XX |
| Amazon Computers | XX.XX ± X.XX |
| Cornell | XX.XX ± X.XX |
| Texas | XX.XX ± X.XX |
| Wisconsin | XX.XX ± X.XX |
| Cora_ML | XX.XX ± X.XX |

---


 
## Acknowledgment
 
The work presented in this paper was supported by the Department of Computer Science and Applications (DCSA), Panjab University, Chandigarh, India, which provided the necessary GPU resources to conduct the experiments.
 
- **[PyTorch](https://pytorch.org/)** — The deep learning framework used for all model implementation and training.
- **[PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/)** — The graph neural network library used for dataset loading, graph convolutions, and sparse operations.
- TAGConv — Du, Jian, et al. "Topology adaptive graph convolutional networks." arXiv preprint arXiv:1710.10370 (2017).
- GENConv — Li, Guohao, et al. "Deepergcn: All you need to train deeper gcns." arXiv preprint arXiv:2006.07739 (2020).
- **[scikit-learn](https://scikit-learn.org/)** — Used for t-SNE embeddings and evaluation utilities.
- **[Google Colab](https://colab.research.google.com/)** — Free GPU/TPU compute used for running main experiments.

 
---
 

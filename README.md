# GTAGCN 

# GTAGCN: Generalized Topology Adaptive Graph Convolutional Networks

**[S Ratna¹, Sukhdeep Singh², Anuj Sharma¹*]** 
[1 Department of Computer Science and Applications, Panjab University, Chandigarh, India
sratna@pu.ac.in, anujs@pu.ac.in
2 Department of Computer Science, D.M. College (affiliated to Panjab University, Chandigarh), Moga, Punjab, India
sukha13@ymail.com]    
> 📦 Framework: PyTorch Geometric   
> 🧪 Task: Node Classification

---

## Overview

This repository contains the official implementation of **GTAGCN**, evaluated on **seven benchmark datasets** from PyTorch Geometric for the node classification task.


## Datasets

All 9 datasets are loaded directly from [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/):

| # | Dataset | PyG Source | Category |
|---|---------|-----------|----------|
| 1 | PubMed | `Planetoid` | Citation Network |
| 2 | Amazon Photo | `Amazon` | Co-purchase Graph |
| 3 | Amazon Computers | `Amazon` | Co-purchase Graph |
| 4 | Cornell | `WebKB` | Webpage Graph |
| 5 | Texas | `WebKB` | Webpage Graph |
| 6 | Wisconsin | `WebKB` | Webpage Graph |
| 7 | CiteSeerFull | `CitationFull` | Citation Graph |

> **Note:** All datasets are automatically downloaded by PyG on first run — no manual download needed.


---

## File Structure

```
.
├── GTAGCNExp1.ipynb         # Main model v1 — run on Google Colab
├── GTAGCNExp2.ipynb         # Main model v2 — run on Google Colab
├── GTAGCNExp3.ipynb         # Main model v3 — run on Google Colab
│
├── GTAGCN_ABlation             # Ablation study 1 — run on GPU
└── README.md
```

> 3 main code files (`GTAGCN1/2/3`) · 1 ablation file (`GTAGCN_ABlation`) · 4 files total  


---

## Running the Code

### Main Experiments (Google Colab)

Open any of the three notebooks in [Google Colab](https://colab.research.google.com/):

```
GTAGCNExp1  ← Model variant 1
GTAGCNExp2  ← Model variant 2
GTAGCNExp3   ← Model variant 3
```


```python
!pip install torch_geometric
```




## Installation

```bash
pip install torch torch_geometric scikit-learn matplotlib
```

---



## Results

Results are reported as **Mean ± Std** over **10 runs** for three random node splits i.e. 60:20:20, 70:15:15, 80:10:10.


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
 

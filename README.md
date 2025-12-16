# 🔬 Evaluation Framework for Centralized, Hierarchical, and Decentralized Aggregation in Federated Systems

> **Official implementation and experimental companion to the arXiv paper:**
> *Evaluation Framework for Centralized and Decentralized Aggregation Algorithm in Federated Systems*
> **https://doi.org/10.48550/arXiv.2512.10987**

[![arXiv](https://img.shields.io/badge/arXiv-2512.10987-b31b1b.svg)](https://arxiv.org/abs/2512.10987)
[![Federated Learning](https://img.shields.io/badge/Domain-Federated%20Learning-green.svg)]()
[![Decentralized FL](https://img.shields.io/badge/Focus-Decentralized%20Aggregation-blue.svg)]()
[![Reproducible](https://img.shields.io/badge/Reproducibility-Deterministic%20Runs-brightgreen.svg)]()

---

## 📌 Abstract 

Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy. This repository provides the **official experimental framework** used in the paper *"Evaluation Framework for Centralized and Decentralized Aggregation Algorithm in Federated Systems"*. The framework rigorously evaluates **Hierarchical Federated Learning (HFL)** against **Decentralized Aggregated Federated Learning (AFL)** and **Decentralized Continual Federated Learning (CFL)** using **MNIST** and **Fashion-MNIST** benchmarks. Experimental results demonstrate that decentralized aggregation strategies consistently outperform hierarchical centralized approaches in terms of **accuracy, precision, recall, F1-score, and computational efficiency**, highlighting the growing importance of decentralized federated paradigms.

---

## 📘 Overview

**Keywords:** Federated Learning, Hierarchical Federated Learning (HFL), Aggregated Federated Learning (AFL), Continual Federated Learning (CFL), Decentralized Aggregation, FedAvg, Distributed Computing, Privacy-Preserving Machine Learning

This repository serves as a **reproducible research companion** to the arXiv paper. It enables systematic comparison of **centralized hierarchical** and **fully decentralized federated learning architectures** under controlled experimental conditions.

The framework is designed for:

* Academic reproducibility
* Benchmark-driven evaluation
* Comparative analysis of aggregation paradigms

---

## ✨ Federated Learning Paradigms Implemented

### 🏛️ Hierarchical Federated Learning (HFL)

* Central server orchestrates global aggregation
* Intermediate cluster/group-level aggregation
* Susceptible to communication bottlenecks and centralized privacy risks

### 🔗 Decentralized Aggregated Federated Learning (AFL)

* No central server dependency
* Clients collaboratively aggregate local models
* Improved scalability and reduced communication overhead

### 🔄 Decentralized Continual Federated Learning (CFL)

* Continuous refinement of local and global models
* Supports continual learning across federated rounds
* Demonstrates superior accuracy and lower inference latency

---

## 🏗️ System Architecture


* **Figure 1:** Architecture overview of Hierarchical Federated Learning (HFL)
  
<p align="center">
  <img width="600" height="400" alt="Image" src="https://github.com/user-attachments/assets/52d5258e-131d-46df-b796-470b6e76b21c" />
</p>

* **Figure 2:** Architecture overview of Decentralized Aggregated Federated Learning (AFL)

<p align="center">
  <img width="600" height="400" alt="Image" src="https://github.com/user-attachments/assets/3212e745-241d-4c54-bd5b-09e6f0fe5585" />
</p>

* **Figure 3:** Architecture overview of Decentralized Continual Federated Learning (CFL)

<p align="center">
  <img width="600" height="400" alt="Image" src="https://github.com/user-attachments/assets/1b8401ea-4d58-4306-81f4-63620a7534ec" />
</p>

---

## 🧠 Model Architecture

### Convolutional Neural Network (CNN)

The same CNN architecture is used across all FL paradigms to ensure fair comparison:

* 3 Convolutional layers (16, 12, 10 filters)
* Kernel size: 3×3
* ReLU activation
* Max-pooling layers for downsampling

<p align="center">
  <img width="700" height="250" alt="Image" src="https://github.com/user-attachments/assets/b74ddeb7-151b-4047-b962-cb4f68a5c7f2" />
</p>

---

## 📊 Datasets

* **MNIST:** Handwritten digit classification (28×28)
* **Fashion-MNIST:** Apparel image classification

Both datasets are publicly available and widely adopted for FL benchmarking.

---

## 📏 Evaluation Metrics

The framework computes the same metrics reported in the paper:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Build Time (Training Time)**
* **Classification Time (Inference Time)**

These metrics enable holistic evaluation of **predictive performance** and **computational efficiency**.

---

## 🔢 Aggregation Mechanism (FedAvg)

```math
\theta_g = \sum_{c=1}^{N} \frac{n_c}{N} \theta_c
```

where:

* $\theta_g$ — global model parameters
* $\theta_c$ — client model parameters
* $n_c$ — number of samples at client $c$
* $N$ — total number of clients

---

## 📈 Experimental Results Summary

Key findings reproduced by this framework:

* **Decentralized FL (AFL, CFL)** consistently outperforms **Hierarchical FL**
* **CFL** achieves the highest accuracy and lowest classification time
* **AFL** offers the fastest build time with competitive accuracy

> Refer to Tables 1 & 2 and Figures 9–14 in the paper for detailed quantitative results.

---

## 🛠️ Installation

```bash
# Python 3.9+ recommended
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

---

## 🔁 Reproducibility Notes

* Fixed random seeds for deterministic runs
* Identical CNN and dataset splits across paradigms
* Metric computation aligned exactly with the paper

---

## 📜 Copyright Registration

**Registered with the Government of India**
Copyright Certificate Number: **SW-19383/2024**

<p align="center">
  <img width="600" height="900" alt="Image" src="https://github.com/user-attachments/assets/e1996703-d64f-49b9-885a-be6cdc7f9708" />
</p>

---

## 📚 Citation

If you use this codebase, please cite the paper:

```bibtex
@article{chongder2025evaluation,
  title   = {Evaluation Framework for Centralized and Decentralized Aggregation Algorithm in Federated Systems},
  author  = {Chongder, Sumit},
  journal = {arXiv preprint arXiv:2512.10987},
  year    = {2025}
}
```

---

## 🎯 Intended Audience

* Federated Learning researchers
* Distributed systems practitioners
* Graduate students and academic reviewers

---

> 🔗 **Paper–Code Linkage:**
> This repository is the official implementation accompanying the arXiv paper:
> 👉 [https://arxiv.org/abs/2512.10987](https://arxiv.org/abs/2512.10987)



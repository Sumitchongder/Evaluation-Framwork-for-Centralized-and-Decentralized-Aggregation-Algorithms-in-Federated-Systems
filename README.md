# Evaluation Framework for Centralized, Hierarchical, and Decentralized Aggregation in Federated Systems

## Overview
This project provides a rigorous, researcher‑grade evaluation framework for federated learning (FL) aggregation strategies:
- **Centralized (FedAvg):** Server‑mediated weighted averaging of client updates.
- **Hierarchical (Clustered):** Multi‑level aggregation via cluster heads, then global fusion.
- **Decentralized (Gossip):** Peer‑to‑peer parameter mixing over a connected graph topology.

It models privacy, robustness, and communication constraints encountered in real federated deployments, enabling principled trade‑off analysis and reproducible benchmarking.

---

## Key features
- **Aggregation strategies:**
  - **Centralized:** FedAvg with straggler modeling.
  - **Hierarchical:** Cluster‑level aggregation made robust with DP/clipping/compression.
  - **Decentralized:** Gossip over a synthetic connected graph (Watts‑Strogatz) with multi‑round neighbor mixing.
- **Privacy mechanisms:**
  - **Differential privacy (DP) noise:** Gaussian noise on model parameters.
  - **Gradient clipping:** Per‑tensor L2 clipping to bound sensitivity.
- **Communication modeling:**
  - **Top‑k compression:** Retain a fraction of high‑magnitude coordinates.
  - **Latency and stragglers:** Simulate network delay and client dropouts.
  - **Per‑round communication cost:** Tracks MB exchanged per round.
- **Robustness modeling:**
  - **Label‑flip attack:** Stochastic mislabeling over a configurable fraction of clients.
  - **Gradient‑amplification attack:** Malicious clients inflate gradients to destabilize aggregation.
  - **Acc‑drop vs baseline:** Quantifies robustness under attack.
- **Fairness analysis:**
  - **Per‑client accuracy distribution:** Computes standard deviation as a fairness signal.
- **Reproducibility:**
  - **Seed control and YAML export:** Deterministic runs; configuration saved for re‑runs.

---

## System architecture
- **Data layer:** Digits dataset (64‑D input, 10 classes), with **IID** and **Non‑IID shard‑based partitioning** to mimic heterogeneous data silos.
- **Client layer:** Local SGD with configurable batch size, epochs, and attacks (label flip / gradient amplification).
- **Model layer:** A **SimpleMLP** (BatchNorm + ReLU) for compact, stable training; can be swapped for CNNs/Transformers.
- **Aggregation layer:**
  - **FedAvg:** Weighted average with DP noise, clipping, compression, stragglers, latency.
  - **Hierarchical:** Cluster‑wise aggregation and global fusion.
  - **Gossip:** Multi‑round neighbor mixing over connected graph.
- **Evaluation layer:** Tracks global performance, fairness, communication, and robustness metrics per round.

---

## Metrics
- **Accuracy and loss:** Global test metrics post‑aggregation.
- **Fairness (std):** Standard deviation of per‑client accuracies; lower implies more equitable performance.
- **Communication cost:** Estimated MB transferred per round (uploads, broadcasts, or neighbor mixing).
- **Robustness (acc drop vs baseline):** Difference from first‑round baseline when attacks are enabled.

Mathematically, fairness is computed as:
\[
\text{FairnessStd} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(a_i - \bar{a}\right)^2}
\]
where \(a_i\) is client \(i\)'s accuracy, and \(\bar{a}\) is the mean accuracy.

---

## Installation
```bash
# Python 3.9+ recommended
pip install streamlit torch torchvision scikit-learn numpy pandas matplotlib networkx pyyaml

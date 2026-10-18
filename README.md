# Deep Learning Foundations: Geoffrey Hinton Papers & Replications in PyTorch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A structured, educational journey to master modern deep learning and AI fundamentals through step-by-step mathematical notes and PyTorch implementations of **Geoffrey Hinton's landmark papers**.

From **Backpropagation** and **Vector Embeddings** to **Transformers (LayerNorm)**, **Mixture of Experts (MoE)**, and **Deep Belief Nets**.

---

## 🚀 Quick Navigation

* 🗺️ **[ROADMAP.md](ROADMAP.md)** — **Curated Modern AI study roadmap (12 high-impact papers)**.
* 📝 **[notes/](notes/)** — Deep-dive mathematical notes and derivations.
* 📥 **[scripts/download_papers.py](scripts/download_papers.py)** — Automatically downloads all research papers into `Papers/`.
* 📦 **[archive/](archive/)** — Additional historical papers and exploratory implementations.

---

## 🎯 Goals

- **Understand the Math:** Line-by-line breakdown of optimization, energy models, and graphical models.
- **Connect Theory to Modern AI:** Trace how 1980s–2010s breakthroughs formed today's Transformers, LLMs, and diffusion models.
- **Clean PyTorch Replications:** Understand each algorithm via readable, reproducible code.
- **Track Progress:** Document assumptions, math derivations, and learning dynamics.

---

## 📥 Downloading Research Papers

To keep the repository lightweight, original PDF papers are not stored in git. You can download all 22 foundational papers in one command:

```bash
python scripts/download_papers.py
```
This downloads the original papers directly into the `Papers/` directory.

---

## 📚 Curated Study Roadmap (Core 12 Papers)

| Phase | # | Paper | Core Modern Concept | Notebook |
| :---: | :---: | :--- | :--- | :--- |
| **1. Optimization & Embeddings** | 01 | **Backpropagation (1986)** | Gradient descent, chain rule | [Notebook](Learning_representations_by_back_propagating_errors.ipynb) |
| | 02 | **Distributed Representations (1986)** | Vector embeddings & latent geometry | [Notebook](Distributed_Representations.ipynb) |
| | 03 | **Initialization & Momentum (2013)** | Xavier/He initialization, Nesterov momentum | [Notebook](On_the_importance_of_initialization_and_momentum_in_deep_learning.ipynb) |
| **2. Modern Architecture Blocks** | 04 | **ReLU Activation (2010)** | Non-saturating activations | [Notebook](Rectified_Linear_Units_Improve_Restricted_Boltzmann_Machines.ipynb) |
| | 05 | **Dropout (2014)** | Ensembling & regularization | [Notebook](Dropout_A_Simple_Way_to_Prevent_Neural_Networks_from_overfitting.ipynb) |
| | 06 | **Layer Normalization (2016)** | **Core of all Transformers & LLMs** | [Notebook](Layer_Normalization.ipynb) |
| | 07 | **Mixture of Experts / MoE (1991)** | **Modern Sparse Frontier LLMs** | [Notebook](Adaptive_Mixtures_of_Local_Experts.ipynb) |
| **3. Scaling & Practical DL** | 08 | **AlexNet (2012)** | Convolutions, GPUs, ImageNet | [Notebook](ImageNet_Classification_with_Deep_Convolutional_Neural_Networks_AlexNet.ipynb) |
| | 09 | **t-SNE (2008)** | High-dimensional embedding visualization | [Notebook](Visualizing_Data_using_t_SNE.ipynb) |
| | 10 | **Knowledge Distillation (2015)** | Teacher-student model compression | [Notebook](Distilling_the_Knowledge_in_a_Neural_Network.ipynb) |
| **4. Generative & Energy Models** | 11 | **Boltzmann Machines & CD (1985/2002)** | Energy landscapes, MCMC, CD-1 | [Notebook 1](A_Learning_Algorithm_for_Boltzmann_Machines.ipynb) / [Notebook 2](Training_Products_of_Experts_by_Minimizing_Contrastive_Divergence.ipynb) |
| | 12 | **Deep Belief Nets (2006)** | Layer-wise pre-training, complementary priors | [Notebook](A_fast_learning_algorithm_for_deep_belief_nets.ipynb) |

*For the full 30-paper collection, see the [archive/](archive/) directory.*

---

## 🛠️ Credits & Attribution

- **Implementations:** Originally created by **Mohammed Fahd Abrah** ([MOHAMMEDFAHD](https://github.com/MOHAMMEDFAHD/Geoffrey-Hinton-Papers-Replicating-In-Pytorch)).
- **Curated Roadmap & Mathematical Study Hub:** Maintained by **Parameswar Rout** ([@parameswarrout](https://github.com/parameswarrout)).

---

## 📄 License

This repository is distributed under the **MIT License**.

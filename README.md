# Assessing and Comparing Carbon Footprint Estimation Tools for Deep Learning Workloads

## Overview

This repository contains the code, experiments, analysis, and results developed for the Master's thesis:

**"Assessing and Comparing Carbon Footprint Estimation Tools for Deep Learning Workloads"**

The objective of this work is to evaluate and compare software tools that estimate the environmental impact of deep learning experiments, focusing on:

* Energy consumption estimation
* Carbon emission estimation
* Estimation accuracy
* Computational overhead
* Reproducibility

The thesis introduces the **Accuracy-Efficiency Ratio (AER)**, a metric designed to quantify the trade-off between machine learning performance improvements and their associated environmental costs.

---

## Research Objectives

This project investigates the following research questions:

1. How do different carbon footprint estimation tools compare in terms of reported energy consumption and CO₂ emissions?
2. What are the differences between process-level and machine-level tracking approaches?
3. What is the overhead introduced by environmental monitoring tools?
4. How can model performance improvements be evaluated alongside environmental impact?
5. Can a unified framework simplify environmentally aware hyperparameter optimization?

---

## Carbon Footprint Estimation Tools

The following tools are evaluated:

* **CodeCarbon**
* **CarbonTracker**
* **eco2ai**
* **Cumulator** 

These tools differ in:

* Hardware monitoring approach
* Carbon intensity assumptions
* Power estimation methodology
* System-level vs process-level tracking

---

## Deep Learning Workloads

Two workloads are used to represent different computational scales.

### MNIST Classification

* Dataset: MNIST
* Model: Simple Neural Network (SimpleNN)
* Metric: Classification Accuracy

This workload represents a lightweight experiment.

### Tiny ImageNet Denoising

* Dataset: Tiny ImageNet
* Model: DnCNN-style denoising network
* Metric: Peak Signal-to-Noise Ratio (PSNR)

This workload represents a more computationally intensive experiment.

---

## Hardware and Software Environment

Experiments were performed using:

* Apple MacBook Air M3
* 16 GB RAM
* macOS
* Python 3.13
* PyTorch
* Apple Metal Performance Shaders (MPS)

The use of MPS allows GPU acceleration on Apple Silicon devices.

---

## Accuracy-Efficiency Ratio (AER)

The thesis proposes the **Accuracy-Efficiency Ratio (AER)**:

$$
AER = \frac{\Delta Performance}{\Delta Environmental\ Cost}
$$

where:

* Performance = Accuracy (MNIST) or PSNR (DnCNN)
* Environmental Cost = Energy consumption, CO₂ emissions, or execution time

Higher AER values indicate greater performance improvements per unit of environmental impact.

The metric supports:

* Energy AER
* CO₂ AER
* Time AER
* Relative AER comparisons

---

## GreenOptim Package

This repository includes the `greenoptim` package, a prototype framework designed to integrate:

* Hyperparameter optimization
* Environmental impact estimation
* Multi-objective evaluation
* Sustainability metrics

### Main Components

```text
greenoptim/
├── datasets.py
├── metrics.py
├── optimizer.py
├── models/
└── trackers/
```

Supported trackers:

* CodeCarbon
* CarbonTracker
* eco2ai
* Cumulator

---

## Repository Structure

```text
thesis_carbon/
│
├── src/                    # Experimental scripts
│   ├── mnist/
│   ├── dncnn/
│   ├── optuna/
│   └── utils/
│
├── greenoptim/             # Python package
│
├── results/
│   ├── csv/
│   └── plots/
│
├── analysis/
│   ├── mnist/
│   └── dncnn/
│
├── README.md
└── pyproject.toml
```

---

## Installation

Clone the repository:

```bash
git clone git@github.com:teszc/thesis_carbon.git
cd thesis_carbon
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

---

## Running Experiments

Example:

MNIST with CodeCarbon:

```bash
python -m src.mnist.mnist_codecarbon
```

DnCNN with CarbonTracker:

```bash
python -m src.dncnn.dncnn_carbontracker
```

Optuna optimization:

```bash
python -m src.optuna.mnist.mnist_optuna_codecarbon
```

---

## Results

The repository contains:

* Experimental CSV files
* Comparative plots
* AER analyses
* Optuna optimization results
* Performance and environmental evaluations

Analysis outputs are available in:

```text
analysis/
```

and

```text
results/
```

---

## Dataset Availability

Datasets are not included in this repository because of their size.

### MNIST

Automatically downloaded using torchvision.

### Tiny ImageNet

Download separately from:

https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

Place the dataset in the expected data directory before running experiments.

---

## License

This repository is provided for academic and research purposes.

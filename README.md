# Hybrid Graph Neural Network + Physics-Informed Neural Network

## Predicting Interface Dynamics in Semiconductor Thin Film Deposition

**Author:** Randall Crawford

**Affiliation:** National University — M.S. Data Science

**Collaborators:** Roman Ostroumov, David Homrighouse

**Date:** December 2025

---

## Overview

This repository contains the research implementation and supporting materials for a **hybrid Graph Neural Network–Physics-Informed Neural Network (GNN-PINN)** framework developed to model and optimize **interface dynamics in atomic layer deposition (ALD)** processes for advanced semiconductor manufacturing.

The framework integrates:

* **Graph Neural Networks (GNNs)** to encode spatial and structural relationships at thin-film interfaces
* **Physics-Informed Neural Networks (PINNs)** to enforce governing diffusion physics through **Fick’s Second Law**, Arrhenius temperature dependence, and boundary constraints
* **Inverse optimization** to identify ALD process recipes that minimize interdiffusion while preserving growth-per-cycle (GPC) and film uniformity

The objective is to bridge the long-standing **R&D-to-fab disconnect** by enabling **physics-consistent surrogate modeling** that supports rapid process exploration, optimization, and future digital-twin deployment.

This repository accompanies the thesis *Hybrid Graph Neural Network and Physics-Informed Neural Network for Predicting Interface Dynamics in Semiconductor Thin Film Deposition*.

---

## Materials and Process Context

### Atomic Layer Deposition (ALD)

ALD was selected as the target deposition process due to its **atomic-scale thickness control**, excellent conformality in high-aspect-ratio structures, and central role in **advanced logic and memory fabrication**. Despite these advantages, ALD processes remain highly sensitive to **interface diffusion**, particularly at elevated temperatures, which can degrade electrical performance and long-term device reliability.

These interface-limited effects are difficult to probe experimentally at scale, motivating the need for **physics-aware surrogate models** capable of operating across realistic process regimes.

### Aluminum Oxide (Al₂O₃ on Si)

This study focuses on **Al₂O₃ deposited on silicon substrates** for several pragmatic and scientific reasons:

* Extensive **open-literature availability** on diffusion behavior, growth kinetics, and interface stability
* Well-characterized **diffusivity ranges, activation energies, and growth-per-cycle behavior**, enabling physics-constrained modeling
* Common use as a **baseline high-k dielectric** in both academic research and industrial process development

The choice of Al₂O₃ is **intentional and strategic**. It provides a well-understood material system that enables rigorous development, validation, and interpretation of physics-informed learning without reliance on proprietary fab data.

Importantly, the hybrid GNN-PINN framework itself is **not material-specific**; Al₂O₃ serves as a reference system for proof-of-concept development.

---

## 📁 Repository Contents (Modifiable)

```text
├── data/                 # Synthetic and/or real-world process datasets
├── notebooks/            # EDA, training, evaluation, and ablation studies
├── models/               # Baseline and hybrid model checkpoints
├── src/                  # Core GNN, PINN, and training logic
├── optimization/         # Bayesian optimization (TPE) workflows
├── figures/              # Generated plots and diagrams
├── docs/                 # Supplementary documentation and references
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Methodology

### 1. Physics-Constrained Synthetic Dataset Creation

A **10,000-point synthetic ALD dataset** was generated to approximate realistic process behavior while preserving physical plausibility.

* **Process parameter ranges** (e.g., temperature, pressure, pulse duration) were constrained using **published ALD research and practical process targets**
* Sampling was performed via **Latin Hypercube Sampling (LHS)** to ensure uniform coverage of the multidimensional design space
* **Target variables** (interdiffusion width, diffusivity, GPC, uniformity) were generated from **empirical and semi-empirical relationships reported in the literature**
* **Controlled stochastic noise** was added to targets to reflect experimental variability and sensor uncertainty while maintaining physical consistency

This approach balances realism with tractability while avoiding dependence on proprietary manufacturing data.

---

### 2. Baseline Model Training

Before introducing physics constraints, two **purely data-driven baseline models** were trained on the same dataset, including:

* Artificial Neural Networks (ANNs)
* Gradient-boosted decision trees (XGBoost)

These baselines establish performance references and highlight the limitations of conventional machine-learning approaches, particularly with respect to extrapolation and physical consistency.

---

### 3. Physics-Informed Pretraining (PINN Phase)

The GNN encoder was pretrained using **physics-only loss terms**, enforcing diffusion behavior through:

* **Fick’s Second Law** residual minimization
* **Arrhenius temperature dependence** for diffusivity
* **Dirichlet boundary conditions**, fixing concentration values at material interfaces and domain boundaries

This phase ensures that the learned latent representations remain **physically meaningful**, even in the absence of labeled concentration profiles.

---

### 4. Hybrid GNN-PINN Surrogate Training

The physics-pretrained GNN encoder was coupled with an MLP decoder to jointly predict:

* Interdiffusion width
* Diffusivity
* Growth-per-cycle (GPC)
* Film uniformity

Training combines **data-driven loss terms** with **physics residual loss**, yielding a surrogate that respects both empirical trends and governing physical laws.

---

### 5. Inverse Process Optimization

The trained surrogate was embedded in a **Bayesian optimization loop** to identify ALD process recipes that minimize interdiffusion while satisfying manufacturability constraints.

* Optimization employed a **Tree-Structured Parzen Estimator (TPE)**
* Enabled rapid exploration of the design space
* Achieved orders-of-magnitude speedup relative to conventional fab-based DOE

This demonstrates the surrogate’s viability as a **proto-digital-twin** for process optimization.

---

## Results Highlights

* Sub-0.50 nm interdiffusion widths achieved within physically plausible diffusivity limits
* Competitive GPC and film uniformity maintained across optimized recipes
* Seconds-level optimization runtime on a single CPU
* Improved extrapolation stability relative to baseline ANN and XGBoost models

---

## Synthetic vs. Real Data Strategy

This work adopts a **synthetic-first, physics-constrained data strategy** as a deliberate and enabling design choice rather than a limitation.

### Role of Synthetic Data

Synthetic data enables:

* Full and uniform coverage of the multidimensional ALD process space without prohibitive experimental cost
* Explicit enforcement of governing physical laws during model training
* Controlled evaluation of extrapolation behavior and inverse optimization performance

For the Al₂O₃ ALD system examined here, the synthetic dataset was constructed using **published research, experimentally reported process windows, and physically motivated target relationships**, ensuring all samples represent **viable and manufacturable process conditions** rather than arbitrary parameter combinations.

Importantly, this approach is **repeatable and extensible**. For other ALD materials or process variants, **additional synthetic datasets can be generated directly from the corresponding literature and targeted experimental studies**, enabling rapid adaptation of the framework to new material systems without requiring large-scale proprietary data collection.

---

### Synthetic vs. Real Process Data Characteristics

The synthetic dataset used in this study represents **physically plausible process parameter possibilities**, not the statistical distribution of data produced by a specific manufacturing tool or fab.

As a result:

* The dataset does **not exhibit the normality, clustering, or tool-specific bias** commonly present in real process data
* Instead, it provides **broad, uniform exploration of feasible design space**, which is particularly valuable for optimization and extrapolation studies

While predictive performance on this dataset was strong, it is expected that **integration of real-world process data would further improve model accuracy**, robustness, and calibration by introducing:

* Realistic parameter correlations
* Measurement noise characteristics
* Tool- and recipe-specific operating regimes

---

### Intended Data Progression

The intended progression of the framework is:

1. **Physics-constrained synthetic training** to establish generalizable, physically grounded representations
2. **Hybrid training** combining synthetic data with limited experimental or fab data
3. **Domain-adapted surrogate refinement** tailored to specific tools, materials, and production environments

This staged approach enables practical deployment while respecting the realities of data availability and confidentiality in semiconductor manufacturing.

---

### Why This Strategy Matters

By separating **physical feasibility** from **statistical process realization**, the framework avoids overfitting to narrow operating regimes and remains adaptable to future materials, tools, and fabs.

This strategy positions the hybrid GNN-PINN surrogate as a **general-purpose interface modeling and optimization framework**, although a customized model could be tied to a single manufacturing process.

---

## Real-World Data and Future Extensions

### Extension to Other High-k Dielectrics

Future work should extend the framework to additional high-k dielectric materials, including:

* Hafnium dioxide (HfO₂)
* Zirconium dioxide (ZrO₂)
* Lanthanum-based and mixed-oxide dielectrics

These materials introduce more complex diffusion mechanisms, phase stability concerns, and defect-mediated transport, making them ideal candidates for validating the generality of the GNN-PINN approach once sufficient data is available.

---

## Integration of Real-World Process Data

The framework is structured to incorporate measured process and metrology data, including:

* Expanded ALD parameter sets (purge durations, precursor flow rates, plasma power)
* In-situ signals (optical emission spectroscopy, ellipsometry)
* Ex-situ characterization (SIMS, TEM-derived interface profiles)

Such integration supports transition from offline optimization toward deployable digital-twin systems.

---

## Citation

If you use or reference this work, please cite:

> Crawford, R., Ostroumov, R., & Homrighouse, D. (2025). *Hybrid Graph Neural Network and Physics-Informed Neural Network for Predicting Interface Dynamics in Semiconductor Thin Film Deposition*. National University.

---

## License

This project is released for **academic and research use**.
Licensing terms may be updated pending publication.

---

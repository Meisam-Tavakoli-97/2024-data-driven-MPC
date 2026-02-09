[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Meisam%20Tavakoli-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/meisam-tavakoli)
[![GitHub](https://img.shields.io/badge/GitHub-Meisam--Tavakoli--97-181717?style=flat&logo=github&logoColor=white)](https://github.com/Meisam-Tavakoli-97)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=aAzQLBoAAAAJ&hl=en)



# Data-Driven Model Predictive Control with Stability and Robustness Guarantees

This repository contains the implementation of a data-driven Model Predictive Control (MPC) framework based on behavioral systems theory. The project reproduces the schemes proposed by Berberich et al., enabling the control of multivariable systems directly from measured input-output trajectories without an explicit parametric model.


## Installation

[cite_start]The code is developed in **Python**[cite: 83]. Install the required numerical and optimization libraries via:

```bash
pip install numpy scipy matplotlib casadi

Repository Structure

## The core components of the implementation include:

    - `Main.py: The entry point for running closed-loop simulations.

    - `Data_Driven_MPC.py: Implementation of the nominal and robust MPC optimization problems.

    - `Hankel_matrix.py: Routines for constructing block Hankel matrices from historical data.

    - `four_tanks.py: The benchmark four-tank system model and discretization logic.

    - `History_Data.py: Data collection using PRBS signals for persistent excitation.

    - `system_id.py: Utilities for data-driven system representation and rank condition validation.

## Running Experiments

```bash
python Main.py
```

## Contact

🧑‍💻 Meisam Tavakoli

📧 meisam.tavakoli@studio.unibo.it

# Machine Learning Project: Car Insurance Risk Estimation

## Overview
This project was developed as part of a **machine learning course**, in collaboration with two fellow classmates.  
The objective is to estimate **risk for car insurance claims** using supervised learning techniques.

---

## Problem Statement
The task is to model and predict the **number of insurance claims** associated with individual car insurance policies based on policyholder and vehicle characteristics.  
The dataset is highly imbalanced, with the majority of policies resulting in zero claims.

---

## Methodology
We approach the problem using **regression-based machine learning models**, allowing us to compare model behavior and performance under different inductive biases.

### Models Implemented
- **M1:** Decision Tree Regressor  
  - Custom implementation (`CustomDecisionTreeRegressor`)
- **M2:** Feedforward Neural Network (FFNN)
- **M3:** Random Forest Regressor

Each model is trained and evaluated using cross-validation and a held-out test set.

---

## Dataset
- Approximately **680,000 observations**
- **11 features** per observation
- Each row corresponds to a **single insurance policy**
- Target variable: **number of claims**

Preprocessing and feature engineering are applied prior to model training.

---

## Reproducibility

To replicate the results:

1. Fork or clone the repository
2. Ensure you have **Python 3.x** installed
3. Jupyter Notebook (used for running experiments)
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

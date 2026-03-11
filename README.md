# Recurrent Neural Networks for Disease Progression Prediction  
## A Reframed Static Diabetes Prediction Comparison on the Pima Dataset

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-Pima%20Diabetes-lightgrey.svg)

## Project Overview

This project began as a proposal on recurrent neural networks for disease progression prediction. The original goal was to model longitudinal clinical patterns over time using recurrent neural architectures such as RNNs or LSTMs. However, after dataset inspection, the final implementation was reframed to honestly match the available data.

The Pima Diabetes dataset is a static tabular classification dataset rather than a true longitudinal dataset with repeated patient visits. Because of that, this project became a controlled comparison of three model families for static diabetes prediction:

- Random Forest
- Gradient Boosting
- LSTM adapted for static tabular input

The central question became:

**Does a more complex recurrent neural architecture outperform strong classical machine learning models on the Pima diabetes dataset?**

## Final Conclusion

No. On this dataset, the classical machine learning models outperformed the adapted LSTM.

- **Random Forest** achieved the strongest overall performance
- **Gradient Boosting** achieved the strongest ROC AUC
- **LSTM** performed worst across all major metrics

This result supports an important machine learning principle:

**Higher model complexity does not automatically lead to better performance, especially on small structured tabular datasets.**

---

## Dataset

**Dataset:** Pima Diabetes Dataset  
**Rows:** 768  
**Target column:** `Class`

### Features
- Times Pregnant
- Blood Glucose
- Blood Pressure
- Skin Fold Thickness
- 2-Hour Insulin
- BMI
- Family History
- Age

### Prediction Task
Binary classification:

- `0` = No diabetes
- `1` = Diabetes

---

## Data Cleaning

Several features contained zero values that are medically implausible in real patients and were treated as missing values.

### Zero values replaced and imputed
- Blood Glucose: 5
- Blood Pressure: 35
- Skin Fold Thickness: 227
- 2-Hour Insulin: 374
- BMI: 11

These values were replaced with missing values and imputed using the **median**.

This step was important for making the dataset more clinically reasonable before training.

---

## Models Evaluated

### 1. Random Forest
A tree-based ensemble model used as a strong classical baseline.

### 2. Gradient Boosting
A boosting-based ensemble model used as a second classical baseline.

### 3. LSTM
A recurrent neural network adapted to the static feature vector by reshaping each patient record into a short ordered sequence of features.

> Note: Because the Pima dataset is not longitudinal, this LSTM does **not** model true disease progression over repeated visits. It is included as an architectural comparison only.

---

## Final Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|------|---------:|----------:|-------:|---:|--------:|
| Random Forest | 0.7727 | 0.7111 | 0.5926 | 0.6465 | 0.8115 |
| Gradient Boosting | 0.7468 | 0.6471 | 0.6111 | 0.6286 | 0.8215 |
| LSTM | 0.7078 | 0.5918 | 0.5370 | 0.5631 | 0.7696 |

### Model Ranking
1. **Random Forest** — best overall average rank  
2. **Gradient Boosting** — very close second, best ROC AUC  
3. **LSTM** — lowest across all metrics  

---

## Key Insight

The original project proposal was motivated by recurrent deep learning for disease progression modeling. After adapting the project to the available dataset, the empirical results showed that:

- the classical ensemble methods remained stronger
- the LSTM did not provide a predictive advantage
- model choice should be driven by data structure, not just model sophistication

This makes the project stronger scientifically because it tests the assumption instead of forcing the original idea onto the wrong dataset.

---

## Repository Structure

```text
Recurrent-Neural-Networks-for-Disease-Progression-Prediction/
│
├── data/
│   ├── raw/
│   │   └── Pima_Diabetes.csv
│   └── processed/
│       └── Pima_Diabetes_clean.csv
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│   ├── config.py
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train_random_forest.py
│   │   ├── train_gradient_boosting.py
│   │   └── train_lstm.py
│   ├── evaluation/
│   │   └── evaluate_models.py
│   └── utils/
│       └── seed.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── tables/
│
├── presentation/
├── docs/
│   └── screenshots/
├── README.md
└── requirements.txt

```

## Visual Outputs 
exploratory data analysis plots
class distribution chart
feature histograms
boxplots by class
correlation heatmap
confusion matrices
ROC curves
feature importance plots
final model comparison charts
model ranking summary tables

## How to run the project 
1. **clone the repo** 
```bash 
git clone https://github.com/galaxygab121/Recurrent-Neural-Networks-for-Disease-Progression-Prediction.git
cd Recurrent-Neural-Networks-for-Disease-Progression-Prediction
```  
2. **Create and activate the environment** 
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. **install dependencies** 
```bash 
pip install -r requirements.txt
```
4. **train models** 
```bash 
python -m src.models.train_random_forest
python -m src.models.train_gradient_boosting
python -m src.models.train_lstm
```
5. **generate final comparison outputs** 
```bash 
python -m src.evaluation.evaluate_models
```

## Main Outputs 
### 1. Metrics
outputs/metrics/random_forest_metrics.json
outputs/metrics/gradient_boosting_metrics.json
outputs/metrics/lstm_metrics.json

### 2. Saved Models
outputs/models/random_forest_pipeline.pkl
outputs/models/gradient_boosting_pipeline.pkl
outputs/models/lstm_model.pt
outputs/models/lstm_scaler.pkl

### 3. Final Tables
outputs/tables/final_model_comparison.csv
outputs/tables/final_model_comparison_rounded.csv
outputs/tables/model_rank_summary.csv

## Limitations 
This project has several important limitations:
The dataset is not longitudinal, so true disease progression over repeated visits could not be modeled.
The LSTM architecture was adapted to static tabular data and should not be interpreted as a true sequential medical forecasting model.
The dataset is relatively small, which favors simpler tabular methods.
External validation was not performed on a second independent diabetes dataset.

## Future Work 
If extended, this project could be improved by:
using a real longitudinal EHR or diabetes follow-up dataset
testing XGBoost or LightGBM
performing hyperparameter tuning with cross-validation
calibrating probabilities
evaluating fairness and subgroup performance
building a small interactive dashboard for model exploration

## WHY THIS PROJECT MATTERS 
This project demonstrates something valuable beyond raw accuracy:
it shows the importance of aligning model architecture with dataset structure.
Rather than forcing a deep learning model to fit a non-sequential problem, this project honestly tested whether recurrence added value. The result was a stronger and more credible machine learning study.

## Author 
Gabrielle Boyer-Baker 
Computer Science Major and Neuroscience minor at DePaul University 

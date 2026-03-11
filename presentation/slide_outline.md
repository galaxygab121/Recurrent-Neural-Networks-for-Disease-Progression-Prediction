# 20-Slide Presentation Outline
## Recurrent Neural Networks for Disease Progression Prediction
### A Reframed Static Diabetes Prediction Comparison on the Pima Dataset

## Slide 1 — Title Slide
- Project title
- Your name
- Course name
- Date
- DePaul University
- Visual: diabetes/clinical ML themed title background

## Slide 2 — Project Motivation
- Why disease prediction matters
- Why machine learning is useful in healthcare
- Original inspiration from recurrent models and temporal clinical patterns

## Slide 3 — Original Proposal vs Final Implementation
- Original proposal: longitudinal disease progression with RNN/LSTM
- Reality: Pima dataset is static, not longitudinal
- Final project became an honest comparison study

## Slide 4 — Research Question
- Can an LSTM style deep learning model outperform strong classical machine learning models on the Pima diabetes dataset?

## Slide 5 — Dataset Overview
- Pima Diabetes dataset
- 768 rows
- 8 predictive features
- binary target class
- Visual: dataset summary table

## Slide 6 — Feature List
- Times Pregnant
- Blood Glucose
- Blood Pressure
- Skin Fold Thickness
- 2-Hour Insulin
- BMI
- Family History
- Age

## Slide 7 — Data Cleaning Strategy
- medically implausible zeros
- replaced with missing values
- median imputation
- Visual: zero-value summary counts

## Slide 8 — Exploratory Data Analysis
- class balance
- feature distributions
- Visual: class_distribution.png

## Slide 9 — Feature Distributions
- highlight one or two important histograms
- Visuals: selected histogram PNGs

## Slide 10 — Boxplots by Diabetes Class
- show feature separation between class 0 and class 1
- Visuals: selected boxplot PNGs

## Slide 11 — Correlation Structure
- discuss relationships between variables
- Visual: correlation_heatmap.png

## Slide 12 — Models Evaluated
- Random Forest
- Gradient Boosting
- LSTM adapted for static input
- explain why each was included

## Slide 13 — Random Forest Results
- metrics
- feature importance
- Visuals: random_forest_feature_importance.png and confusion matrix

## Slide 14 — Gradient Boosting Results
- metrics
- feature importance
- Visuals: gradient_boosting_feature_importance.png and confusion matrix

## Slide 15 — LSTM Results
- metrics
- training loss
- confusion matrix / ROC
- explain adaptation of static features into short sequences

## Slide 16 — Final Comparison Table
- show all metrics side by side
- Visual: final_model_comparison_rounded.csv converted into slide table

## Slide 17 — Final Comparison Charts
- Visuals:
  - comparison_accuracy.png
  - comparison_f1.png
  - comparison_roc_auc.png
  - overall_model_scorecard.png

## Slide 18 — Key Findings
- Random Forest strongest overall
- Gradient Boosting best ROC AUC
- LSTM underperformed
- complexity did not improve performance

## Slide 19 — Limitations and Future Work
- static dataset
- not true disease progression
- need real longitudinal EHR data
- future improvements

## Slide 20 — Final Conclusion
- the importance of matching model to data
- classical models can outperform deep learning on tabular data
- closing takeaway and thank you
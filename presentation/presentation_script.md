# Presentation Script

## Slide 1 — Title Slide
Today I will present my project, Recurrent Neural Networks for Disease Progression Prediction, which evolved into an honest comparison of machine learning models for static diabetes prediction using the Pima dataset.

## Slide 2 — Project Motivation
Diabetes prediction is an important healthcare problem because earlier identification of risk can support better intervention, monitoring, and long term health outcomes. My original interest was in whether recurrent neural networks could model clinical progression patterns.

## Slide 3 — Original Proposal vs Final Implementation
My original proposal assumed a longitudinal clinical dataset with repeated measurements over time. After examining the Pima dataset more closely, I found that it is actually static tabular data. Instead of forcing the original plan, I reframed the project honestly to compare whether an LSTM style model could still outperform classical machine learning baselines.

## Slide 4 — Research Question
The final research question became whether an LSTM adapted for static tabular input could outperform Random Forest and Gradient Boosting on the Pima diabetes dataset.

## Slide 5 — Dataset Overview
The dataset contains 768 patient records and eight predictive clinical features, with a binary class label indicating diabetes outcome.

## Slide 6 — Feature List
These eight features include pregnancy count, blood glucose, blood pressure, skin fold thickness, insulin, BMI, family history, and age. Together they provide a compact set of structured health indicators.

## Slide 7 — Data Cleaning Strategy
A key preprocessing step involved identifying medically implausible zero values in several clinical fields such as glucose, blood pressure, insulin, and BMI. These were treated as missing values and imputed using the median.

## Slide 8 — Exploratory Data Analysis
Exploratory analysis helped reveal the class balance of the dataset and the general shape of each feature distribution before model training.

## Slide 9 — Feature Distributions
These histograms show how feature values are distributed after cleaning, helping us understand scale, skew, and potential outliers.

## Slide 10 — Boxplots by Diabetes Class
These boxplots compare feature distributions between patients with and without diabetes and highlight which variables appear most separable.

## Slide 11 — Correlation Structure
The correlation heatmap provides a quick overview of linear relationships between features and helps identify which variables move together.

## Slide 12 — Models Evaluated
I evaluated three models: Random Forest, Gradient Boosting, and an LSTM adapted for static tabular input. The first two served as strong classical baselines, while the LSTM tested whether a recurrent architecture added value.

## Slide 13 — Random Forest Results
Random Forest produced the strongest overall performance in the project, achieving the best accuracy and F1 score among the three models.

## Slide 14 — Gradient Boosting Results
Gradient Boosting performed very competitively and achieved the best ROC AUC, making it a strong second place model overall.

## Slide 15 — LSTM Results
The LSTM model required reshaping each patient’s feature vector into a short sequence, since the dataset did not contain true time steps. While this allowed architectural comparison, the LSTM underperformed both classical models.

## Slide 16 — Final Comparison Table
This table summarizes all major metrics across the three models and shows that the tree based approaches remained strongest overall.

## Slide 17 — Final Comparison Charts
These summary charts make the comparison easier to interpret visually across accuracy, F1, ROC AUC, and overall performance trends.

## Slide 18 — Key Findings
The main finding is that increasing model complexity did not improve predictive performance on this dataset. Classical ensemble methods remained the best fit.

## Slide 19 — Limitations and Future Work
The largest limitation is that the dataset is static and not longitudinal, so true disease progression modeling was not possible. Future work should use real sequential clinical data.

## Slide 20 — Final Conclusion
In conclusion, this project showed that model architecture must match the structure of the data. On small structured tabular medical data, classical machine learning can outperform deep learning, and testing that assumption directly made the study stronger.
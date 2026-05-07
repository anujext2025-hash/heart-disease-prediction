# Heart Disease Prediction 

## Overview
End-to-end ML project predicting heart disease using 
Logistic Regression and Random Forest classifier.
Dataset: 303 patients, 14 medical features.

## Results
| Metric    | Logistic Regression | Random Forest |
|-----------|--------------------:|--------------|
| Accuracy  | 86.9%               | 86.9%        |
| Precision | 87.5%               | 85.3%        |
| Recall    | 87.5%               | 90.6%        |
| F1 Score  | 87.5%               | 87.9%        |
| ROC-AUC   | 0.914               | 0.867        |

## Key Findings

# Finding 1 — Chest Pain & Blocked Vessels
Atypical angina (chest pain type 1) showed the highest 
disease rate at 82%, surprisingly outperforming typical 
angina. Confirmed by correlation analysis where chest pain 
had the strongest positive correlation with target.
The number of blocked vessels (ca) was the most important 
feature, directly reflecting the physical cause of heart disease.

# Finding 2 — Cholesterol Paradox
Contrary to expectation, patients with normal cholesterol 
(200-240) showed a higher disease rate than high cholesterol 
groups. This likely reflects statin medication use among 
high-cholesterol patients — a known phenomenon called the 
cholesterol paradox. This highlights why cholesterol alone 
is insufficient as a diagnostic marker.

# Finding 3 — Model Selection & Medical Impact
Despite identical accuracy (86.9%), Random Forest was 
selected for deployment over Logistic Regression because 
of superior Recall (0.906 vs 0.875) — missing only 3 
actual disease cases vs 4. In healthcare, Recall always 
takes priority — a missed diagnosis could be fatal.
5-fold cross validation confirmed both models are stable 
at ~81-82% mean accuracy.

## Feature Importance
1. ca (blocked vessels) → direct physical evidence
2. oldpeak (ST depression) → cardiac stress signal  
3. cp (chest pain type) → primary symptom

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Models
- Logistic Regression
- Random Forest (deployed)
- 5-fold Cross Validation


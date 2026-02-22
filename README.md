# Stroke Prediction - ML Reproduction

This repository reproduces the modelling approach reported in **Akinwumi et al. (2025)**: "Evaluating machine learning models for stroke prediction based on clinical variables" (*Frontiers in Neurology*).

## Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Dataset

**Source:** Kaggle Stroke Prediction Dataset

- Rows: 5,110
- Input features: 11
- Target: `stroke` (0 = No, 1 = Yes)
- Class balance: ~4.9% positive (imbalanced)

Expected feature groups:

- Demographic: `age`, `gender`, `ever_married`, `work_type`, `Residence_type`
- Clinical: `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`
- Behavioural: `smoking_status`

Place the CSV at:

- `data\`

## Pipeline

- Missing data: mean imputation for `bmi`
- Encoding: binary label encoding + one-hot for categoricals
- Scaling: min-max to [0, 1]
- Imbalance handling: random over-sampling applied to training folds only
- Split: 80/20 train-test with stratification
- Cross-validation: 5-fold stratified
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

## Results comparison

| Model | Paper Accuracy | Reproduced Accuracy | Paper ROC-AUC | Reproduced ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 95.11% | TBD | 0.836 | TBD |
| Gradient Boosting | 95.11% | TBD | 0.824 | TBD |
| Random Forest | ~95% | TBD | 0.792 | TBD |
| SVM | ~95% | TBD | 0.615 | TBD |
| KNN | ~95% | TBD | 0.614 | TBD |

Top 3 predictors reported in the paper:

1. Age
2. Average glucose level
3. BMI

## Repository structure

stroke-prediction-reproduction/
  README.md
  requirements.txt
  .gitignore
  data/
    stroke_data.csv
  src/
    preprocessing.py
    models.py
    evaluation.py
  results/
    model_comparison.csv
    confusion_matrices/
    roc_curves.png
    feature_importance.png

## Setup (Windows CMD)

cmd
git clone <repo-url>
cd stroke-prediction-reproduction

py -m venv .venv
.venv\Scripts\activate

py -m pip install --upgrade pip
pip install -r requirements.txt


## References

Akinwumi, P. O., Ojo, S., Nathaniel, T. I., Wanliss, J., Karunwi, O., & Sulaiman, M. (2025).
Evaluating machine learning models for stroke prediction based on clinical variables.
*Frontiers in Neurology, 16*. https://doi.org/10.3389/fneur.2025.1668420

## Author

Joshua Ajemiri

- BSc Psychology with Cognitive Neuroscience, University of Leicester
- LinkedIn: https://www.linkedin.com/in/joshuaajemiri-7794682a4/
- GitHub: https://github.com/Shortjosh619

License: MIT

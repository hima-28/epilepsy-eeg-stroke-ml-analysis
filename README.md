# 🧠 Healthcare Machine Learning Analysis
## Epilepsy EEG Classification & Stroke Prediction

This repository contains the analysis for **Individual Task 1 – Part 1.3: Data Analysis**. The project applies and compares two supervised machine learning algorithms — **Support Vector Machine (SVM)** and **Multi-Layer Perceptron (MLP)** — across two healthcare datasets:

1. **BEED (Bangalore EEG Epilepsy Dataset)** — EEG-related features for epilepsy classification.
2. **Stroke Prediction Dataset** — demographic and clinical features for stroke prediction.

The analysis demonstrates a complete machine learning workflow including data inspection, train-test splitting, missing-value treatment, categorical encoding, feature scaling, model training, and evaluation.

> **Note:** This README documents the code and results contained in the uploaded project archive. It does not replace the original assessment instructions or rubric.

---

## 📌 Project Overview

The main objective is to investigate how different machine learning algorithms perform on healthcare datasets with different characteristics.

The project compares:

| Dataset | Problem Type | Models |
|---|---|---|
| BEED EEG Dataset | Multi-class classification | SVM, MLP |
| Stroke Prediction Dataset | Binary classification | SVM, MLP |

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

A fixed random seed of `42` is used to make the train-test split and model training reproducible.

---

# 📂 Repository Structure

The uploaded project contains the following files:

```text
CaseStudiesIndividual Task 1 Part 1/
│
├── beed_+bangalore+eeg+epilepsy+dataset/
│   └── BEED_Data.csv
│
├── healthcare-dataset-stroke-data.csv
│
└── part1.3-analysis.ipynb
```

### File descriptions

| File | Description |
|---|---|
| `part1.3-analysis.ipynb` | Main Jupyter Notebook containing the complete analysis |
| `BEED_Data.csv` | Bangalore EEG Epilepsy Dataset |
| `healthcare-dataset-stroke-data.csv` | Stroke prediction dataset |

---

# 🧰 Technologies & Libraries

The notebook uses Python and the following major libraries:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder
)

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.impute import SimpleImputer
```

### Main technologies

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical computation
- **Scikit-learn** — preprocessing, machine learning and evaluation
- **Jupyter Notebook** — analysis and presentation

---

# 📊 Dataset 1 — BEED Epilepsy EEG Dataset

The first dataset is the **BEED (Bangalore EEG Epilepsy Dataset)**.

It contains EEG-related numerical features that are used to classify epilepsy-related patterns.

## Dataset dimensions

The dataset contains:

```text
Rows:       8,000
Columns:    17
Features:   16
Target:      1
```

The columns are:

```text
X1, X2, X3, X4, X5, X6, X7, X8,
X9, X10, X11, X12, X13, X14, X15, X16, y
```

The target variable is:

```text
y
```

The target contains four classes:

```text
0
1
2
3
```

Each class contains 2,000 observations, making the BEED dataset balanced.

### Loading the dataset

```python
BEED_PATH = "beed_+bangalore+eeg+epilepsy+dataset/BEED_Data.csv"

beed = pd.read_csv(BEED_PATH)

print("Shape:", beed.shape)
print(list(beed.columns))
display(beed.head())
```

---

# 🔎 BEED Data Preparation

## Target and feature separation

The target column is identified as `y`.

```python
label_col = "y" if "y" in beed.columns else beed.columns[-1]

X_beed = beed.drop(columns=[label_col])
y_beed = beed[label_col]

X_beed = X_beed.select_dtypes(
    include=[np.number]
)
```

The resulting feature matrix contains:

```text
8,000 observations
16 numerical features
```

### Target distribution

```text
Class 0    2000
Class 1    2000
Class 2    2000
Class 3    2000
```

---

# ✂️ Train-Test Split — BEED

The data are split into:

- **80% training data**
- **20% testing data**

Stratification is used to preserve the class distribution.

```python
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_beed,
    y_beed,
    test_size=0.2,
    random_state=42,
    stratify=y_beed
)
```

Result:

```text
Training observations: 6400
Testing observations:  1600
```

---

# 🧹 Missing Value Handling — BEED

Numerical missing values are handled using mean imputation.

Importantly, the imputer is fitted only on the training data.

```python
imputer_beed = SimpleImputer(
    strategy="mean"
)

X_train_b = pd.DataFrame(
    imputer_beed.fit_transform(X_train_b),
    columns=X_beed.columns
)

X_test_b = pd.DataFrame(
    imputer_beed.transform(X_test_b),
    columns=X_beed.columns
)
```

This approach avoids using information from the test set during preprocessing.

---

# 📏 Feature Scaling — BEED

Standardisation is applied because both SVM and MLP are sensitive to differences in feature scale.

```python
scaler_beed = StandardScaler()

X_train_b_scaled = scaler_beed.fit_transform(
    X_train_b
)

X_test_b_scaled = scaler_beed.transform(
    X_test_b
)
```

Result:

```text
Training: 6400 × 16
Testing:  1600 × 16
```

---

# 🤖 Model 1 — SVM on BEED

An SVM with an **RBF kernel** is used.

```python
svm_beed = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm_beed.fit(
    X_train_b_scaled,
    y_train_b
)

pred_svm_b = svm_beed.predict(
    X_test_b_scaled
)

proba_svm_b = svm_beed.predict_proba(
    X_test_b_scaled
)
```

The RBF kernel allows the classifier to model nonlinear decision boundaries.

---

# 🧠 Model 2 — MLP on BEED

A Multi-Layer Perceptron neural network is also trained.

Architecture:

```text
Input Layer
     ↓
64 neurons
     ↓
32 neurons
     ↓
Output Layer
```

Implementation:

```python
mlp_beed = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    early_stopping=True,
    random_state=42
)

mlp_beed.fit(
    X_train_b_scaled,
    y_train_b
)

pred_mlp_b = mlp_beed.predict(
    X_test_b_scaled
)

proba_mlp_b = mlp_beed.predict_proba(
    X_test_b_scaled
)
```

`early_stopping=True` helps reduce unnecessary training and can help control overfitting.

---

# 📊 BEED Results

The trained models produced the following test-set results:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.7444 | 0.7731 | 0.7444 | 0.7310 | 0.9272 |
| MLP | **0.9419** | **0.9428** | **0.9419** | **0.9422** | **0.9939** |

### Interpretation

The **MLP substantially outperformed SVM on the BEED dataset**.

The MLP achieved:

```text
Accuracy : 94.19%
F1 Score : 94.22%
ROC-AUC  : 99.39%
```

This indicates that the nonlinear neural network was able to capture the relationships between the EEG-related features and the four target classes more effectively than the SVM configuration used in this analysis.

### MLP confusion matrix

```text
[[398   0   0   2]
 [  0 381   6  13]
 [  0   1 367  32]
 [  0   5  34 361]]
```

The model performed particularly strongly on class `0`, while classes `2` and `3` showed more confusion.

---

# 🩺 Dataset 2 — Stroke Prediction Dataset

The second dataset contains demographic and clinical characteristics associated with stroke.

It contains:

```text
Rows:       5,110
Columns:    12
Predictors: 10 after removing ID
Target:      stroke
```

Columns:

```text
id
gender
age
hypertension
heart_disease
ever_married
work_type
Residence_type
avg_glucose_level
bmi
smoking_status
stroke
```

### Loading the dataset

```python
STROKE_PATH = "healthcare-dataset-stroke-data.csv"

stroke = pd.read_csv(STROKE_PATH)

print("Shape:", stroke.shape)
print(list(stroke.columns))
display(stroke.head())
```

---

# 🔎 Stroke Dataset Inspection

The dataset contains one important missing-value issue:

```text
bmi: 201 missing values
```

The target is highly imbalanced:

```text
No Stroke (0): 4861
Stroke (1):     249
```

Target proportions:

```text
No Stroke: 95.13%
Stroke:     4.87%
```

This imbalance is important when interpreting model performance.

---

# 🧹 Removing the ID Column

The `id` column is removed because it is an identifier rather than a meaningful predictive feature.

```python
if "id" in stroke.columns:
    stroke = stroke.drop(columns=["id"])

y_stroke = stroke["stroke"]

X_stroke = stroke.drop(
    columns=["stroke"]
)
```

---

# 🏷️ Feature Classification

The predictor variables are divided into three groups.

### Binary categorical variables

```python
binary_cols = [
    "gender",
    "ever_married",
    "Residence_type"
]
```

### Nominal categorical variables

```python
nominal_cols = [
    "work_type",
    "smoking_status"
]
```

### Numerical variables

```python
numeric_cols = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]
```

---

# ✂️ Train-Test Split — Stroke

An 80/20 stratified split is used.

```python
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_stroke,
    y_stroke,
    test_size=0.2,
    random_state=42,
    stratify=y_stroke
)
```

Result:

```text
Training observations: 4088
Testing observations:  1022
```

Stratification is especially important because only around 4.9% of observations belong to the stroke class.

---

# 🔢 Binary Encoding

Binary categorical variables are encoded using `LabelEncoder`.

```python
for col in binary_cols:

    le = LabelEncoder()

    le.fit(
        X_train_s[col].astype(str)
    )

    X_train_s[col] = le.transform(
        X_train_s[col].astype(str)
    )

    known = set(le.classes_)

    X_test_s[col] = (
        X_test_s[col]
        .astype(str)
        .apply(
            lambda v:
            v if v in known
            else le.classes_[0]
        )
    )

    X_test_s[col] = le.transform(
        X_test_s[col]
    )
```

The encoder is fitted using training data only.

---

# 🔠 One-Hot Encoding

Nominal categorical variables are one-hot encoded.

```python
ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

ohe.fit(
    X_train_s[nominal_cols]
)

ohe_train = pd.DataFrame(
    ohe.transform(
        X_train_s[nominal_cols]
    ),
    columns=ohe.get_feature_names_out(
        nominal_cols
    ),
    index=X_train_s.index
)

ohe_test = pd.DataFrame(
    ohe.transform(
        X_test_s[nominal_cols]
    ),
    columns=ohe.get_feature_names_out(
        nominal_cols
    ),
    index=X_test_s.index
)
```

`handle_unknown="ignore"` prevents unseen test-set categories from causing encoding errors.

After encoding:

```text
Training features: 17
Testing features:  17
```

---

# 🧹 Missing BMI Values

The `bmi` feature contains 201 missing observations.

Mean imputation is applied:

```python
imputer_stroke = SimpleImputer(
    strategy="mean"
)

X_train_s[numeric_cols] = (
    imputer_stroke.fit_transform(
        X_train_s[numeric_cols]
    )
)

X_test_s[numeric_cols] = (
    imputer_stroke.transform(
        X_test_s[numeric_cols]
    )
)
```

The imputer is fitted only on training data.

---

# 📏 Feature Scaling — Stroke

All predictor variables are standardised.

```python
scaler_stroke = StandardScaler()

X_train_s_scaled = (
    scaler_stroke.fit_transform(
        X_train_s
    )
)

X_test_s_scaled = (
    scaler_stroke.transform(
        X_test_s
    )
)
```

---

# 🤖 SVM — Stroke Prediction

Because the stroke dataset is highly imbalanced, the SVM uses:

```python
class_weight="balanced"
```

This gives greater importance to the minority stroke class.

```python
svm_stroke = SVC(
    kernel="rbf",
    probability=True,
    random_state=42,
    class_weight="balanced"
)

svm_stroke.fit(
    X_train_s_scaled,
    y_train_s
)

pred_svm_s = svm_stroke.predict(
    X_test_s_scaled
)

proba_svm_s = svm_stroke.predict_proba(
    X_test_s_scaled
)
```

---

# 🧠 MLP — Stroke Prediction

The same neural network architecture is applied:

```python
mlp_stroke = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    early_stopping=True,
    random_state=42
)

mlp_stroke.fit(
    X_train_s_scaled,
    y_train_s
)

pred_mlp_s = mlp_stroke.predict(
    X_test_s_scaled
)

proba_mlp_s = mlp_stroke.predict_proba(
    X_test_s_scaled
)
```

---

# 📊 Stroke Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.7720 | 0.9327 | 0.7720 | 0.8345 | **0.7956** |
| MLP | **0.9511** | 0.9045 | **0.9511** | **0.9272** | 0.5416 |

At first glance, the MLP appears to have much higher accuracy. However, the confusion matrix reveals an important problem.

---

# ⚠️ Important Stroke Model Finding

The MLP confusion matrix is:

```text
[[972   0]
 [ 50   0]]
```

This means the MLP predicted **every test observation as non-stroke**.

Therefore:

```text
Stroke recall = 0.00
Stroke F1     = 0.00
```

Although the MLP has:

```text
Accuracy = 95.11%
```

this accuracy is misleading because the dataset is highly imbalanced.

The model fails to identify the minority stroke class.

### MLP classification report

```text
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       972
           1       0.00      0.00      0.00        50
```

This demonstrates why **accuracy should not be used as the only evaluation metric in imbalanced healthcare classification**.

---

# 🩺 SVM Stroke Model Analysis

The SVM produced:

```text
Confusion Matrix:

[[759 213]
 [ 20  30]]
```

The SVM identified:

```text
30 / 50 stroke cases
```

giving the positive class a recall of:

```text
60%
```

The complete metrics were:

```text
Accuracy : 0.7720
Precision: 0.9327
Recall   : 0.7720
F1 Score : 0.8345
ROC-AUC  : 0.7956
```

Although its overall accuracy is lower than the MLP, the SVM provides meaningful detection of the minority stroke class.

For a healthcare-oriented use case, this behaviour can be more valuable than simply maximising overall accuracy.

---

# 📈 Model Comparison

The notebook creates a combined results DataFrame:

```python
results = []

models = [
    (
        "SVM",
        "BEED",
        y_test_b,
        pred_svm_b,
        proba_svm_b
    ),
    (
        "MLP",
        "BEED",
        y_test_b,
        pred_mlp_b,
        proba_mlp_b
    ),
    (
        "SVM",
        "Stroke",
        y_test_s,
        pred_svm_s,
        proba_svm_s
    ),
    (
        "MLP",
        "Stroke",
        y_test_s,
        pred_mlp_s,
        proba_mlp_s
    )
]
```

The resulting comparison is:

```text
  Dataset  Model  Accuracy  Precision  Recall  F1 Score  ROC-AUC
0    BEED    SVM    0.7444     0.7731  0.7444    0.7310   0.9272
1    BEED    MLP    0.9419     0.9428  0.9419    0.9422   0.9939
2  Stroke    SVM    0.7720     0.9327  0.7720    0.8345   0.7956
3  Stroke    MLP    0.9511     0.9045  0.9511    0.9272   0.5416
```

---

# 📊 Evaluation Metrics

## Accuracy

Accuracy measures the proportion of correctly classified observations:

```text
Accuracy = Correct Predictions / Total Predictions
```

Accuracy is useful when classes are reasonably balanced.

---

## Precision

Precision measures how many predicted positive/classified cases are actually correct.

```text
Precision = TP / (TP + FP)
```

High precision means fewer false-positive predictions.

---

## Recall

Recall measures how many actual positive cases are correctly identified.

```text
Recall = TP / (TP + FN)
```

Recall is particularly important in healthcare applications where missing a genuine condition can have serious consequences.

---

## F1 Score

F1-score combines precision and recall:

```text
F1 = 2 × (Precision × Recall)
     ---------------------------
       Precision + Recall
```

It is useful when both false positives and false negatives matter.

---

## ROC-AUC

ROC-AUC measures the model's ability to discriminate between classes based on predicted probabilities.

A value closer to:

```text
1.0
```

generally indicates stronger discrimination.

For the multi-class BEED dataset, one-vs-rest ROC-AUC is used.

---

# 🔍 Confusion Matrices

The notebook generates confusion matrices for all four model-dataset combinations.

```python
print("SVM - BEED")
print(confusion_matrix(y_test_b, pred_svm_b))

print("\nMLP - BEED")
print(confusion_matrix(y_test_b, pred_mlp_b))

print("\nSVM - Stroke")
print(confusion_matrix(y_test_s, pred_svm_s))

print("\nMLP - Stroke")
print(confusion_matrix(y_test_s, pred_mlp_s))
```

Confusion matrices are especially important in healthcare because they reveal the types of errors that overall accuracy can hide.

---

# 🧪 Complete Evaluation Function

The notebook uses a reusable evaluation function:

```python
def evaluate_model(name, y_test, y_pred, y_proba=None):

    print(f"\n--- {name} ---")

    print(
        "Accuracy :",
        round(
            accuracy_score(y_test, y_pred),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall   :",
        round(
            recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1 Score :",
        round(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        )
    )

    if y_proba is not None:

        try:

            if (
                y_proba.ndim == 2
                and y_proba.shape[1] == 2
            ):
                auc = roc_auc_score(
                    y_test,
                    y_proba[:, 1]
                )

            else:
                auc = roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr"
                )

            print(
                "ROC-AUC  :",
                round(auc, 4)
            )

        except Exception as e:

            print(
                "ROC-AUC: could not compute",
                e
            )

    print(
        "Confusion Matrix:\n",
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print(
        "\nFull classification report:\n",
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )
```

---

# 🔄 End-to-End Workflow

The complete analysis follows this workflow:

```text
                    ┌─────────────────────┐
                    │    Load Dataset     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Inspect Data &      │
                    │ Target Distribution │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train-Test Split    │
                    │     80% / 20%       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Preprocessing   │
                    │ Imputation/Encoding │
                    │     / Scaling       │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌─────────────┐       ┌─────────────┐
             │     SVM     │       │     MLP     │
             └──────┬──────┘       └──────┬──────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Accuracy / F1 / AUC │
                    │ Confusion Matrix    │
                    └─────────────────────┘
```

---

# 🔒 Data Leakage Prevention

A major strength of the analysis is that preprocessing is performed **after the train-test split**.

The general principle is:

```text
Raw Data
   ↓
Train-Test Split
   ↓
Training Data ──→ Fit Imputer / Encoder / Scaler
   │
   └─────────────→ Transform Training Data
   │
Test Data ───────→ Transform Using Training Parameters
```

The test data are therefore not used to learn preprocessing parameters.

This helps prevent data leakage and provides a more realistic estimate of model performance.

---

# ⚠️ Important Interpretation

The results should be interpreted carefully, particularly for the Stroke Prediction Dataset.

The MLP achieves:

```text
95.11% accuracy
```

but predicts no stroke cases.

Therefore, it is **not appropriate to conclude that the MLP is the better stroke model simply because its accuracy is higher**.

The SVM has lower overall accuracy but identifies a meaningful proportion of actual stroke cases:

```text
Stroke recall = 60%
```

For a healthcare screening scenario, recall and class-specific metrics can be more important than overall accuracy.

---

# 🏆 Main Findings

## BEED Dataset

The MLP performs substantially better than the SVM.

```text
MLP ROC-AUC: 0.9939
MLP F1:      0.9422
MLP Accuracy:0.9419
```

The MLP appears to capture the nonlinear patterns in the EEG features effectively.

---

## Stroke Dataset

The results demonstrate the challenge of severe class imbalance.

### MLP

```text
Accuracy: 95.11%
Stroke Recall: 0%
ROC-AUC: 0.5416
```

### SVM

```text
Accuracy: 77.20%
Stroke Recall: 60%
ROC-AUC: 0.7956
```

The SVM is more useful for identifying the minority stroke class in the current experiment.

---

# 🧠 Overall Conclusion

This analysis demonstrates that **model performance depends strongly on the characteristics of the dataset and the evaluation objective**.

For the BEED epilepsy dataset:

> **MLP is the stronger model in the current experiment**, achieving approximately 94% accuracy and 99.39% ROC-AUC.

For the Stroke Prediction dataset:

> **The SVM provides more meaningful minority-class detection**, despite having lower overall accuracy than the MLP.

The analysis also demonstrates why healthcare machine learning should not rely on accuracy alone, especially when the target classes are highly imbalanced.

---

# ⚙️ Installation

## 1. Clone or extract the project

```bash
git clone <your-repository-url>
cd "CaseStudiesIndividual Task 1 Part 1"
```

If the project is downloaded as a ZIP file, extract it while preserving the following structure:

```text
CaseStudiesIndividual Task 1 Part 1/
├── beed_+bangalore+eeg+epilepsy+dataset/
│   └── BEED_Data.csv
├── healthcare-dataset-stroke-data.csv
└── part1.3-analysis.ipynb
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install pandas numpy scikit-learn jupyter
```

If you maintain a `requirements.txt`, the core dependencies are:

```text
pandas
numpy
scikit-learn
jupyter
```

---

# ▶️ Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
part1.3-analysis.ipynb
```

Run the notebook from top to bottom.

The dataset paths are already configured for the extracted project structure:

```python
BEED_PATH = "beed_+bangalore+eeg+epilepsy+dataset/BEED_Data.csv"

STROKE_PATH = "healthcare-dataset-stroke-data.csv"
```

---

# 📋 Reproducibility

The analysis uses:

```python
RANDOM_STATE = 42
```

The same random state is used for train-test splitting and model configuration where supported.

This helps make results reproducible when the same software environment and library versions are used.

---

# 📚 Project Outputs

Running the notebook produces:

- Dataset dimensions and column information
- Target distributions
- Train/test split information
- Missing-value checks
- Encoded feature dimensions
- Scaled feature dimensions
- SVM predictions
- MLP predictions
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices
- Classification reports
- Final model comparison table

---

# ⚠️ Limitations

This analysis is intended for **educational and research purposes**.

The models should not be interpreted as clinical diagnostic systems.

Important limitations include:

- No external validation dataset
- No hyperparameter optimisation
- No cross-validation comparison
- Severe class imbalance in the Stroke dataset
- The MLP stroke model demonstrates poor minority-class detection
- Dataset characteristics may not represent real-world clinical populations
- Model outputs should not be used for medical diagnosis

For a production healthcare system, additional validation, calibration, interpretability, fairness analysis, clinical review, and regulatory considerations would be required.

---

# 🚀 Possible Future Improvements

Future versions could improve the analysis by adding:

### Model tuning

```python
from sklearn.model_selection import GridSearchCV
```

Use grid search or random search to optimise:

- SVM `C`
- SVM `gamma`
- MLP hidden-layer sizes
- Learning rate
- Regularisation

### Cross-validation

```python
from sklearn.model_selection import StratifiedKFold
```

Evaluate model stability across multiple folds.

### Better imbalance handling

Potential approaches include:

```text
Class weighting
SMOTE
Random oversampling
Random undersampling
Threshold optimisation
Precision-recall analysis
```

### Additional models

Possible comparisons:

```text
Logistic Regression
Random Forest
XGBoost
Gradient Boosting
KNN
Decision Tree
```

### More healthcare-focused metrics

For the stroke task, consider:

```text
Sensitivity / Recall
Specificity
Precision-Recall AUC
Balanced Accuracy
Matthews Correlation Coefficient
```

---

# 📜 Academic Integrity & Data Usage

The datasets included in this project should be used according to their respective dataset licences and terms of use.

This repository contains an academic machine learning analysis and should not be interpreted as a clinical decision-support system.

---

# 👤 Author

**Himasri Madala**

**Student ID:** S4222141  
**Program:** Master of Data Science  
**University:** RMIT University

---

# 📄 Project Summary

```text
Healthcare Machine Learning
        │
        ├── BEED EEG Dataset
        │       │
        │       ├── 8,000 observations
        │       ├── 16 numerical features
        │       ├── 4 classes
        │       │
        │       ├── SVM → 74.44% accuracy
        │       └── MLP → 94.19% accuracy
        │
        └── Stroke Dataset
                │
                ├── 5,110 observations
                ├── Clinical + demographic features
                ├── Highly imbalanced target
                │
                ├── SVM → 77.20% accuracy
                │          60% stroke recall
                │
                └── MLP → 95.11% accuracy
                           0% stroke recall
```

**Key takeaway:** The best model should be selected based on the **problem, class distribution, and evaluation objective**, not accuracy alone.

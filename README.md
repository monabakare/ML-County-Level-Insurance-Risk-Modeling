# ML-County-Level-Insurance-Risk-Modeling

# County-Level Insurance Risk Modeling

This repository contains a machine learning project focused on modeling county-level uninsured health insurance risk in the United States using socioeconomic indicators from Data Commons. The project combines regression, classification, clustering, feature engineering, and interpretability techniques to analyze structural drivers of insurance vulnerability.

The project was originally developed as part of Georgia Tech CS 4641 and later refactored into a modular machine learning pipeline emphasizing reproducibility, centralized evaluation, and model interpretability.

---

## Project Goal

The primary goal of this project is to identify and analyze county-level uninsured risk using socioeconomic and demographic indicators.

The project addresses three core tasks:

1. **Regression**
   - Predict continuous county-level uninsured rates.

2. **Classification**
   - Identify whether counties belong to a high-risk uninsured category.

3. **Clustering**
   - Discover structural groupings of counties with similar socioeconomic profiles.

The broader objective is to better understand which factors contribute most strongly to insurance vulnerability and whether machine learning can effectively identify high-risk counties for policy-oriented analysis.

---

## Repository Structure

```text
ML-County-Level-Insurance-Risk-Modeling/
├── README.md
├── index.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   └── figures/
│       ├── Elbow_method.png
│       ├── PCA.png
│       ├── elasticnet_cv_metrics.png
│       ├── logistic_confusion_matrix.png
│       ├── random_forest_cv_metrics.png
│       └── rf_classifier_confusion_matrix.png
│
└── src/
    ├── preprocess.py
    ├── metrics.py
    ├── interpretability.py
    │
    └── models/
        ├── elastic_net.py
        ├── logistic_regression.py
        ├── random_forest.py
        ├── rf_classifier.py
        └── kmeans.py
```
## Project Documentation

The primary project report and analysis are located in:

```text
index.md
```
The report includes:

- Background and motivation
- Problem definition
- Data processing pipeline
- Model implementation details
- Evaluation methodology
- Regression, classification, and clustering results
- Model interpretation
- Limitations

---

## Core Components

### `src/preprocess.py`

Centralized preprocessing pipeline used across all models.

Responsibilities include:

- Data fetching from Data Commons
- Feature engineering
- Missing value imputation
- Scaling and encoding
- Train/test splitting
- Leakage prevention
- Construction of reusable preprocessing pipelines

Engineered features include:

- Log-transformed income
- Log-transformed population density
- Bachelor degree rate
- Economic distress index

---

### `src/models/`

Contains standalone model implementations.

#### Regression Models

- `elastic_net.py`
- `random_forest.py`

#### Classification Models

- `logistic_regression.py`
- `rf_classifier.py`

#### Clustering

- `kmeans.py`

Each model is implemented using reusable preprocessing pipelines and standardized training workflows.

---

### `src/metrics.py`

Centralized evaluation and visualization pipeline.

This script performs:

- Holdout evaluation
- 5-fold cross-validation
- Regression metrics
- Classification metrics
- Clustering evaluation
- Visualization generation
- Confusion matrices
- PCA and elbow-method plots

Generated figures are automatically saved into:

```text
docs/figures/
```

---

### `src/interpretability.py`

Provides model explainability and feature analysis.

This script extracts:

- ElasticNet coefficients
- Logistic Regression coefficients
- Random Forest feature importances
- Random Forest Classifier feature importances

The interpretability pipeline helps identify which socioeconomic variables most strongly contribute to uninsured risk.

---

## Dataset

This project uses county-level socioeconomic and demographic data from Data Commons, including:

- Population
- Uninsured households
- Median household income
- Unemployment rate
- Poverty rate
- Median age
- Population density
- Education statistics

Source:

Data Commons  
https://datacommons.org/

---

## Modeling Approach

### Regression

Used to predict continuous uninsured rates.

#### Models

- ElasticNetCV
- Random Forest Regressor

#### Metrics

- MAE
- RMSE
- R²

---

### Classification

Used to classify counties as high-risk or lower-risk based on uninsured rate thresholds.

#### Models

- Logistic Regression
- Random Forest Classifier

#### Metrics

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

---

### Clustering

Used to identify structural county groupings.

#### Model

- KMeans Clustering

#### Metrics

- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score

#### Visualization

- PCA projection
- Elbow method analysis

## Key Findings
Classification models substantially outperformed regression models overall.
Logistic Regression achieved the strongest classification performance with cross-validation ROC-AUC approaching 0.90.
Random Forest models improved nonlinear regression performance and captured more structural variation than ElasticNet.
Median household income and unemployment rate consistently emerged as the strongest predictors of uninsured risk.
Clustering analysis revealed meaningful structural county groupings based on income, unemployment, and population characteristics.

## Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd ML-County-Level-Insurance-Risk-Modeling
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended: create a virtual environment.

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Project

### Run Full Evaluation Pipeline

```bash
python3 -m src.metrics
```

### Run Specific Evaluation Sections

#### Regression Only

```bash
RUN_SECTION=regression python3 -m src.metrics
```

#### Classification Only

```bash
RUN_SECTION=classification python3 -m src.metrics
```

#### Clustering Only

```bash
RUN_SECTION=clustering python3 -m src.metrics
```

---

## Run Interpretability Analysis

```bash
python3 -m src.interpretability
```

Run specific interpretability sections:

### ElasticNet

```bash
SECTION=elasticnet python3 -m src.interpretability
```

### Random Forest Regressor

```bash
SECTION=rf_regressor python3 -m src.interpretability
```

### Logistic Regression

```bash
SECTION=logistic python3 -m src.interpretability
```

### Random Forest Classifier

```bash
SECTION=rf_classifier python3 -m src.interpretability
```

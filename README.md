# Credit Risk Classification using Neural Networks

This project focuses on assessing credit risk using the German Credit dataset (`german.data-numeric`). We build and evaluate an Artificial Neural Network (ANN) model to classify customers as good or bad credit risks. The ANN is compared against traditional machine learning classifiers including Random Forest, Logistic Regression, and XGBoost.

## 📁 Project Structure

nnfl/

├── output/ # Folder containing generated plots and figures

├── ann.py # Script for training ANN

├── ann_model.h5 # Saved ANN model

├── comparison.py # Classifier comparison with traditional models

├── evaluation.py # Model evaluation metrics

├── german.data # Raw dataset (original format)

├── german.data-numeric # Preprocessed numeric version of dataset

├── german_credit_data.csv # CSV-formatted version of dataset

├── main.py # Main execution file

├── model.py # ANN model architecture

├── preprocessing.py # Preprocessing pipeline (normalization, SMOTE, etc.)

├── visualization.py # Code for visualizing EDA and evaluation results


## 🔍 Dataset

- **Source**: UCI German Credit Dataset
- **Type**: Binary classification (Good Credit = 1, Bad Credit = 0)
- **Format**: Numeric features (`german.data-numeric`)

## ⚙️ Workflow

### 1. Exploratory Data Analysis (EDA)
- Class imbalance and distribution inspection
- Visualizations for key features

### 2. Data Preprocessing
- Missing value handling
- Feature normalization
- Class balancing using **SMOTE**

### 3. Model Training
- Feedforward ANN with:
  - Input layer
  - One or more hidden layers
  - Output layer with sigmoid activation

### 4. Benchmarking
- Comparative classifiers:
  - Random Forest
  - Logistic Regression
  - XGBoost

### 5. Evaluation
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC Curve (visualized)
- Confusion matrix plots
- Score summaries

## 📊 Results

- ANN showed performance comparable to ensemble methods while better capturing non-linear interactions.
- ROC and F1-score indicated ANN’s strong suitability for imbalanced classification.

## 📌 Requirements

- Python 3.8+
- `scikit-learn`
- `tensorflow` / `keras`
- `xgboost`
- `pandas`, `matplotlib`, `seaborn`, `imblearn`

## 🚀 Usage

```bash
# Run full training and evaluation
python main.py

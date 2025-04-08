# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import seaborn as sns
#%matplotlib inline
import matplotlib.pyplot as plt

# Step 2: Load Framingham dataset (replace 'framingham.csv' with your file path)
data = pd.read_csv('framingham.csv')

# Quick exploration of data
print(data.head())
print(data.info())
print(data['TenYearCHD'].value_counts())  # Target variable distribution

# Step 3: Handle missing values (mean/median imputation for simplicity)
data.fillna(data.median(), inplace=True)

# Check again for missing values
print("Missing values after imputation:\n", data.isnull().sum())

# Step 4: Feature Engineering and Preprocessing
X = data.drop("TenYearCHD", axis=1)
y = data["TenYearCHD"]

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split data into training and test sets (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Function to evaluate models quickly
def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_te)
    probas = model.predict_proba(X_te)[:, 1]
    print(classification_report(y_te, predictions))
    print("ROC-AUC:", roc_auc_score(y_te, probas))
    cm = confusion_matrix(y_te, predictions)
    plt.figure(figsize=(8, 6)) # check later
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()  # check later
    plt.show()

# Step 6: Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# Step 7: Experiment groups clearly defined
datasets = {
    "Original (Imbalanced)": (X_train, y_train),
    "SMOTE": SMOTE(random_state=42).fit_resample(X_train, y_train),
    "ADASYN": ADASYN(random_state=42).fit_resample(X_train, y_train)
}

# Step 8: Train and evaluate each classifier on each dataset
for dataset_name, (X_resampled, y_resampled) in datasets.items():
    print(f"\n{'='*30}\nDataset: {dataset_name}\n{'='*30}")
    for model_name, model in models.items():
        print(f"\n--- Model: {model_name} ---")
        evaluate_model(model, X_resampled, y_resampled, X_test, y_test)
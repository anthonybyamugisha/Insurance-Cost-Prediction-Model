import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import shap

# Load data
df = pd.read_csv('insurance_data.csv')
print(f"Original shape: {df.shape}")

# Remove duplicates
df.drop_duplicates(inplace=True)
print(f"After duplicate removal: {df.shape}")

# Encoding
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})

# Statistics
print(f"\n=== DATASET STATISTICS ===")
print(f"Average cost: ${df['charges'].mean():,.2f}")
print(f"Median cost: ${df['charges'].median():,.2f}")
print(f"Min cost: ${df['charges'].min():,.2f}")
print(f"Max cost: ${df['charges'].max():,.2f}")
print(f"Male %: {(df['sex'] == 0).mean()*100:.1f}%")
print(f"Female %: {(df['sex'] == 1).mean()*100:.1f}%")
print(f"Non-smoker %: {(df['smoker'] == 0).mean()*100:.1f}%")
print(f"Smoker %: {(df['smoker'] == 1).mean()*100:.1f}%")

# Feature selection
df_final = df.drop(['sex', 'region'], axis=1)
Xf = df_final.drop(['charges'], axis=1)
Yf = df_final[['charges']]

# Split
xtrain_f, xtest_f, ytrain_f, ytest_f = train_test_split(Xf, Yf, test_size=0.2, random_state=42)

# Train final model
finalmodel = XGBRegressor(n_estimators=15, max_depth=3, gamma=0)
finalmodel.fit(xtrain_f, ytrain_f)

# SHAP Analysis
explainer = shap.Explainer(finalmodel, xtrain_f)
shap_values = explainer(xtest_f)

# Get SHAP importance
shap_importance = shap_values.abs.mean(0).values
total_importance = shap_importance.sum()
shap_percentages = (shap_importance / total_importance) * 100

print(f"\n=== SHAP FEATURE IMPORTANCE ===")
for i, feature in enumerate(xtest_f.columns):
    print(f"{feature}: {shap_percentages[i]:.1f}%")

print(f"\n=== TRAINING/TEST SET SIZES ===")
print(f"Training set: {xtrain_f.shape[0]} samples")
print(f"Test set: {xtest_f.shape[0]} samples")

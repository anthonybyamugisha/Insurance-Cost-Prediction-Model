# 🏥 Insurance Cost Prediction Model

A comprehensive machine learning web application that predicts medical insurance costs using XGBoost with **90.2% accuracy**, evaluated against 5 other ML/DL models and validated with SHAP interpretability analysis.

## 📊 Project Overview

This application predicts insurance premiums based on personal health factors using an XGBoost regression model trained on **1,337 insurance records**. The project evaluates **6 machine learning and deep learning approaches**, with XGBoost selected as the champion model. **SHAP analysis** validates model decisions and ensures transparency.

## ✨ Features

- 🎯 **Real-time Predictions**: Get instant insurance cost estimates
- 📊 **Interactive Visualizations**: Explore data through beautiful charts
- 🎨 **Modern UI**: Professional, card-based interface with smooth animations
- 🏆 **90.2% Accuracy**: Best among 6 models evaluated (ML + DL)
- 🧠 **SHAP Interpretability**: Transparent, explainable AI decisions
- 🤖 **6 Models Compared**: LR, SVR, RF, GB, XGB, Deep Neural Network
- 📈 **Data Insights**: Key findings and statistical analysis

## 🚀 Deployment on Streamlit Cloud

### Prerequisites

Make sure you have the following files in your GitHub repository:
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `insurance_data.csv` - Dataset
- `insurancemodelf.pkl` - Trained model

### Steps to Deploy

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Insurance Cost Prediction App"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set the main file path to `app.py`
   - Click "Deploy!"

3. **Verify Dependencies**
   - Streamlit Cloud will automatically install packages from `requirements.txt`
   - Check the deployment logs for any errors

## 🛠️ Local Development

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd INSURANCE

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Requirements

- Python 3.8+
- streamlit==1.31.0
- pandas==2.1.4
- numpy==1.26.3
- matplotlib==3.8.2
- seaborn==0.13.1
- scikit-learn==1.3.2
- xgboost==2.0.3
- feature-engine==1.6.1

## 📁 Project Structure

```
INSURANCE/
├── app.py                      # Main Streamlit application
├── insurance_cost.ipynb        # Complete ML pipeline (6 models + SHAP)
├── insurance_data.csv          # Dataset (1,337 records after deduplication)
├── insurancemodelf.pkl         # Trained XGBoost champion model
├── requirements.txt            # Python dependencies
├── PROJECT_REPORT.md           # Comprehensive technical report
├── README.md                   # This file
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── .gitignore                 # Git ignore rules
```

## 🎯 Model Performance

### Comprehensive Model Comparison

| Model | Train R² | Test R² | CV Score | Status |
|-------|----------|---------|----------|--------|
| Linear Regression | 0.730 | 0.806 | 0.747 | ❌ Underfitting |
| Support Vector Regressor | -0.102 | -0.134 | -0.104 | ❌ Failed |
| Random Forest | 0.975 | 0.882 | 0.837 | ✅ Strong |
| Gradient Boosting | 0.868 | 0.902 | 0.861 | ✅ Very Strong |
| **XGBoost** | **0.869** | **0.902** | **0.861** | 🏆 **Champion** |
| Deep Neural Network | 0.747 | 0.816 | N/A | ⚠️ Overfitting |

### Champion Model Details

- **Algorithm**: XGBoost Regressor
- **Hyperparameters**: 
  - n_estimators: 15
  - max_depth: 3
  - gamma: 0
- **Features Used**: 4 (age, bmi, children, smoker)
- **Features Removed**: sex, region (minimal predictive power)

## 🧠 SHAP Feature Importance

| Feature | SHAP Importance | Impact |
|---------|----------------|--------|
| **Smoker** | 57.7% | Smokers pay ~3× more than non-smokers |
| **Age** | 22.5% | Older individuals have higher premiums |
| **BMI** | 14.5% | Higher BMI increases costs |
| **Children** | 5.3% | Minimal impact on premiums |

**Note**: SHAP analysis validates that the model makes fair decisions without gender or regional bias.

## 📈 Key Insights

1. **Smoking status** is the dominant predictor (57.7% SHAP importance) - smokers pay ~3× more
2. **Age** shows strong positive correlation (22.5% importance) - older = higher costs
3. **BMI** has moderate impact (14.5% importance) - obesity increases health risks
4. **Number of children** has minimal influence (5.3% importance)
5. **Gender and region** removed during feature selection - no predictive power
6. **XGBoost (0.902 R²)** outperformed Deep Neural Network (0.816 R²)
7. **SHAP analysis** confirms model fairness - no demographic bias detected

## 🔧 Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: XGBoost, Scikit-learn
- **Deep Learning**: TensorFlow, Keras
- **Interpretability**: SHAP (SHapley Additive exPlanations)
- **Visualization**: Matplotlib, Seaborn
- **Feature Engineering**: feature-engine

## 📊 Dataset

- **Source**: [Hugging Face - Insurance Dataset](https://huggingface.co/datasets/adegoke655/Insurance)
- **Original Records**: 1,338
- **After Deduplication**: 1,337 unique records
- **Features**: 7 (age, sex, bmi, children, smoker, region, charges)
- **Final Features Used**: 4 (age, bmi, children, smoker)
- **Target**: charges (annual insurance cost in USD)

## 📝 License

This project is created for portfolio demonstration and educational purposes.

## 🙏 Acknowledgments

- Dataset provided via Hugging Face
- Built with Streamlit framework
- Powered by XGBoost machine learning library
- Deep Learning implemented with TensorFlow/Keras
- Model interpretability with SHAP

---

**Live Demo**: https://insurance-cost-prediction-model.streamlit.app/  
**GitHub Repository**: https://github.com/anthonybyamugisha/Insurance-Cost-Prediction-Model

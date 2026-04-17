# 🏥 Insurance Cost Prediction Model

A machine learning-powered web application that predicts medical insurance costs using XGBoost with 90.4% accuracy.

## 📊 Project Overview

This application predicts insurance premiums based on personal health and demographic factors using an advanced XGBoost regression model trained on 1,339 insurance records.

## ✨ Features

- 🎯 **Real-time Predictions**: Get instant insurance cost estimates
- 📊 **Interactive Visualizations**: Explore data through beautiful charts
- 🎨 **Modern UI**: Professional, card-based interface with smooth animations
- 📈 **90.4% Accuracy**: High-precision ML model
- 🔍 **Data Insights**: Key findings and statistical analysis

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
├── insurance_cost.ipynb        # Jupyter notebook with ML pipeline
├── insurance_data.csv          # Dataset (1,339 records)
├── insurancemodelf.pkl         # Trained XGBoost model
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Training R² | 0.870 |
| Testing R² | 0.904 |
| Cross-Validation | 0.860 |

### Model Details

- **Algorithm**: XGBoost Regressor
- **Hyperparameters**: 
  - n_estimators: 15
  - max_depth: 3
  - gamma: 0

## 📈 Key Insights

1. **Smoking status** is the most significant predictor (smokers pay ~3x more)
2. **Age** shows strong positive correlation with premiums
3. **BMI** has moderate impact on costs
4. **Gender and region** have minimal predictive power

## 🔧 Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: XGBoost, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Feature Engineering**: feature-engine

## 📊 Dataset

- **Source**: [Hugging Face - Insurance Dataset](https://huggingface.co/datasets/adegoke655/Insurance)
- **Records**: 1,339
- **Features**: 6 (age, sex, bmi, children, smoker, region)
- **Target**: charges (insurance cost)

## 👨‍💻 Author

**Anthony Byamugisha**

- WorldQuant University - Data Analytics
- Makerere University
- Ugandan Data Analyst & Machine Learning Enthusiast

## 📝 License

This project is created for portfolio demonstration and educational purposes.

## 🙏 Acknowledgments

- Dataset provided via Hugging Face
- Built with Streamlit framework
- Powered by XGBoost machine learning library

---

**Live Demo**: [Your Streamlit App URL]

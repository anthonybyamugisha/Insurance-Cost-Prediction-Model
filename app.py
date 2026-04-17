import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import load

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('insurancemodelf.pkl', 'rb') as file:
        model = load(file)
    return model

# Load data for EDA
@st.cache_data
def load_data():
    df = pd.read_csv("insurance_data.csv")
    return df

# Main application
def main():
    # Header
    st.markdown('<p class="main-header">🏥 Insurance Cost Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">Machine Learning-Powered Insurance Premium Estimation</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Data Analysis", "🔮 Predict Costs", "ℹ️ About"])
    
    # Load model and data
    model = load_model()
    df = load_data()
    
    # Home Page
    if page == "🏠 Home":
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Size", f"{df.shape[0]} records")
        with col2:
            st.metric("Features", f"{df.shape[1]-1} variables")
        with col3:
            st.metric("Model Accuracy", "90.4%")
        
        st.markdown("---")
        st.markdown("## 🎯 Project Overview")
        st.write("""
        This application uses an advanced **XGBoost machine learning model** to predict insurance costs 
        based on personal health and demographic factors. The model has been trained on 1,339 insurance 
        records and achieves a **test accuracy of 90.4%**.
        """)
        
        st.markdown("### ✨ Key Features")
        features_col1, features_col2 = st.columns(2)
        with features_col1:
            st.write("• 📈 Real-time cost predictions")
            st.write("• 📊 Interactive data visualizations")
            st.write("• 🎯 90.4% prediction accuracy")
        with features_col2:
            st.write("• 🔍 Exploratory data analysis")
            st.write("• 💡 Key insights extraction")
            st.write("• ⚡ Fast and reliable predictions")
        
        st.markdown("---")
        st.markdown("### 📋 Features Used in Prediction")
        st.write("""
        - **Age**: Policyholder's age
        - **Sex**: Gender (male/female)
        - **BMI**: Body Mass Index
        - **Children**: Number of dependents
        - **Smoker**: Smoking status (yes/no)
        - **Region**: Geographic region
        """)
    
    # Data Analysis Page
    elif page == "📊 Data Analysis":
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
        
        # Dataset overview
        st.markdown("### Dataset Overview")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Distribution plots
        st.markdown("### Feature Distributions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            df['smoker'].value_counts().plot.pie(
                ax=ax1, 
                autopct='%1.1f%%',
                labels=['No', 'Yes'],
                colors=['#2ecc71', '#e74c3c']
            )
            ax1.set_ylabel('')
            ax1.set_title('Smoker Distribution')
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            df['sex'].value_counts().plot.pie(
                ax=ax2, 
                autopct='%1.1f%%',
                labels=['Male', 'Female'],
                colors=['#3498db', '#e91e63']
            )
            ax2.set_ylabel('')
            ax2.set_title('Gender Distribution')
            st.pyplot(fig2)
        
        with col3:
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            df['region'].value_counts().plot.pie(
                ax=ax3, 
                autopct='%1.1f%%',
                colors=['#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
            )
            ax3.set_ylabel('')
            ax3.set_title('Region Distribution')
            st.pyplot(fig3)
        
        st.markdown("---")
        
        # Average charges by category
        st.markdown("### Average Insurance Charges by Category")
        col1, col2 = st.columns(2)
        
        with col1:
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            avg_charges_smoker = df.groupby('smoker')['charges'].mean()
            avg_charges_smoker.plot(kind='bar', ax=ax4, color=['#2ecc71', '#e74c3c'])
            ax4.set_title('Average Charges by Smoker Status')
            ax4.set_xlabel('Smoker')
            ax4.set_ylabel('Average Charges ($)')
            ax4.set_xticklabels(['No', 'Yes'], rotation=0)
            st.pyplot(fig4)
        
        with col2:
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            avg_charges_sex = df.groupby('sex')['charges'].mean()
            avg_charges_sex.plot(kind='bar', ax=ax5, color=['#3498db', '#e91e63'])
            ax5.set_title('Average Charges by Gender')
            ax5.set_xlabel('Gender')
            ax5.set_ylabel('Average Charges ($)')
            ax5.set_xticklabels(['Male', 'Female'], rotation=0)
            st.pyplot(fig5)
        
        st.markdown("---")
        
        # Scatter plots
        st.markdown("### Relationship Between Features and Charges")
        col1, col2 = st.columns(2)
        
        with col1:
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='age', y='charges', hue='smoker', ax=ax6, 
                          palette=['#2ecc71', '#e74c3c'], alpha=0.6)
            ax6.set_title('Age vs Insurance Charges')
            ax6.set_xlabel('Age')
            ax6.set_ylabel('Charges ($)')
            st.pyplot(fig6)
        
        with col2:
            fig7, ax7 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', ax=ax7,
                          palette=['#2ecc71', '#e74c3c'], alpha=0.6)
            ax7.set_title('BMI vs Insurance Charges')
            ax7.set_xlabel('BMI')
            ax7.set_ylabel('Charges ($)')
            st.pyplot(fig7)
        
        st.markdown("---")
        
        # Key insights
        st.markdown("### 🔍 Key Insights")
        st.write("""
        1. **Smokers pay significantly higher premiums** - approximately 3x more than non-smokers
        2. **Age positively correlates with charges** - older individuals tend to pay more
        3. **BMI shows a moderate positive relationship** with insurance costs
        4. **Gender has minimal impact** on the final charges
        5. **Number of children doesn't strongly influence** the premium amount
        """)
    
    # Prediction Page
    elif page == "🔮 Predict Costs":
        st.markdown("---")
        st.markdown('<p class="sub-header">🔮 Insurance Cost Predictor</p>', unsafe_allow_html=True)
        st.write("Enter your details below to get an instant insurance cost estimate")
        
        st.markdown("---")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=18, max_value=64, value=30, step=1)
            sex = st.selectbox("Sex", ["male", "female"])
            bmi = st.number_input("BMI (Body Mass Index)", 
                                 min_value=15.0, 
                                 max_value=55.0, 
                                 value=25.0, 
                                 step=0.1)
        
        with col2:
            children = st.slider("Number of Children", min_value=0, max_value=5, value=0, step=1)
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northwest", "northeast", "southeast", "southwest"])
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🎯 Predict Insurance Cost", type="primary", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })
            
            # Encode categorical variables (matching the training data)
            input_data['sex'] = input_data['sex'].map({'male': 0, 'female': 1})
            input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})
            input_data['region'] = input_data['region'].map({
                'northwest': 0, 
                'northeast': 1,
                'southeast': 2,
                'southwest': 3
            })
            
            # Drop columns not used in final model (sex and region were dropped)
            input_data_final = input_data.drop(['sex', 'region'], axis=1)
            
            # Make prediction
            prediction = model.predict(input_data_final)[0]
            
            # Display result
            st.markdown("### 💰 Predicted Insurance Cost")
            st.markdown(f"""
                <div class="prediction-result">
                    ${prediction:,.2f}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Additional information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Age Factor:** {age} years old")
            with col2:
                st.info(f"**Smoker:** {smoker.capitalize()}")
            with col3:
                st.info(f"**BMI:** {bmi}")
            
            st.markdown("---")
            
            # Comparison with dataset average
            avg_charge = df['charges'].mean()
            if prediction > avg_charge:
                st.warning(f"⚠️ Your predicted cost is **${(prediction - avg_charge):,.2f}** above the average (${avg_charge:,.2f})")
            else:
                st.success(f"✅ Your predicted cost is **${(avg_charge - prediction):,.2f}** below the average (${avg_charge:,.2f})")
    
    # About Page
    elif page == "ℹ️ About":
        st.markdown("---")
        st.markdown('<p class="sub-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 Project Objective")
        st.write("""
        This project aims to predict medical insurance costs using machine learning techniques. 
        By analyzing various personal and health-related factors, the model provides accurate 
        cost estimations that can help both insurance providers and customers.
        """)
        
        st.markdown("### 🛠️ Technology Stack")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.write("**Data Processing:**")
            st.write("• Pandas")
            st.write("• NumPy")
            st.write("• Feature-engine")
        with tech_col2:
            st.write("**Machine Learning:**")
            st.write("• XGBoost")
            st.write("• Scikit-learn")
            st.write("• Streamlit")
        
        st.markdown("### 📊 Model Performance")
        st.write("""
        | Metric | Score |
        |--------|-------|
        | Training R² | 0.870 |
        | Testing R² | 0.904 |
        | Cross-Validation | 0.860 |
        """)
        
        st.markdown("### 🚀 Model Details")
        st.write("""
        **Algorithm:** XGBoost Regressor  
        **Optimized Hyperparameters:**
        - n_estimators: 15
        - max_depth: 3
        - gamma: 0
        
        **Data Preprocessing:**
        - Outlier treatment using IQR method (BMI capping)
        - Label encoding for categorical variables
        - Feature selection (sex and region removed for optimal performance)
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Key Findings")
        st.write("""
        1. **Smoking status** is the most significant predictor of insurance costs
        2. **Age** shows a strong positive correlation with premiums
        3. **BMI** has a moderate impact on cost predictions
        4. **Geographic region** and **gender** have minimal predictive power
        5. The model achieves **90.4% accuracy** on unseen test data
        """)
        
        st.markdown("---")
        st.markdown("**Dataset Source:** [Hugging Face - Insurance Dataset](https://huggingface.co/datasets/adegoke655/Insurance)")

if __name__ == "__main__":
    main()

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
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4a5568;
        margin: 1.5rem 0 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        box-shadow: 0 15px 35px rgba(245, 87, 108, 0.4);
        margin: 1.5rem 0;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #e2e8f0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        border-color: #667eea;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .insight-box {
        background: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-1r5gz7h {
        background: transparent;
    }
    
    .sidebar-content {
        color: white;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Page transition effects */
    .main .block-container {
        animation: fadeInPage 0.5s ease-in;
    }
    
    @keyframes fadeInPage {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Input styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div:focus {
        border-color: #667eea;
    }
    
    /* Metric styling */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
    # Header with icon
    st.markdown('<p class="main-header">🏥 Insurance Cost Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; font-size: 1.3rem; margin-bottom: 2rem;">Machine Learning-Powered Insurance Premium Estimation</p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown('---')
    
    # Sidebar navigation with enhanced styling
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🏥</div>
                <h2 style="color: white; margin: 0; font-weight: 700;">Insurance Predictor</h2>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">ML-Powered Predictions</p>
                <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">
            </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "📊 Data Analysis", "🔮 Predict Costs", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown('<hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">', unsafe_allow_html=True)
        
        st.markdown("""
            <div style="color: rgba(255,255,255,0.9); padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <h4 style="color: white; margin: 0 0 0.5rem 0;">📊 Quick Stats</h4>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">• Model: XGBoost</p>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">• R² Score: 0.901</p>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">• Features: 4</p>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">• Records: 1,338</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    df = load_data()
    
    # Home Page
    if page == "🏠 Home":
        st.markdown("---")
        # Enhanced stat cards with custom styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="stat-card">
                    <h2 style='margin: 0; font-size: 2.5rem;'>1,338</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>Dataset Records</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="stat-card">
                    <h2 style='margin: 0; font-size: 2.5rem;'>4</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>Input Features</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="stat-card">
                    <h2 style='margin: 0; font-size: 2.5rem;'>0.901</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>R² Test Score</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div class="info-box">
                <h3 style='margin: 0 0 1rem 0;'>🎯 Project Overview</h3>
                <p style='margin: 0; line-height: 1.8; font-size: 1.1rem;'>
                This application uses an <strong>XGBoost regression model</strong> to predict medical insurance costs 
                based on personal health and demographic factors. The model was trained on 1,338 insurance 
                records and achieves an <strong>R² score of 0.901 on the test set</strong>, meaning it explains 90.1% of the 
                variance in insurance charges.
                </p>
            </div>
        """, unsafe_allow_html=True)
                
        st.markdown('<p class="section-title">✨ Key Features</p>', unsafe_allow_html=True)
        features_col1, features_col2 = st.columns(2)
        with features_col1:
            st.markdown("""
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>📈 Real-time Predictions</h4>
                    <p style='margin: 0; color: #718096;'>Get instant insurance cost estimates</p>
                </div>
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>🎯 R² Score: 0.901</h4>
                    <p style='margin: 0; color: #718096;'>Strong predictive performance</p>
                </div>
            """, unsafe_allow_html=True)
        with features_col2:
            st.markdown("""
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>📊 Data Visualizations</h4>
                    <p style='margin: 0; color: #718096;'>Interactive charts and insights</p>
                </div>
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>⚡ Fast & Reliable</h4>
                    <p style='margin: 0; color: #718096;'>Optimized for performance</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-title">📋 Features Used in Prediction</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <p style='margin: 0.5rem 0;'><strong>🎂 Age:</strong> Policyholder's age in years</p>
                <p style='margin: 0.5rem 0;'><strong>📊 BMI:</strong> Body Mass Index (weight/height ratio)</p>
                <p style='margin: 0.5rem 0;'><strong>👶 Children:</strong> Number of dependents covered</p>
                <p style='margin: 0.5rem 0;'><strong>🚬 Smoker:</strong> Smoking status (yes/no)</p>
            </div>
            <div class="insight-box" style="margin-top: 1rem;">
                <strong>Note:</strong> The final model uses only 4 features (age, bmi, children, smoker). Features like sex and region were removed during feature selection as they had minimal impact on predictions.
            </div>
        """, unsafe_allow_html=True)
    
    # Data Analysis Page
    elif page == "📊 Data Analysis":
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
        
        # Dataset overview in a styled container
        st.markdown('<p class="section-title">📄 Dataset Statistics</p>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown('---')
        
        # Dataset info
        st.markdown('<p class="section-title">ℹ️ Dataset Information</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>📦 Dataset Size</h4>
                    <p style='margin: 0; color: #718096;'><strong>1,338</strong> records</p>
                    <p style='margin: 0.5rem 0 0 0; color: #718096;'><strong>7</strong> columns (6 features + 1 target)</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>🎯 Target Variable</h4>
                    <p style='margin: 0; color: #718096;'><strong>charges</strong> (insurance cost)</p>
                    <p style='margin: 0.5rem 0 0 0; color: #718096;'>Range: $1,121 - $63,770</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<p class="section-title">📊 Categorical Feature Distributions</p>', unsafe_allow_html=True)
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
        
        st.markdown('<p class="section-title">💰 Average Insurance Charges by Category</p>', unsafe_allow_html=True)
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
        
        st.markdown('<p class="section-title">🔗 Relationship Between Features and Charges</p>', unsafe_allow_html=True)
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
        
        # Key insights in styled boxes
        st.markdown('<p class="section-title">💡 Key Insights</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="insight-box">
                <strong>1. Smokers pay significantly higher premiums</strong> - approximately 3x more than non-smokers
            </div>
            <div class="insight-box">
                <strong>2. Age positively correlates with charges</strong> - older individuals tend to pay more
            </div>
            <div class="insight-box">
                <strong>3. BMI shows a moderate positive relationship</strong> with insurance costs
            </div>
            <div class="insight-box">
                <strong>4. Gender has minimal impact</strong> on the final charges
            </div>
            <div class="insight-box">
                <strong>5. Number of children doesn't strongly influence</strong> the premium amount
            </div>
        """, unsafe_allow_html=True)
    
    # Prediction Page
    elif page == "🔮 Predict Costs":
        st.markdown("---")
        st.markdown('<p class="sub-header">🔮 Insurance Cost Predictor</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #718096; font-size: 1.1rem;">Enter your details below to get an instant insurance cost estimate powered by AI</p>', unsafe_allow_html=True)
        
        st.markdown('---')
        
        # Input form with better organization
        st.markdown('<p class="section-title">📝 Enter Your Information</p>', unsafe_allow_html=True)
        
        # Organize inputs into logical sections
        st.markdown("""<p style="color: #718096; font-size: 0.95rem; margin-bottom: 1rem;">👤 <strong>Personal Information</strong></p>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age (years)", min_value=18, max_value=64, value=30, step=1)
            sex = st.selectbox("Sex", ["male", "female"])
        
        with col2:
            bmi = st.number_input("BMI (Body Mass Index)", 
                                 min_value=15.0, 
                                 max_value=55.0, 
                                 value=25.0, 
                                 step=0.1,
                                 help="BMI = weight(kg) / height(m)²")
            children = st.slider("Number of Children", min_value=0, max_value=5, value=0, step=1)
        
        st.markdown("""<p style="color: #718096; font-size: 0.95rem; margin: 1.5rem 0 1rem 0;">🏥 <strong>Health Information</strong></p>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            smoker = st.selectbox("Smoking Status", ["no", "yes"])
        
        with col2:
            region = st.selectbox("Geographic Region", ["northwest", "northeast", "southeast", "southwest"])
        
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
            
            # Display result with animation
            st.markdown('<p class="section-title" style="text-align: center;">💰 Your Predicted Annual Insurance Cost</p>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="prediction-result">
                    $ {prediction:,.2f}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('---')
            
            # Additional information in styled cards
            st.markdown('<p class="section-title">📊 Input Summary</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="feature-card" style="text-align: center;">
                        <h3 style='color: #667eea; margin: 0;'>{age}</h3>
                        <p style='margin: 0.5rem 0 0 0; color: #718096;'>Years Old</p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="feature-card" style="text-align: center;">
                        <h3 style='color: #667eea; margin: 0;'>{smoker.capitalize()}</h3>
                        <p style='margin: 0.5rem 0 0 0; color: #718096;'>Smoking Status</p>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="feature-card" style="text-align: center;">
                        <h3 style='color: #667eea; margin: 0;'>{bmi}</h3>
                        <p style='margin: 0.5rem 0 0 0; color: #718096;'>BMI Value</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('---')
            
            # Comparison with dataset average in styled box
            avg_charge = df['charges'].mean()
            if prediction > avg_charge:
                st.markdown(f"""
                    <div class="insight-box" style="border-left-color: #f5576c; background: #fff5f5;">
                        <strong>⚠️ Above Average:</strong> Your predicted cost is <strong>${(prediction - avg_charge):,.2f}</strong> above the average (${avg_charge:,.2f})
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="insight-box" style="border-left-color: #48bb78; background: #f0fff4;">
                        <strong>✅ Below Average:</strong> Your predicted cost is <strong>${(avg_charge - prediction):,.2f}</strong> below the average (${avg_charge:,.2f})
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('---')
            
            # Interpretation guide
            st.markdown('<p class="section-title">📖 Understanding Your Prediction</p>', unsafe_allow_html=True)
            st.markdown("""
                <div class="feature-card">
                    <p style='margin: 0.5rem 0; color: #718096;'>
                        This prediction is based on an XGBoost regression model trained on 1,338 insurance records. 
                        The model considers your age, BMI, number of children, and smoking status to estimate your 
                        annual insurance cost.
                    </p>
                    <p style='margin: 0.5rem 0; color: #718096;'>
                        <strong>Key factors affecting your premium:</strong><br>
                        • Smoking status has the largest impact<br>
                        • Age is the second most important factor<br>
                        • BMI and number of children have moderate effects
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # About Page
    elif page == "ℹ️ About":
        st.markdown("---")
        st.markdown('<p class="sub-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h3 style='margin: 0 0 1rem 0;'>🎯 Project Objective</h3>
                <p style='margin: 0; line-height: 1.8; font-size: 1.1rem;'>
                This project aims to predict medical insurance costs using machine learning techniques. 
                By analyzing various personal and health-related factors, the model provides accurate 
                cost estimations that can help both insurance providers and customers make informed decisions.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="section-title">🛠️ Technology Stack</p>', unsafe_allow_html=True)
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.markdown("""
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 1rem 0;'>📦 Data Processing</h4>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Pandas</strong> - Data manipulation</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>NumPy</strong> - Numerical computing</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Feature-engine</strong> - Outlier treatment</p>
                </div>
                <div class="feature-card" style="margin-top: 1rem;">
                    <h4 style='color: #667eea; margin: 0 0 1rem 0;'>📊 Visualization</h4>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Matplotlib</strong> - Plotting library</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Seaborn</strong> - Statistical graphics</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Streamlit</strong> - Web framework</p>
                </div>
            """, unsafe_allow_html=True)
        with tech_col2:
            st.markdown("""
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 1rem 0;'>🤖 Machine Learning</h4>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>XGBoost</strong> - Gradient boosting</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Scikit-learn</strong> - ML toolkit</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>GridSearchCV</strong> - Hyperparameter tuning</p>
                </div>
                <div class="feature-card" style="margin-top: 1rem;">
                    <h4 style='color: #667eea; margin: 0 0 1rem 0;'>🔧 Development</h4>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Jupyter Notebook</strong> - Exploration</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Python 3.8+</strong> - Programming language</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• <strong>Git</strong> - Version control</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                        <th style='padding: 1rem; text-align: left;'>Metric</th>
                        <th style='padding: 1rem; text-align: left;'>Score</th>
                    </tr>
                    <tr style='border-bottom: 1px solid #e2e8f0;'>
                        <td style='padding: 1rem;'>Training R²</td>
                        <td style='padding: 1rem; font-weight: 600; color: #667eea;'>0.869</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #e2e8f0;'>
                        <td style='padding: 1rem;'>Testing R²</td>
                        <td style='padding: 1rem; font-weight: 600; color: #667eea;'>0.901</td>
                    </tr>
                    <tr>
                        <td style='padding: 1rem;'>Cross-Validation (CV)</td>
                        <td style='padding: 1rem; font-weight: 600; color: #667eea;'>0.861</td>
                    </tr>
                </table>
                <p style='margin-top: 1rem; color: #718096; font-size: 0.95rem;'>
                    <strong>Note:</strong> R² (R-squared) score measures how well the model explains the variance in the data. 
                    An R² of 0.901 means the model explains 90.1% of the variance in insurance charges.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="section-title">🚀 Model Details</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <p style='margin: 0.5rem 0;'><strong>Algorithm:</strong> XGBoost Regressor</p>
                <p style='margin: 1rem 0 0.5rem 0;'><strong>Optimized Hyperparameters:</strong></p>
                <p style='margin: 0.5rem 0; color: #718096;'>• n_estimators: 15</p>
                <p style='margin: 0.5rem 0; color: #718096;'>• max_depth: 3</p>
                <p style='margin: 0.5rem 0; color: #718096;'>• gamma: 0</p>
                <p style='margin: 1rem 0 0.5rem 0;'><strong>Data Preprocessing:</strong></p>
                <p style='margin: 0.5rem 0; color: #718096;'>• Outlier treatment using IQR method (BMI capping)</p>
                <p style='margin: 0.5rem 0; color: #718096;'>• Label encoding for categorical variables</p>
                <p style='margin: 0.5rem 0; color: #718096;'>• Feature selection (sex and region removed for optimal performance)</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('---')
                
        st.markdown('<p class="section-title">📂 Project Files</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <p style='margin: 0.5rem 0; color: #718096;'><strong>📓 insurance_cost.ipynb</strong> - Jupyter notebook with complete ML pipeline</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>📊 insurance_data.csv</strong> - Dataset with 1,338 records</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>🤖 insurancemodelf.pkl</strong> - Trained XGBoost model</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>🌐 app.py</strong> - Streamlit web application</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>📋 requirements.txt</strong> - Python dependencies</p>
            </div>
        """, unsafe_allow_html=True)
                
        st.markdown('---')
        
        st.markdown('<p class="section-title">📂 Project Files</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <p style='margin: 0.5rem 0; color: #718096;'><strong>📓 insurance_cost.ipynb</strong> - Jupyter notebook with complete ML pipeline</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>📊 insurance_data.csv</strong> - Dataset with 1,338 records</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>🤖 insurancemodelf.pkl</strong> - Trained XGBoost model</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>🌐 app.py</strong> - Streamlit web application</p>
                <p style='margin: 0.5rem 0; color: #718096;'><strong>📋 requirements.txt</strong> - Python dependencies</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('---')
        st.markdown('<p class="section-title">📈 Key Findings</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="insight-box">
                <strong>1.</strong> <strong>Smoking status</strong> is the most significant predictor of insurance costs (smokers pay approximately 3x more)
            </div>
            <div class="insight-box">
                <strong>2.</strong> <strong>Age</strong> shows a strong positive correlation with premiums
            </div>
            <div class="insight-box">
                <strong>3.</strong> <strong>BMI</strong> has a moderate impact on cost predictions
            </div>
            <div class="insight-box">
                <strong>4.</strong> <strong>Number of children</strong> has minimal influence on premiums
            </div>
            <div class="insight-box">
                <strong>5.</strong> Features like <strong>sex</strong> and <strong>region</strong> were removed during feature selection as they had little predictive power
            </div>
            <div class="insight-box">
                <strong>6.</strong> The model achieves an <strong>R² score of 0.901</strong> on the test set
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('---')
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                <h3 style="margin: 0 0 1rem 0;">📊 Dataset Source</h3>
                <p style="margin: 0; font-size: 1.1rem;">Hugging Face - Insurance Dataset</p>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">adegoke655/Insurance</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

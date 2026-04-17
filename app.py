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
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
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
    st.markdown('<p style="text-align: center; color: #718096; font-size: 1.3rem; margin-bottom: 2rem;">AI-Powered Insurance Premium Estimation with 90.4% Accuracy</p>', unsafe_allow_html=True)
    
    # Add a separator line
    st.markdown('---')
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Data Analysis", "🔮 Predict Costs", "ℹ️ About"])
    
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
                    <h2 style='margin: 0; font-size: 2.5rem;'>1,339</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>Dataset Records</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="stat-card">
                    <h2 style='margin: 0; font-size: 2.5rem;'>6</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>Input Features</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="stat-card">
                    <h2 style='margin: 0; font-size: 2.5rem;'>90.4%</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>Model Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div class="info-box">
                <h3 style='margin: 0 0 1rem 0;'>🎯 Project Overview</h3>
                <p style='margin: 0; line-height: 1.8; font-size: 1.1rem;'>
                This application uses an advanced <strong>XGBoost machine learning model</strong> to predict insurance costs 
                based on personal health and demographic factors. The model has been trained on 1,339 insurance 
                records and achieves a <strong>test accuracy of 90.4%</strong>.
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
                    <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>🎯 90.4% Accuracy</h4>
                    <p style='margin: 0; color: #718096;'>High-precision ML model</p>
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
                <p style='margin: 0.5rem 0;'><strong>👤 Sex:</strong> Gender (male/female)</p>
                <p style='margin: 0.5rem 0;'><strong>📊 BMI:</strong> Body Mass Index (weight/height ratio)</p>
                <p style='margin: 0.5rem 0;'><strong>👶 Children:</strong> Number of dependents covered</p>
                <p style='margin: 0.5rem 0;'><strong>🚬 Smoker:</strong> Smoking status (yes/no)</p>
                <p style='margin: 0.5rem 0;'><strong>📍 Region:</strong> Geographic location in the US</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Data Analysis Page
    elif page == "📊 Data Analysis":
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
        
        # Dataset overview in a styled container
        st.markdown('<p class="section-title">📄 Dataset Statistics</p>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
        
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
                    <p style='margin: 0.5rem 0; color: #718096;'>• Pandas</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• NumPy</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• Feature-engine</p>
                </div>
            """, unsafe_allow_html=True)
        with tech_col2:
            st.markdown("""
                <div class="feature-card">
                    <h4 style='color: #667eea; margin: 0 0 1rem 0;'>🤖 Machine Learning</h4>
                    <p style='margin: 0.5rem 0; color: #718096;'>• XGBoost</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• Scikit-learn</p>
                    <p style='margin: 0.5rem 0; color: #718096;'>• Streamlit</p>
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
                        <td style='padding: 1rem; font-weight: 600; color: #667eea;'>0.870</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #e2e8f0;'>
                        <td style='padding: 1rem;'>Testing R²</td>
                        <td style='padding: 1rem; font-weight: 600; color: #667eea;'>0.904</td>
                    </tr>
                    <tr>
                        <td style='padding: 1rem;'>Cross-Validation</td>
                        <td style='padding: 1rem; font-weight: 600; color: #667eea;'>0.860</td>
                    </tr>
                </table>
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
        
        st.markdown("---")
        st.markdown('<p class="section-title">📈 Key Findings</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="insight-box">
                <strong>1.</strong> <strong>Smoking status</strong> is the most significant predictor of insurance costs
            </div>
            <div class="insight-box">
                <strong>2.</strong> <strong>Age</strong> shows a strong positive correlation with premiums
            </div>
            <div class="insight-box">
                <strong>3.</strong> <strong>BMI</strong> has a moderate impact on cost predictions
            </div>
            <div class="insight-box">
                <strong>4.</strong> <strong>Geographic region</strong> and <strong>gender</strong> have minimal predictive power
            </div>
            <div class="insight-box">
                <strong>5.</strong> The model achieves <strong>90.4% accuracy</strong> on unseen test data
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Dataset Source:** [Hugging Face - Insurance Dataset](https://huggingface.co/datasets/adegoke655/Insurance)")

if __name__ == "__main__":
    main()

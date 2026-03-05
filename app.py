"""
Student Study Hours Prediction - Streamlit ML Pipeline Application
=====================================================================

This Streamlit application demonstrates a complete Machine Learning pipeline
using Linear Regression to predict student study hours based on lifestyle
and behavior features.

Author: ML Engineer
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import io

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Set page configuration
st.set_page_config(
    page_title="Student Study Hours Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar-section {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    h3 {
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_dataset():
    """Load the student productivity dataset."""
    df = pd.read_csv('ultimate_student_productivity_dataset_5000.csv')
    return df

def clean_dataset(df):
    """
    Clean the dataset by removing duplicates and handling missing values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to clean
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
    # Create a copy
    df_clean = df.copy()
    
    # Remove duplicate records
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # Handle missing values using median for numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    return df_clean, duplicates_removed

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    
    Returns:
    --------
    LinearRegression
        Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using various metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }

def save_model_pkl(model, feature_names):
    """Save the trained model as a pickle file."""
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    return pickle.dumps(model_data)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main function to run the Streamlit application."""
    
    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    
    st.sidebar.title("📚 Navigation")
    st.sidebar.markdown("---")
    
    # Sidebar menu
    menu_options = [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Data Understanding",
        "🧹 Data Cleaning",
        "🎯 Feature Selection",
        "📉 Visualization",
        "🔧 Model Training",
        "🎓 Prediction",
        "📏 Model Evaluation",
        "📊 Model Performance"
    ]
    
    selected_section = st.sidebar.radio("Go to Section", menu_options)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application demonstrates a complete ML pipeline "
        "to predict student study hours based on lifestyle habits."
    )
    
    # =========================================================================
    # HOME SECTION
    # =========================================================================
    
    if selected_section == "🏠 Home":
        st.title("📚 Student Study Hours Predictor")
        st.markdown("---")
        
        # Welcome message
        st.markdown("""
        ### Welcome to the Student Study Hours Prediction Application! 🎓
        
        This application demonstrates a complete Machine Learning pipeline 
        using **Linear Regression** to predict how many hours a student 
        studies per day based on their lifestyle and behavior features.
        
        ### 🎯 Objective
        Build a predictive model that analyzes student lifestyle habits 
        and predicts study hours.
        
        ### 📋 Features Used for Prediction
        - **Age**: Student's age
        - **Social Media Hours**: Hours spent on social media daily
        - **Gaming Hours**: Hours spent gaming daily
        - **Sleep Hours**: Hours of sleep per night
        - **Exercise Minutes**: Minutes of exercise per day
        
        ### 🚀 How to Use
        1. Navigate through the sections using the sidebar
        2. Explore the dataset and understand the data
        3. Clean the data if needed
        4. Train the Linear Regression model
        5. Make predictions with custom inputs
        6. Evaluate model performance
        
        ### 📊 ML Pipeline Steps
        """)
        
        # Pipeline steps visualization
        pipeline_steps = [
            ("1. Data Loading", "📥", "Load and explore the dataset"),
            ("2. Data Understanding", "🔍", "Analyze statistics and distributions"),
            ("3. Data Cleaning", "🧹", "Remove duplicates, handle missing values"),
            ("4. Feature Selection", "🎯", "Select relevant features"),
            ("5. Visualization", "📉", "Create visualizations"),
            ("6. Train-Test Split", "✂️", "Split data 80/20"),
            ("7. Model Training", "⚙️", "Train Linear Regression"),
            ("8. Prediction", "🎓", "Make predictions"),
            ("9. Evaluation", "📏", "Measure model performance")
        ]
        
        cols = st.columns(3)
        for idx, (step, icon, desc) in enumerate(pipeline_steps):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="sidebar-section">
                    <h4>{icon} {step}</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Get started button
        st.markdown("---")
        if st.button("🚀 Get Started - Load Dataset"):
            st.sidebar.success("Navigate to Dataset Overview section!")
    
    # =========================================================================
    # DATASET OVERVIEW SECTION
    # =========================================================================
    
    elif selected_section == "📊 Dataset Overview":
        st.title("📊 Dataset Overview")
        st.markdown("---")
        
        # Load dataset
        df = load_dataset()
        
        # Dataset path info
        st.markdown("### 📁 Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("File Size", "Available")
        
        # Dataset preview
        st.markdown("### 👀 Dataset Preview")
        st.markdown("First 5 rows of the dataset:")
        
        # Scrollable table
        st.dataframe(df.head(5), use_container_width=True, height=300)
        
        # Column names
        st.markdown("### 📋 Column Names")
        st.code("\n".join(df.columns.tolist()), language="python")
        
        # Data types
        st.markdown("### 🔢 Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtype_df, use_container_width=True)
        
        # Dataset summary card
        st.markdown("### 📊 Dataset Summary Card")
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset Summary</h3>
            <p><strong>Source:</strong> ultimate_student_productivity_dataset_5000.csv</p>
            <p><strong>Total Records:</strong> 5,000 students</p>
            <p><strong>Features:</strong> 21 columns including demographics, lifestyle, and academic metrics</p>
            <p><strong>Target Variable:</strong> study_hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # DATA UNDERSTANDING SECTION
    # =========================================================================
    
    elif selected_section == "📈 Data Understanding":
        st.title("📈 Data Understanding")
        st.markdown("---")
        
        # Load dataset
        df = load_dataset()
        
        # Summary statistics
        st.markdown("### 📊 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Missing values
        st.markdown("### ❌ Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': (missing.values / len(df) * 100).round(2)
        })
        
        if missing.sum() == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.dataframe(missing_df, use_container_width=True)
        
        # Distribution of study hours
        st.markdown("### 📊 Distribution of Study Hours")
        st.markdown("Histogram showing the distribution of study hours:")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['study_hours'], bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        ax.set_xlabel('Study Hours', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Study Hours', fontsize=14, fontweight='bold')
        ax.axvline(df['study_hours'].mean(), color='red', linestyle='--', label=f'Mean: {df["study_hours"].mean():.2f}')
        ax.legend()
        st.pyplot(fig)
        
        # Average study hours by age group
        st.markdown("### 📊 Average Study Hours by Age Group")
        st.markdown("Bar chart showing average study hours by age:")
        
        # Create age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 20, 22, 25, 100], 
                                  labels=['≤18', '19-20', '21-22', '23-25', '>25'])
        avg_by_age = df.groupby('age_group')['study_hours'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(avg_by_age.index, avg_by_age.values, edgecolor='black', alpha=0.7, color='#2ecc71')
        ax.set_xlabel('Age Group', fontsize=12)
        ax.set_ylabel('Average Study Hours', fontsize=12)
        ax.set_title('Average Study Hours by Age Group', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_by_age.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        st.pyplot(fig)
    
    # =========================================================================
    # DATA CLEANING SECTION
    # =========================================================================
    
    elif selected_section == "🧹 Data Cleaning":
        st.title("🧹 Data Cleaning")
        st.markdown("---")
        
        # Load dataset
        df = load_dataset()
        
        # Show original data info
        st.markdown("### 📊 Original Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", df.shape[0])
        with col2:
            st.metric("Original Columns", df.shape[1])
        
        # Clean dataset button
        st.markdown("### 🧹 Clean Dataset")
        if st.button("🗑️ Clean Dataset"):
            with st.spinner("Cleaning dataset..."):
                df_cleaned, duplicates_removed = clean_dataset(df)
                st.session_state.df_cleaned = df_cleaned
                
                st.success(f"✅ Dataset cleaned successfully!")
                st.info(f"📝 Removed {duplicates_removed} duplicate records")
                
                # Show cleaned dataset info
                st.markdown("### 📊 Cleaned Dataset Info")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cleaned Rows", df_cleaned.shape[0])
                with col2:
                    st.metric("Cleaned Columns", df_cleaned.shape[1])
                
                # Show cleaned dataset preview
                st.markdown("### 👀 Cleaned Dataset Preview")
                st.dataframe(df_cleaned.head(10), use_container_width=True)
        else:
            # Use original data if not cleaned
            st.session_state.df_cleaned = df
    
    # =========================================================================
    # FEATURE SELECTION SECTION
    # =========================================================================
    
    elif selected_section == "🎯 Feature Selection":
        st.title("🎯 Feature Selection")
        st.markdown("---")
        
        # Define features
        target_variable = "study_hours"
        input_features = ['age', 'social_media_hours', 'gaming_hours', 'sleep_hours', 'exercise_minutes']
        
        # Display target variable
        st.markdown("### 🎯 Target Variable")
        st.markdown(f"""
        <div class="metric-card">
            <h3>{target_variable}</h3>
            <p>This is the variable we want to predict - the number of hours a student studies per day.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display input features
        st.markdown("### 📥 Input Features")
        st.markdown("The following features will be used for prediction:")
        
        feature_descriptions = {
            'age': "Student's age in years",
            'social_media_hours': "Hours spent on social media daily",
            'gaming_hours': "Hours spent gaming daily",
            'sleep_hours': "Hours of sleep per night",
            'exercise_minutes': "Minutes of exercise per day"
        }
        
        for feature in input_features:
            st.markdown(f"""
            <div class="sidebar-section">
                <h4>📌 {feature}</h4>
                <p>{feature_descriptions[feature]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show selected features summary
        st.markdown("### ✅ Selected Features Summary")
        
        df = load_dataset()
        selected_df = df[input_features + [target_variable]]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Input Features", len(input_features))
        with col2:
            st.metric("Target Variable", 1)
        
        st.dataframe(selected_df.head(), use_container_width=True)
    
    # =========================================================================
    # VISUALIZATION SECTION
    # =========================================================================
    
    elif selected_section == "📉 Visualization":
        st.title("📉 Data Visualization")
        st.markdown("---")
        
        # Load dataset
        df = load_dataset()
        
        # Features for visualization
        features = ['age', 'social_media_hours', 'gaming_hours', 'sleep_hours', 'exercise_minutes', 'study_hours']
        df_viz = df[features]
        
        # 1. Correlation Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        st.markdown("Shows relationships between lifestyle habits and study hours:")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df_viz.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        # 2. Scatter Plots
        st.markdown("### 📈 Scatter Plots")
        st.markdown("Relationship between lifestyle habits and study hours:")
        
        scatter_features = ['social_media_hours', 'gaming_hours', 'sleep_hours']
        
        for feature in scatter_features:
            st.markdown(f"**{feature.replace('_', ' ').title()} vs Study Hours**")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df[feature], df['study_hours'], alpha=0.5, c='#3498db', edgecolors='white')
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Study Hours', fontsize=12)
            ax.set_title(f'{feature.replace("_", " ").title()} vs Study Hours', fontsize=12, fontweight='bold')
            
            # Add trend line
            z = np.polyfit(df[feature], df['study_hours'], 1)
            p = np.poly1d(z)
            ax.plot(df[feature].sort_values(), p(df[feature].sort_values()), 
                   "r--", alpha=0.8, label='Trend Line')
            ax.legend()
            st.pyplot(fig)
        
        # 3. Pair Plot
        st.markdown("### 🔗 Pair Plot")
        st.markdown("Shows relationships between all selected features:")
        
        # Select subset for pair plot (to keep it fast)
        pairplot_features = ['study_hours', 'social_media_hours', 'sleep_hours', 'exercise_minutes']
        df_pairplot = df[pairplot_features].head(500)  # Limit for performance
        
        fig = plt.figure(figsize=(12, 10))
        g = sns.pairplot(df_pairplot, diag_kind='kde', plot_kws={'alpha': 0.5}, 
                        diag_kws={'fill': True})
        g.fig.suptitle('Pair Plot of Selected Features', y=1.02, fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    # =========================================================================
    # MODEL TRAINING SECTION
    # =========================================================================
    
    elif selected_section == "🔧 Model Training":
        st.title("🔧 Model Training")
        st.markdown("---")
        
        # Load and prepare data
        df = load_dataset()
        
        # Define features
        input_features = ['age', 'social_media_hours', 'gaming_hours', 'sleep_hours', 'exercise_minutes']
        X = df[input_features]
        y = df['study_hours']
        
        # Train-Test Split
        st.markdown("### ✂️ Train-Test Split")
        st.markdown("Splitting the dataset into training and testing sets:")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Set Size", f"{len(X_train)} samples (80%)")
        with col2:
            st.metric("Testing Set Size", f"{len(X_test)} samples (20%)")
        
        # Train Model Button
        st.markdown("### ⚙️ Train Linear Regression Model")
        if st.button("🚀 Train Model"):
            with st.spinner("Training Linear Regression model..."):
                # Train the model
                model = train_linear_regression(X_train, y_train)
                st.session_state.model = model
                
                # Make predictions
                y_pred = model.predict(X_test)
                st.session_state.y_pred = y_pred
                
                st.success("✅ Model trained successfully!")
                
                # Display model coefficients
                st.markdown("### 📊 Model Coefficients")
                st.markdown("The coefficients represent the impact of each feature on study hours:")
                
                coef_df = pd.DataFrame({
                    'Feature': input_features,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                st.dataframe(coef_df, use_container_width=True)
                
                # Display intercept
                st.markdown(f"### 📍 Intercept")
                st.metric("Intercept Value", f"{model.intercept_:.4f}")
                
                # Model equation
                equation = f"study_hours = {model.intercept_:.4f}"
                for feat, coef in zip(input_features, model.coef_):
                    equation += f" + ({coef:.4f} × {feat})"
                
                st.code(equation, language="python")
        else:
            st.info("👆 Click the button above to train the model")
    
    # =========================================================================
    # PREDICTION SECTION
    # =========================================================================
    
    elif selected_section == "🎓 Prediction":
        st.title("🎓 Prediction")
        st.markdown("---")
        
        # Check if model is trained
        if st.session_state.model is None:
            st.warning("⚠️ Please train the model first! Go to Model Training section.")
        else:
            st.markdown("### 🎯 Predict Study Hours")
            st.markdown("Enter the values below to predict study hours:")
            
            # Interactive input form
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", min_value=15, max_value=30, value=20, step=1)
                social_media = st.slider("Social Media Hours", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
                gaming = st.slider("Gaming Hours", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            
            with col2:
                sleep = st.slider("Sleep Hours", min_value=4.0, max_value=12.0, value=7.0, step=0.5)
                exercise = st.slider("Exercise Minutes", min_value=0, max_value=180, value=30, step=5)
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'social_media_hours': [social_media],
                'gaming_hours': [gaming],
                'sleep_hours': [sleep],
                'exercise_minutes': [exercise]
            })
            
            st.markdown("### 📝 Input Summary")
            st.dataframe(input_data.T, use_container_width=True)
            
            # Predict button
            if st.button("🎓 Predict Study Hours"):
                # Make prediction
                prediction = st.session_state.model.predict(input_data)[0]
                
                st.markdown("---")
                st.markdown(f"""
                <div class="metric-card">
                    <h2>🎓 Predicted Study Hours</h2>
                    <h1 style="color: #27ae60; font-size: 48px;">{prediction:.2f} hours/day</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 💡 Insights")
                if prediction > 6:
                    st.success("📚 High study hours predicted! This student appears to be very dedicated.")
                elif prediction > 3:
                    st.info("📖 Moderate study hours predicted. This is a balanced study routine.")
                else:
                    st.warning("⚠️ Low study hours predicted. Consider encouraging better study habits.")
    
    # =========================================================================
    # MODEL EVALUATION SECTION
    # =========================================================================
    
    elif selected_section == "📏 Model Evaluation":
        st.title("📏 Model Evaluation")
        st.markdown("---")
        
        # Check if model is trained and predictions are made
        if st.session_state.model is None or st.session_state.y_pred is None:
            st.warning("⚠️ Please train the model first! Go to Model Training section.")
        else:
            # Calculate metrics
            metrics = evaluate_model(st.session_state.y_test, st.session_state.y_pred)
            
            st.markdown("### 📊 Evaluation Metrics")
            st.markdown("Model performance metrics:")
            
            # Display metrics in styled cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>MAE</h4>
                    <h2 style="color: #3498db;">{metrics['MAE']:.4f}</h2>
                    <p>Mean Absolute Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>MSE</h4>
                    <h2 style="color: #e74c3c;">{metrics['MSE']:.4f}</h2>
                    <p>Mean Squared Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>RMSE</h4>
                    <h2 style="color: #9b59b6;">{metrics['RMSE']:.4f}</h2>
                    <p>Root Mean Squared Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>R² Score</h4>
                    <h2 style="color: #27ae60;">{metrics['R2 Score']:.4f}</h2>
                    <p>Coefficient of Determination</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics explanation
            st.markdown("### 📖 Metric Explanations")
            st.markdown("""
            - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
            - **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
            - **RMSE (Root Mean Squared Error)**: Square root of MSE, provides error in same units as target
            - **R² Score**: Proportion of variance explained by the model (1.0 is perfect)
            """)
            
            # Download model button
            st.markdown("---")
            st.markdown("### 💾 Download Trained Model")
            
            if st.button("📥 Download Model as .pkl"):
                model_data = save_model_pkl(st.session_state.model, 
                                           ['age', 'social_media_hours', 'gaming_hours', 'sleep_hours', 'exercise_minutes'])
                st.download_button(
                    label="Click to Download model.pkl",
                    data=model_data,
                    file_name="student_study_hours_model.pkl",
                    mime="application/octet-stream"
                )
    
    # =========================================================================
    # MODEL PERFORMANCE VISUALIZATION SECTION
    # =========================================================================
    
    elif selected_section == "📊 Model Performance":
        st.title("📊 Visual Model Performance")
        st.markdown("---")
        
        # Check if model is trained and predictions are made
        if st.session_state.model is None or st.session_state.y_pred is None:
            st.warning("⚠️ Please train the model first! Go to Model Training section.")
        else:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            # 1. Actual vs Predicted Scatter Plot
            st.markdown("### 📈 Actual vs Predicted Scatter Plot")
            st.markdown("Compare predicted values against actual values:")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(y_test, y_pred, alpha=0.5, c='#3498db', edgecolors='white', s=50)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Study Hours', fontsize=12)
            ax.set_ylabel('Predicted Study Hours', fontsize=12)
            ax.set_title('Actual vs Predicted Study Hours', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 2. Residual Error Plot
            st.markdown("### 📉 Residual Error Plot")
            st.markdown("Shows the difference between predicted and actual values:")
            
            residuals = y_test - y_pred
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_pred, residuals, alpha=0.5, c='#e74c3c', edgecolors='white', s=50)
            ax.axhline(y=0, color='black', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Study Hours', fontsize=12)
            ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
            ax.set_title('Residual Error Plot', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Residual distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='#e74c3c')
            ax.axvline(x=0, color='black', linestyle='--', lw=2)
            ax.set_xlabel('Residuals', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
            # 3. Prediction Trend Graph
            st.markdown("### 📊 Prediction Trend Graph")
            st.markdown("Shows prediction trend across test samples:")
            
            # Sort by actual values for better visualization
            sorted_indices = np.argsort(y_test.values)
            sorted_actual = y_test.values[sorted_indices]
            sorted_predicted = y_pred[sorted_indices]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(sorted_actual)), sorted_actual, 'b-', label='Actual', alpha=0.7, linewidth=2)
            ax.plot(range(len(sorted_predicted)), sorted_predicted, 'r--', label='Predicted', alpha=0.7, linewidth=2)
            ax.set_xlabel('Sample Index (sorted by actual)', fontsize=12)
            ax.set_ylabel('Study Hours', fontsize=12)
            ax.set_title('Prediction Trend Graph', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import io

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .medium-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'best_threshold' not in st.session_state:
    st.session_state.best_threshold = 0.5
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None

# Helper Functions
def load_and_preprocess_data(df):
    """Preprocess the uploaded data"""
    # Map Churn Label if it exists
    if 'Churn Label' in df.columns:
        df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    
    # Columns to drop (if they exist)
    columns_to_drop = ['Customer ID', 'Country', 'Population', 'Referred a Friend', 
                      'Dependents', 'State', 'City', 'Zip Code', 'Latitude', 
                      'Longitude', 'Quarter', 'Churn Category', 'Churn Reason']
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(existing_columns_to_drop, axis=1)
    
    return df

def train_model(df):
    """Train the Random Forest model"""
    with st.spinner("🔄 Training model... This may take a few minutes."):
        X = df.drop("Churn Label", axis=1)
        y = df["Churn Label"]
        
        categorical_cols = X.select_dtypes(include='object').columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)
        
        param_dist = {
            'n_estimators': [100, 150],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        rand_search = RandomizedSearchCV(
            rf, param_distributions=param_dist,
            n_iter=15, cv=3, scoring='f1', n_jobs=-1, random_state=42
        )
        
        rand_search.fit(X_train_bal, y_train_bal)
        best_rf = rand_search.best_estimator_
        
        X_test_processed = preprocessor.transform(X_test)
        y_proba = best_rf.predict_proba(X_test_processed)[:,1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        j_scores = tpr - fpr
        best_thresh = thresholds[np.argmax(j_scores)]
        
        return best_rf, preprocessor, best_thresh, X_test, y_test

def predict_churn(model, preprocessor, df, threshold):
    """Make predictions on the dataset"""
    X = df.drop("Churn Label", axis=1, errors='ignore')
    X_processed = preprocessor.transform(X)
    
    y_proba = model.predict_proba(X_processed)[:,1]
    y_pred = (y_proba >= threshold).astype(int)
    
    predictions_df = df.copy()
    predictions_df['Churn_Probability'] = y_proba
    predictions_df['Predicted_Churn'] = y_pred
    predictions_df['Risk_Level'] = pd.cut(
        y_proba, 
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return predictions_df

def get_risk_color(risk):
    """Return color based on risk level"""
    if risk == 'High':
        return '#d62728'
    elif risk == 'Medium':
        return '#ff7f0e'
    else:
        return '#2ca02c'

# Sidebar
st.sidebar.title("🎛️ Control Panel")

# Data Upload Section
st.sidebar.header("📁 Data Management")
upload_option = st.sidebar.radio(
    "Choose data upload method:",
    ["Upload New Dataset", "Incremental Update"]
)

if upload_option == "Upload New Dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        key="full_upload"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values=['NA','?','-'])
        st.session_state.df = load_and_preprocess_data(df)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.df)} records")
        
        if st.sidebar.button("🚀 Train Model"):
            if 'Churn Label' not in st.session_state.df.columns:
                st.sidebar.error("❌ 'Churn Label' column required for training!")
            else:
                model, preprocessor, threshold, X_test, y_test = train_model(st.session_state.df)
                st.session_state.model = model
                st.session_state.preprocessor = preprocessor
                st.session_state.best_threshold = threshold
                st.sidebar.success(f"✅ Model trained! Threshold: {threshold:.3f}")

else:  # Incremental Update
    incremental_file = st.sidebar.file_uploader(
        "Upload additional CSV data",
        type=['csv'],
        key="incremental_upload"
    )
    
    if incremental_file is not None and st.session_state.df is not None:
        new_df = pd.read_csv(incremental_file, na_values=['NA','?','-'])
        new_df = load_and_preprocess_data(new_df)
        
        st.sidebar.info(f"📊 Current records: {len(st.session_state.df)}")
        st.sidebar.info(f"📥 New records: {len(new_df)}")
        
        if st.sidebar.button("➕ Append Data"):
            st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
            st.sidebar.success(f"✅ Updated! Total: {len(st.session_state.df)} records")
    elif incremental_file is not None:
        st.sidebar.warning("⚠️ Please upload a base dataset first!")

# Main Content
st.markdown("<h1 class='main-header'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

if st.session_state.df is not None:
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Churn Predictions", 
        "📊 Analytics", 
        "📈 Custom Visualizations",
        "📋 Data Overview"
    ])
    
    # TAB 1: Churn Predictions
    with tab1:
        st.header("🎯 Customer Churn Risk Analysis")
        
        if st.session_state.model is not None:
            # Make predictions
            predictions_df = predict_churn(
                st.session_state.model,
                st.session_state.preprocessor,
                st.session_state.df,
                st.session_state.best_threshold
            )
            st.session_state.predictions_df = predictions_df
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_risk = len(predictions_df[predictions_df['Risk_Level'] == 'High'])
                st.metric("🔴 High Risk", high_risk, delta=f"{high_risk/len(predictions_df)*100:.1f}%")
            
            with col2:
                medium_risk = len(predictions_df[predictions_df['Risk_Level'] == 'Medium'])
                st.metric("🟠 Medium Risk", medium_risk, delta=f"{medium_risk/len(predictions_df)*100:.1f}%")
            
            with col3:
                low_risk = len(predictions_df[predictions_df['Risk_Level'] == 'Low'])
                st.metric("🟢 Low Risk", low_risk, delta=f"{low_risk/len(predictions_df)*100:.1f}%")
            
            with col4:
                avg_prob = predictions_df['Churn_Probability'].mean()
                st.metric("📊 Avg Churn Prob", f"{avg_prob:.2%}")
            
            st.markdown("---")
            
            # Filter section
            col1, col2 = st.columns([1, 3])
            
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    options=['High', 'Medium', 'Low'],
                    default=['High', 'Medium', 'Low']
                )
            
            with col2:
                prob_range = st.slider(
                    "Churn Probability Range",
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.05
                )
            
            # Filter data
            filtered_df = predictions_df[
                (predictions_df['Risk_Level'].isin(risk_filter)) &
                (predictions_df['Churn_Probability'] >= prob_range[0]) &
                (predictions_df['Churn_Probability'] <= prob_range[1])
            ]
            
            st.subheader(f"📋 Customer List ({len(filtered_df)} customers)")
            
            # Display predictions table
            display_cols = ['Churn_Probability', 'Risk_Level', 'Predicted_Churn']
            other_cols = [col for col in filtered_df.columns if col not in display_cols + ['Churn Label']]
            
            display_df = filtered_df[display_cols + other_cols].copy()
            display_df['Churn_Probability'] = display_df['Churn_Probability'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                display_df.head(100),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("⚠️ Please train the model first using the sidebar!")
    
    # TAB 2: Analytics
    with tab2:
        st.header("📊 Churn Analytics")
        
        if st.session_state.predictions_df is not None:
            predictions_df = st.session_state.predictions_df
            
            # Risk distribution pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Distribution")
                risk_counts = predictions_df['Risk_Level'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker=dict(colors=['#d62728', '#ff7f0e', '#2ca02c']),
                    hole=0.3
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Churn Probability Distribution")
                fig = px.histogram(
                    predictions_df,
                    x='Churn_Probability',
                    nbins=50,
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Categorical analysis
            st.subheader("📊 Churn Analysis by Demographics")
            
            categorical_vars = ['Contract', 'Internet Service', 'Payment Method', 
                              'Gender', 'Senior Citizen', 'Phone Service', 
                              'Online Security', 'Tech Support']
            
            available_cats = [col for col in categorical_vars if col in predictions_df.columns]
            
            if available_cats:
                selected_cat = st.selectbox("Select demographic variable:", available_cats)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Count by category
                    fig = px.histogram(
                        predictions_df,
                        x=selected_cat,
                        color='Risk_Level',
                        barmode='group',
                        color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
                    )
                    fig.update_layout(height=400, title=f"Risk Distribution by {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Average churn probability by category
                    avg_churn = predictions_df.groupby(selected_cat)['Churn_Probability'].mean().sort_values(ascending=False)
                    
                    fig = go.Figure(data=[
                        go.Bar(x=avg_churn.index, y=avg_churn.values, marker_color='#1f77b4')
                    ])
                    fig.update_layout(
                        height=400,
                        title=f"Average Churn Probability by {selected_cat}",
                        yaxis_title="Churn Probability",
                        xaxis_title=selected_cat
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Numerical analysis
            st.subheader("📈 Churn vs Numerical Features")
            
            numerical_vars = ['Tenure in Months', 'Monthly Charge', 'Total Charges', 'CLTV']
            available_nums = [col for col in numerical_vars if col in predictions_df.columns]
            
            if available_nums:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_num1 = st.selectbox("Select first variable:", available_nums, key='num1')
                
                with col2:
                    selected_num2 = st.selectbox("Select second variable:", available_nums, index=1 if len(available_nums) > 1 else 0, key='num2')
                
                # Box plots
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.box(
                        predictions_df,
                        x='Risk_Level',
                        y=selected_num1,
                        color='Risk_Level',
                        color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        predictions_df,
                        x=selected_num1,
                        y=selected_num2,
                        color='Churn_Probability',
                        color_continuous_scale='RdYlGn_r',
                        hover_data=['Risk_Level']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Please make predictions first in the 'Churn Predictions' tab!")
    
    # TAB 3: Custom Visualizations
    with tab3:
        st.header("📈 Custom Interactive Visualizations")
        
        if st.session_state.predictions_df is not None:
            predictions_df = st.session_state.predictions_df
            
            st.subheader("🎨 Create Custom Plots")
            
            # Get all columns
            all_cols = predictions_df.columns.tolist()
            numerical_cols = predictions_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = predictions_df.select_dtypes(include=['object']).columns.tolist()
            
            # Plot type selection
            plot_type = st.selectbox(
                "Select plot type:",
                ["Scatter Plot", "Bar Chart", "Box Plot", "Violin Plot", "Heatmap"]
            )
            
            col1, col2, col3 = st.columns(3)
            
            if plot_type == "Scatter Plot":
                with col1:
                    x_var = st.selectbox("X-axis:", numerical_cols, key='scatter_x')
                with col2:
                    y_var = st.selectbox("Y-axis:", numerical_cols, index=1 if len(numerical_cols) > 1 else 0, key='scatter_y')
                with col3:
                    color_var = st.selectbox("Color by:", ['Risk_Level', 'Churn_Probability'] + categorical_cols, key='scatter_color')
                
                fig = px.scatter(
                    predictions_df,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    size='Churn_Probability',
                    hover_data=all_cols[:5],
                    color_continuous_scale='RdYlGn_r' if color_var == 'Churn_Probability' else None
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Bar Chart":
                with col1:
                    x_var = st.selectbox("X-axis (categorical):", categorical_cols, key='bar_x')
                with col2:
                    y_var = st.selectbox("Y-axis (numerical):", numerical_cols, key='bar_y')
                with col3:
                    agg_func = st.selectbox("Aggregation:", ["mean", "median", "sum", "count"], key='bar_agg')
                
                grouped_data = predictions_df.groupby(x_var)[y_var].agg(agg_func).sort_values(ascending=False)
                
                fig = go.Figure(data=[
                    go.Bar(x=grouped_data.index, y=grouped_data.values, marker_color='#1f77b4')
                ])
                fig.update_layout(
                    height=600,
                    xaxis_title=x_var,
                    yaxis_title=f"{agg_func.capitalize()} of {y_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Box Plot":
                with col1:
                    x_var = st.selectbox("X-axis (categorical):", categorical_cols + ['Risk_Level'], key='box_x')
                with col2:
                    y_var = st.selectbox("Y-axis (numerical):", numerical_cols, key='box_y')
                
                fig = px.box(
                    predictions_df,
                    x=x_var,
                    y=y_var,
                    color=x_var,
                    points="outliers"
                )
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Violin Plot":
                with col1:
                    x_var = st.selectbox("X-axis (categorical):", categorical_cols + ['Risk_Level'], key='violin_x')
                with col2:
                    y_var = st.selectbox("Y-axis (numerical):", numerical_cols, key='violin_y')
                
                fig = px.violin(
                    predictions_df,
                    x=x_var,
                    y=y_var,
                    color=x_var,
                    box=True
                )
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Heatmap":
                st.info("📊 Correlation heatmap of numerical features")
                
                corr_matrix = predictions_df[numerical_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(height=600, width=800)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Please make predictions first in the 'Churn Predictions' tab!")
    
    # TAB 4: Data Overview
    with tab4:
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(st.session_state.df))
        with col2:
            st.metric("Features", len(st.session_state.df.columns))
        with col3:
            st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        with col4:
            if 'Churn Label' in st.session_state.df.columns:
                churn_rate = st.session_state.df['Churn Label'].mean()
                st.metric("Churn Rate", f"{churn_rate:.2%}")
        
        st.markdown("---")
        
        # Data preview
        st.subheader("📊 Data Preview")
        st.dataframe(st.session_state.df.head(100), use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Numerical Features Summary")
            st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("🏷️ Data Types")
            dtypes_df = pd.DataFrame({
                'Column': st.session_state.df.dtypes.index,
                'Data Type': st.session_state.df.dtypes.values
            })
            st.dataframe(dtypes_df, use_container_width=True, height=400)

else:
    st.info("👈 Please upload a dataset using the sidebar to get started!")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload Data**: Use the sidebar to upload your customer data CSV file
    2. **Train Model**: Click the "Train Model" button to build the prediction model
    3. **View Predictions**: Navigate to the "Churn Predictions" tab to see risk assessments
    4. **Analyze**: Explore the "Analytics" tab for insights
    5. **Customize**: Create custom visualizations in the "Custom Visualizations" tab
    
    ### 📋 Required Data Format
    
    Your CSV should include:
    - Customer features (demographics, services, charges, etc.)
    - `Churn Label` column (Yes/No) for training
    
    ### 🔄 Incremental Updates
    
    You can append new customer data without retraining by:
    1. Selecting "Incremental Update" in the sidebar
    2. Uploading additional CSV data
    3. Clicking "Append Data"
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>Churn Prediction Dashboard v1.0</p>
    <p>Powered by Random Forest ML</p>
    </div>
""", unsafe_allow_html=True)
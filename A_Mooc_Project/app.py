import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
from sklearn.metrics import confusion_matrix

# Set page config
st.set_page_config(
    page_title="Milling Machine Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stAlert {
        border-radius: 5px;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #e6f3ff;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .header {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Load models and components
@st.cache_resource
def load_models():
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    shap_explainer = joblib.load('shap_explainer.pkl')
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    performance_df = pd.read_pickle('model_performance.pkl')
    
    # Load real test data
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    
    return model, scaler, shap_explainer, feature_names, performance_df, X_test, y_test

model, scaler, shap_explainer, feature_names, performance_df, X_test, y_test = load_models()


original_feature_names = feature_names['original']
clean_feature_names = feature_names['clean']

# Dashboard Header
st.title("⚙️ Milling Machine Predictive Maintenance Dashboard")
st.markdown("""
This dashboard helps predict machine failures based on operational parameters.
Use the sidebar to input current machine parameters or explore model insights.
""")

# Sidebar for user input
with st.sidebar:
    st.header("Machine Parameters")
    
    # Create input sliders for each feature
    input_data = {}
    for i, feature in enumerate(original_feature_names):
        if '[K]' in feature:  # Temperature features
            min_val = 290 if 'Air' in feature else 305
            max_val = 305 if 'Air' in feature else 315
            default = 298 if 'Air' in feature else 308
            input_data[feature] = st.slider(
                feature, min_val, max_val, default
            )
        elif 'speed' in feature.lower():  # Rotational speed
            input_data[feature] = st.slider(
                feature, 1100, 3000, 1500
            )
        elif 'Torque' in feature:  # Torque
            input_data[feature] = st.slider(
                feature, 3.0, 80.0, 40.0
            )
        else:  # Tool wear
            input_data[feature] = st.slider(
                feature, 0, 250, 100
            )
    
    predict_button = st.button("Predict Failure Probability")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Insights", "Feature Analysis", "Performance"])

with tab1:
    st.header("Machine Failure Prediction")
    
    if predict_button:
        # Create input dataframe
        input_df = pd.DataFrame([input_data])
        
        # Scale the input
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        proba = model.predict_proba(scaled_input)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"🚨 Failure predicted with {proba[0][1]*100:.2f}% confidence")
            else:
                st.success(f"✅ No failure predicted ({proba[0][0]*100:.2f}% confidence)")
            
            # Show probability gauge
            fig = px.bar(
                x=["No Failure", "Failure"],
                y=[proba[0][0], proba[0][1]],
                labels={'x': 'Outcome', 'y': 'Probability'},
                color=["No Failure", "Failure"],
                color_discrete_map={"No Failure": "#2ecc71", "Failure": "#e74c3c"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Input Parameters")
            st.dataframe(input_df.style.highlight_max(axis=0, color='#f39c12'))

with tab2:
    st.header("Model Insights")
    
    col1, col2 = st.columns(2)

    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")

    # Ensure the feature names match (optional rename if needed)
    X_test_df = pd.DataFrame(X_test, columns=clean_feature_names)


    with col1:
        st.subheader("Global Feature Importance")
        
        sample_indices = np.random.choice(len(X_test), size=50, replace=False)
        sample_data = X_test[sample_indices]
        shap_values = shap_explainer.shap_values(sample_data)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, sample_data, feature_names=clean_feature_names, show=False)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Impact Direction")
        
        most_important_feature = np.argmax(np.abs(shap_values).mean(0))
        
        fig, ax = plt.subplots()
        shap.dependence_plot(
            most_important_feature,
            shap_values,
            sample_data,
            feature_names=clean_feature_names,
            show=False
        )
        st.pyplot(fig)

with tab3:
    st.header("Feature Analysis")
    
    st.subheader("Feature Relationships")
    
    # Replace with actual test data and labels from training phase
    X_test_df = pd.DataFrame(X_test, columns=clean_feature_names)
    
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("X-axis feature", original_feature_names)
    with col2:
        y_feature = st.selectbox("Y-axis feature", original_feature_names)
    
    # Clean feature names
    def clean_feat(f):
        return f.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_')
    
    fig = px.scatter(
        X_test_df,
        x=clean_feat(x_feature),
        y=clean_feat(y_feature),
        color=y_test,
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        title=f"{x_feature} vs {y_feature} colored by Failure Status"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Performance")
    
    st.subheader("Comparison of All Models")
    
    format_dict = {
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1-Score': '{:.2%}',
        'MCC score': '{:.2%}',
        'time to train': '{:.1f} s',
        'time to predict': '{:.1f} s',
        'total time': '{:.1f} s'
    }
    
    styled_performance = performance_df.style.format(format_dict).background_gradient(cmap='Blues')
    st.dataframe(styled_performance)
    
    st.subheader("ROC Curve Comparison")
    st.info("ROC curves would be displayed here if saved during model training")
    
    st.subheader("Confusion Matrix")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Predictive Maintenance Dashboard**  
*Using XGBoost for failure prediction*  
[GitHub Repository](#) | [Documentation](#)
""")


# streamlit run app.py
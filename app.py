import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.title("Sales Forecasting and Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    # Handling missing values
    st.write("### Handling Missing Values")
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    
    # Feature Engineering
    if "Outlet_Establishment_Year" in df.columns:
        df["Store_Age"] = 2025 - df["Outlet_Establishment_Year"]
    
    # Encoding categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Display unique values per column
    st.write("### Unique Values Per Column")
    unique_val_f_col = pd.DataFrame({
        "Column": df.columns,
        "Unique Count": [df[col].nunique() for col in df.columns]
    })
    st.dataframe(unique_val_f_col)
    
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Sales Forecasting Model
    st.write("### Sales Forecasting")
    target_col = "Item_Outlet_Sales"
    feature_cols = [col for col in numeric_df.columns if col != target_col]
    
    if target_col in df.columns:
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        df["Forecasted_Sales"] = model.predict(X)
        
        # Visualization
        st.write("### Actual vs Forecasted Sales")
        fig, ax = plt.subplots()
        ax.scatter(df["Item_Outlet_Sales"], df["Forecasted_Sales"], alpha=0.5)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Forecasted Sales")
        ax.set_title("Sales Prediction Performance")
        st.pyplot(fig)
        
        # User input for future predictions
        st.write("### Predict Future Sales")
        future_inputs = {}
        for col in feature_cols:
            future_inputs[col] = st.number_input(f"Enter value for {col}", float(df[col].mean()))
        
        future_df = pd.DataFrame([future_inputs])
        future_prediction = model.predict(future_df)
        st.write(f"Predicted Sales: {future_prediction[0]:.2f}")
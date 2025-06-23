# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import io

# App Configuration
st.set_page_config(page_title="Samhita Sync", layout="wide")

# Auth credentials
AUTH_USERNAME = "dmkc1998@gmail.com"
AUTH_PASSWORD = "Dmkc@1998"

# Session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Theme", ["Light", "Dark"])

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.experimental_rerun()

user_role = "admin" if st.session_state.username == AUTH_USERNAME else "user"
nav_options = ["Home", "EDA", "Data Cleaning"]
if user_role == "admin":
    nav_options.append("ML")
section = st.sidebar.radio("Go to", nav_options)

# File Upload
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
else:
    df = None

# App Sections
if section == "Home":
    st.title("📊 Welcome to Samhita Sync")
    if df is not None:
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())
    else:
        st.info("Upload a CSV or Excel file to get started.")

elif section == "EDA" and df is not None:
    st.header("📈 Exploratory Data Analysis")
    st.write("Shape:", df.shape)
    st.write("Data Types:", df.dtypes)
    st.write("Missing Values:", df.isnull().sum())
    st.write("Summary Statistics:")
    st.write(df.describe())

    col = st.selectbox("Select Column for Univariate Plot", df.columns)
    if pd.api.types.is_numeric_dtype(df[col]):
        st.subheader("Histogram")
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)
    else:
        st.subheader("Bar Chart")
        fig = px.bar(df[col].value_counts().reset_index(), x="index", y=col)
        st.plotly_chart(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, ax=ax)
    st.pyplot(fig)

elif section == "Data Cleaning" and df is not None:
    st.header("🧹 Data Cleaning")

    action = st.radio("Choose Cleaning Operation", ["Drop Nulls", "Fill Numeric Nulls (Mean)", "Fill Categorical Nulls (Mode)"])

    if action == "Drop Nulls":
        df.dropna(inplace=True)
    elif action == "Fill Numeric Nulls (Mean)":
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif action == "Fill Categorical Nulls (Mode)":
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

    st.success("Cleaning applied.")
    st.dataframe(df.head())

    # Export cleaned data
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    st.download_button("📥 Download Cleaned Data", data=buffer.getvalue(), file_name="cleaned_data.xlsx")

elif section == "ML" and user_role == "admin" and df is not None:
    st.header("🧠 Machine Learning")

    target = st.selectbox("Select Target Column", df.columns)
    features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target])

    model_type = st.selectbox("Choose Model", ["Linear Regression", "Random Forest Classifier", "KMeans Clustering"])

    if st.button("Train Model"):
        X = df[features]
        y = df[target]

        # Handle non-numeric columns
        X = pd.get_dummies(X)
        if model_type != "KMeans Clustering":
            y = pd.get_dummies(y).iloc[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            st.write("Model R² Score:", score)

        elif model_type == "Random Forest Classifier":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, preds))
            st.text(classification_report(y_test, preds))

        elif model_type == "KMeans Clustering":
            model = KMeans(n_clusters=3)
            model.fit(X)
            df["Cluster"] = model.labels_
            st.success("Clustering complete!")
            st.write(df.head())

# Footer
st.markdown("---")
st.caption("➤ Created by Manikanta Damacharla")

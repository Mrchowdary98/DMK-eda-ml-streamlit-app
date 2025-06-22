# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# Page config
st.set_page_config(page_title="EDA & ML App", layout="wide")
st.markdown("<h4 style='text-align: right; opacity: 0.6;'>Created by Manikanta Chowdary Damacharla</h4>", unsafe_allow_html=True)

# Theme toggle
mode = st.sidebar.radio("Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
        <style>
            .main, .stApp {
                background-color: #1e1e1e;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("📊 End-to-End EDA & ML App")

# File upload
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Data Cleaning
    st.sidebar.subheader("🧹 Data Cleaning")
    if st.sidebar.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)

    if st.sidebar.checkbox("Fill numeric NaNs with mean"):
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    if st.sidebar.checkbox("Fill categorical NaNs with mode"):
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Data Summary
    st.subheader("📋 Data Summary")
    st.write("**Shape:**", df.shape)
    st.write("**Column Types:**", df.dtypes)
    st.write("**Missing Values:**", df.isnull().sum())
    st.write("**Descriptive Statistics:**")
    st.dataframe(df.describe())

    # Visualizations
    st.subheader("📈 Visualizations")
    plot_type = st.selectbox("Choose Plot Type", ["Histogram", "Bar Chart", "Scatter Plot", "Correlation Heatmap"])

    if plot_type == "Histogram":
        col = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

    elif plot_type == "Bar Chart":
        col = st.selectbox("Select Column", df.select_dtypes(include='object').columns)
        fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col)
        st.plotly_chart(fig)

    elif plot_type == "Scatter Plot":
        x = st.selectbox("X-axis", df.select_dtypes(include=np.number).columns, key='scatter_x')
        y = st.selectbox("Y-axis", df.select_dtypes(include=np.number).columns, key='scatter_y')
        fig = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig)

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, ax=ax)
        st.pyplot(fig)

    # ML Section
    st.subheader("🧠 Machine Learning")
    ml_task = st.selectbox("Select ML Task", ["Linear Regression", "Random Forest Classification", "KMeans Clustering"])

    features = st.multiselect("Select Feature Columns", df.columns)
    target = st.selectbox("Select Target Column", df.columns)

    if features and target:
        X = df[features]
        y = df[target]

        if ml_task == "Linear Regression":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.write("**MSE:**", mean_squared_error(y_test, preds))

        elif ml_task == "Random Forest Classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.write("**Accuracy:**", accuracy_score(y_test, preds))
            st.text("Classification Report:\n" + classification_report(y_test, preds))

        elif ml_task == "KMeans Clustering":
            k = st.slider("Number of Clusters", 2, 10, 3)
            model = KMeans(n_clusters=k)
            clusters = model.fit_predict(X)
            df['Cluster'] = clusters
            st.write(df[['Cluster'] + features].head())

    # Export
    st.subheader("📤 Export Cleaned Data")
    if st.button("Download as Excel"):
        df.to_excel("cleaned_data.xlsx", index=False)
        with open("cleaned_data.xlsx", "rb") as f:
            st.download_button("Download Excel", f, "cleaned_data.xlsx")
else:
    st.info("Please upload a dataset to begin.")

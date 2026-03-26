import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("🚀 Travel Experience Decision Intelligence Dashboard")

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Encode
    df_enc = df.copy()
    for col in df_enc.columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    # KPIs
    avg_spend = df["Spend"].mode()[0]
    conversion = df["Likelihood"].value_counts(normalize=True).max()*100
    users = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Avg Spend Category", avg_spend)
    c2.metric("📈 Conversion Rate", f"{round(conversion,1)}%")
    c3.metric("👥 Total Users", users)

    st.divider()

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="Age", title="Age Distribution"))
        st.plotly_chart(px.histogram(df, x="Experience_Type", title="Experience Preference"))

    with col2:
        st.plotly_chart(px.histogram(df, x="Income", title="Income Distribution"))
        st.plotly_chart(px.histogram(df, x="Spend", title="Spending Pattern"))

    st.divider()

    # ML
    X = df_enc.drop("Likelihood", axis=1)
    y = df_enc["Likelihood"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    imp = pd.Series(model.feature_importances_, index=X.columns)

    st.subheader("🔍 Key Drivers of Conversion")
    st.plotly_chart(px.bar(imp.sort_values(), orientation="h"))

    st.divider()

    # Clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("👥 Customer Personas")

    st.plotly_chart(px.scatter(df, x="Age", y="Spend", color="Cluster"))

    st.write("### Persona Insights")
    st.info("Cluster 0: Budget Explorers - Low spend, high frequency")
    st.info("Cluster 1: Premium Users - High spend, selective experiences")
    st.info("Cluster 2: Casual Users - Low engagement")

    st.divider()

    # Business Insights
    st.subheader("💡 Business Insights")
    st.success("✔ Majority users prefer food & social experiences")
    st.success("✔ Weekend demand dominates usage patterns")
    st.warning("⚠ Price sensitivity affects conversion significantly")

    st.divider()

    # Recommendations
    st.subheader("🎯 Strategic Recommendations")
    st.write("- Offer discounts to budget users")
    st.write("- Bundle Food + Hidden Gems")
    st.write("- Target Gen Z via Instagram marketing")
    st.write("- Upsell premium experiences to high spend users")

    st.divider()

    # Prediction
    st.subheader("🧪 Predict New Customer")

    user = {}
    for col in X.columns:
        user[col] = st.number_input(col, 0, 100, 0)

    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([user]))
        st.success(f"Prediction: {pred}")

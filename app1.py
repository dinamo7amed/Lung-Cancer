import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(
    page_title="Lung Cancer Hospital System",
    layout="wide",
    page_icon="🫁"
)

df = pd.read_csv("survey lung cancer.csv")
model = joblib.load("model.pkl")

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #1f3b57;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}

.sub {
    color: gray;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "🏥 Navigation",
    ["Home", "🧠 Prediction", "📊 EDA"]
)

if page == "Home":

    st.markdown("<div class='main-title'>🫁 Lung Cancer Hospital Dashboard</div>", unsafe_allow_html=True)

    st.markdown("""
    ### Welcome Doctor!

    This AI system helps:
    - 🧠 Predict Lung Cancer risk
    - 📊 Analyze patient data
    - 📈 Visualize medical patterns
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Dataset", "Loaded")
    col2.metric("🧠 Model", "Random Forest")
    col3.metric("🏥 Status", "Active")

elif page == "🧠 Prediction":

    st.title("🧠 Patient Diagnosis System")

    features = df.drop("LUNG_CANCER", axis=1).columns
    input_data = {}

    st.sidebar.header("🧾 Patient Info")

    for col in features:
        if df[col].dtype == "object" or df[col].nunique() < 3:
            input_data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
        else:
            input_data[col] = st.sidebar.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    if st.button("🔍 Run Diagnosis"):

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("🧠 Risk Probability", f"{prob:.2f}%")
        col2.metric("📊 Status", "High Risk" if pred == 1 else "Low Risk")
        col3.metric("🏥 Case", "Under Review")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Cancer Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        if pred == 1:
            st.error("⚠️ HIGH RISK DETECTED — Please consult a doctor immediately.")
        else:
            st.success("LOW RISK — No immediate concern detected.")

elif page == "📊 EDA":

    st.title("Medical Data Analysis")

    # Age Distribution
    st.subheader("📈 Age Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["AGE"], kde=True, ax=ax1)
    st.pyplot(fig1)

    # Smoking vs Cancer
    st.subheader(" Smoking vs Cancer")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="SMOKING", hue="LUNG_CANCER", data=df, ax=ax2)
    st.pyplot(fig2)

    # Correlation Heatmap
    st.subheader(" Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)
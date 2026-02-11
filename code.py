import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="ESG AI App", layout="centered")

# -----------------------------
# 1. Create Synthetic Dataset
# -----------------------------

@st.cache_resource
def train_model():

    data = {
        "Carbon_Emission":[85,30,60,90,45,70,25,88,55,35],
        "Board_Diversity":[12,40,25,10,35,20,50,15,28,38],
        "Debt_Ratio":[0.70,0.30,0.55,0.80,0.40,0.65,0.25,0.75,0.50,0.35],
        "Renewable":[10,55,30,5,45,20,60,12,35,50],
        "Turnover":[22,10,18,25,12,20,8,24,15,11],
        "ESG_Risk":[2,0,1,2,0,1,0,2,1,0]  # 0=Low, 1=Medium, 2=High
    }

    df = pd.DataFrame(data)

    X = df.drop("ESG_Risk", axis=1)
    y = df["ESG_Risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(5,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=100, verbose=0)

    return model, scaler

model, scaler = train_model()

# -----------------------------
# 2. UI
# -----------------------------

st.title("🌍 ESG Risk Prediction & AI Sustainability Report")

st.subheader("Enter Company ESG Metrics")

carbon = st.slider("Carbon Emission Level (0-100)", 0, 100, 50)
diversity = st.slider("Board Diversity %", 0, 100, 30)
debt = st.slider("Debt Ratio (0-1)", 0.0, 1.0, 0.5)
renewable = st.slider("Renewable Energy Usage %", 0, 100, 30)
turnover = st.slider("Employee Turnover %", 0, 50, 15)

if st.button("Predict ESG Risk"):

    input_data = np.array([[carbon, diversity, debt, renewable, turnover]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)

    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    risk_label = risk_labels[predicted_class]

    st.subheader(f"📊 Predicted ESG Risk: {risk_label}")

    # -----------------------------
    # 3. Simple AI-Style Report
    # -----------------------------

    st.subheader("🤖 AI-Generated ESG Report")

    report = f"""
    This company shows a {risk_label} ESG risk profile.

    Key Metrics:
    - Carbon Emission: {carbon}
    - Board Diversity: {diversity}%
    - Debt Ratio: {debt}
    - Renewable Energy Usage: {renewable}%
    - Employee Turnover: {turnover}%

    """

    if risk_label == "High":
        report += """
        Major sustainability risks detected.
        High emissions and governance weaknesses require immediate strategic intervention.

        Recommended Actions:
        • Reduce carbon emissions by 20% over 3 years
        • Increase renewable energy usage above 50%
        • Improve board diversity beyond 35%
        • Strengthen ESG transparency
        """

    elif risk_label == "Medium":
        report += """
        Moderate ESG exposure.
        Sustainability practices exist but require improvement.

        Recommended Actions:
        • Gradually reduce emissions
        • Increase renewable energy investment
        • Enhance governance diversity
        """

    else:
        report += """
        Strong ESG positioning detected.
        Sustainable practices are well implemented.

        Recommended Actions:
        • Maintain renewable investment
        • Continue governance improvements
        • Focus on innovation-driven sustainability
        """

    st.write(report)

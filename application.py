import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go


ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


st.set_page_config(
    page_title="Forest Fire Index Prediction",
    page_icon="🔥",
    layout="centered",
)

st.markdown(
    """
    <h1 style="text-align:center; color:#f54242;">
        🔥 Forest Fire Weather Index (FWI) Prediction
    </h1>
    <p style="text-align:center; font-size:18px;">
        Enter environmental parameters to predict the Fire Weather Index
    </p>
    """,
    unsafe_allow_html=True,
)


# Input Section
st.subheader("🌡️ Input Environmental Values")

col1, col2 = st.columns(2)

with col1:
    Temperature = st.number_input("Temperature (°C)", value=25.0)
    RH = st.number_input("Relative Humidity (%)", value=50.0)
    Ws = st.number_input("Wind Speed (km/h)", value=10.0)
    Rain = st.number_input("Rain (mm)", value=0.0)

with col2:
    FFMC = st.number_input("FFMC", value=85.0)
    DMC = st.number_input("DMC", value=150.0)
    ISI = st.number_input("ISI", value=10.0)
    Classes = st.number_input("Classes", value=1.0)
    Region = st.number_input("Region", value=1.0)



# Prediction Button

if st.button("🚀 Predict FWI", use_container_width=True):

    # Prepare input
    new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    scaled_data = scaler.transform(new_data)

    # Predict
    prediction = ridge_model.predict(scaled_data)[0]

    
    # Fire Level Categorization
   
    if prediction < 10:
        level = "🟢 Low"
        gauge_color = "green"
    elif prediction < 20:
        level = "🟡 Moderate"
        gauge_color = "yellow"
    elif prediction < 30:
        level = "🟠 High"
        gauge_color = "orange"
    else:
        level = "🔴 Extreme"
        gauge_color = "red"

    
    # Fire Gauge Chart
    
    st.subheader("🔥 Fire Danger Gauge")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            title={'text': f"Fire Level: {level}", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 40]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 10], 'color': "green"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 30], 'color': "orange"},
                    {'range': [30, 40], 'color': "red"},
                ],
            }
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"🔥 Predicted FWI: **{prediction:.2f}** — Fire Danger Level: **{level}**")



# Footer

st.markdown("<br><hr><p style='text-align:center;'>Made with ❤️ by Keshav Tejuja</p>", unsafe_allow_html=True)

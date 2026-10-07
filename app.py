import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# --- 1. Synthetic Spatio-Temporal Model ---
class SpatialLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=1):
        super(SpatialLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. Streamlit Dashboard ---
st.set_page_config(page_title="Urban Spatio-Temporal Demand Agent", layout="wide")
st.title("🌆 Urban Traffic & Mobility Demand Forecaster")

st.sidebar.header("Model Parameters")
selected_zone = st.sidebar.selectbox("Select Traffic Zone ID", ["Zone_A", "Zone_B", "Zone_C", "Zone_D"])
time_horizon = st.sidebar.slider("Forecast Horizon (Hours)", 1, 24, 6)

# Generate Dummy Spatial Grid Data
zones = pd.DataFrame({
    'zone_id': ["Zone_A", "Zone_B", "Zone_C", "Zone_D"],
    'latitude': [33.4255, 33.4155, 33.4355, 33.4055],
    'longitude': [-111.9400, -111.9300, -111.9500, -111.9200],
    'base_demand': [120, 85, 200, 45]
})

# Run Prediction Simulation
model = SpatialLSTM()
sample_input = torch.randn(1, 10, 5) # Batch, Seq_Len, Features
predicted_demand = float(model(sample_input).detach().numpy()[0][0] * 50 + 100)

col1, col2 = st.subplots([1, 1])

with col1:
    st.subheader(f"Forecast for {selected_zone}")
    st.metric(label=f"Predicted Congestion Index ({time_horizon}h)", value=f"{predicted_demand:.1f} Units")
    
    # Time Series Plot
    time_series = [predicted_demand + np.sin(i)*15 for i in range(time_horizon)]
    st.line_chart(pd.DataFrame({"Projected Load": time_series}))

with col2:
    st.subheader("Spatial Zone Distribution")
    st.map(zones, latitude='latitude', longitude='longitude')

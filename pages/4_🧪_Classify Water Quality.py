import streamlit as st
import pandas as pd
import numpy as np
from joblib import load


st.set_page_config(
    page_title="Classify Water Quality",
    page_icon="🧪"
)

# Load data to extract features and ranges
wq = pd.read_csv("data/WQD.tsv", sep='\t')
wq = wq.rename(columns={wq.columns[4]: 'CO₂', wq.columns[5]: 'pH'})
params = wq.drop('Water Quality', axis=1).columns.to_list()

# Load best model and scaler
model = load("best_model.joblib")
scaler = load("scaler.joblib")


st.markdown("# Classify Your Own Water Data")

# Selection section
st.markdown("## Select Your Parameters")

input_params = []

for row in range(5):
    cols = st.columns(3)
    for i, col in enumerate(cols):
        index = row * 3 + i
        if index < len(params):
            with col:
                min_val = wq[params[index]].min()
                max_val = wq[params[index]].max()
                middle = min_val + max_val // 2
                value = st.number_input(f"{params[index]}:",
                                  min_value=min_val,
                                  max_value=max_val,
                                  value=middle)
                input_params.append(value)

# Normalize params
input_params = np.array(input_params).reshape(1, -1)
input_params_normalized = scaler.transform(input_params)

# Make classification
if st.button("Classify"):
    quality = model.predict(input_params_normalized)
    label_map = {2: "Poor", 1: "Good", 0: "Excellent"}
    st.success(f"Your water is {label_map[quality[0]]}")

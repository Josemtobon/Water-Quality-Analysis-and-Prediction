import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

wq = pd.read_csv("WQD.tsv", sep="\t")
wq = wq.rename(columns={wq.columns[4]: 'CO₂', wq.columns[5]: 'pH'})
wq['Quality'] = wq['Water Quality'].map({0: 'Execellent', 1: 'Good', 2: 'Poor'})

parameters = wq.columns[:-2]
parameter = st.selectbox(
    'Select a Parameter to Plot',
    parameters
)

fig, ax = plt.subplots()
sns.histplot(wq, x=parameter, hue='Quality', multiple='stack', ax=ax)

st.pyplot(fig)

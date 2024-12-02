import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


wq = pd.read_csv("WQD.tsv", sep="\t")
wq = wq.rename(columns={wq.columns[4]: 'CO₂', wq.columns[5]: 'pH'})
wq['Quality'] = wq['Water Quality'].map({0: 'Execellent', 1: 'Good', 2: 'Poor'})

parameters = wq.columns[:-2]

# Page start
st.title("Parameter Distribution by Water Quality")

parameter = st.selectbox(
    'Select a Parameter',
    parameters
)

qualities = st.pills("Water Quality", ["Execellent", "Good", "Poor"], selection_mode="multi")
filtered_wq = wq[wq['Quality'].isin(qualities)]

fig, ax = plt.subplots(figsize=(10, 8))
if len(qualities) > 1:
    sns.histplot(filtered_wq, x=parameter, hue='Quality', fill=True, ax=ax, palette='Set2')
    ax.legend(title='Water Quality', labels=wq['Quality'].unique()[::-1])
else:
    sns.histplot(filtered_wq, x=parameter, fill=True, ax=ax)

if qualities:
    st.pyplot(fig)


st.header('Parameter Resume')

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Excellent')
    st.write(wq[parameter][wq['Quality'] == 'Execellent'].describe())

with col2:
    st.subheader('Good')
    st.write(wq[parameter][wq['Quality'] == 'Good'].describe())

with col3:
    st.subheader('Poor')
    st.write(wq[parameter][wq['Quality'] == 'Poor'].describe())

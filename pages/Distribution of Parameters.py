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


st.header("Kernel Density Stimation")

fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(wq, x=parameter, hue='Quality', fill=True, ax=ax, palette='Set2')
ax.legend(title='Water Quality', labels=wq['Quality'].unique()[::-1])

st.pyplot(fig)

st.header("Boxplot")

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=wq, x='Quality', y=parameter, hue='Quality', ax=ax, palette='Set2')

ax.set_xlabel("Water Quality")
ax.set_ylabel(parameter)

st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

wq = pd.read_csv("WQD.tsv", sep="\t")
wq = wq.rename(columns={wq.columns[4]: 'CO₂', wq.columns[5]: 'pH'})
wq['Quality'] = wq['Water Quality'].map({0: 'Execellent', 1: 'Good', 2: 'Poor'})

parameters = wq.columns[:-2]


correlation_matrix = wq[parameters].corr()

st.header("Parameter Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', ax=ax)

st.pyplot(fig)


feature_correlations = wq[parameters].corrwith(wq['Water Quality'])

st.header('Feature Correlation with Water Quality')

fig, ax = plt.subplots(figsize=(10, 8))
feature_correlations.sort_values(ascending=False).plot(kind='bar', ax=ax)
ax.set_title("Correlation of Features with Water Quality")
ax.set_ylabel("Correlation Coefficient")
ax.set_xlabel("Features")

st.pyplot(fig)

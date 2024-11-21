import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv('performance_results.tsv', sep='\t')
st.dataframe(results.head())

fig, axs = plt.subplots(3, 1, figsize=(8, 14))
sns.boxplot(results, x='Precision', y='Model', hue='Model', ax=axs[0])
sns.boxplot(results, x='Recall', y='Model', hue='Model', ax=axs[1])
sns.boxplot(results, x='F1', y='Model', hue='Model', ax=axs[2])
plt.tight_layout()

st.pyplot(fig)

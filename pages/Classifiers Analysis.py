import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv('performance_results.tsv', sep='\t')
st.table(results.head())

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.image("confusion_matrix_K-Nearest Neighbors.png")

with col2:
    st.image("confusion_matrix_Random Forest.png")

with col3:
    st.image("confusion_matrix_SVM.png")

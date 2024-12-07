import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv('performance_results.tsv', sep='\t')
st.table(results.head())

import streamlit as st
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

wq = pd.read_csv("WQD.tsv", sep="\t")
wq = wq.rename(columns={wq.columns[4]: 'CO₂', wq.columns[5]: 'pH'})
wq['Quality'] = wq['Water Quality'].map({0: 'Execellent', 1: 'Good', 2: 'Poor'})

parameters = wq.columns[:-2]

# Normality test
# DataFrame to store Shapiro-Wilk results
shapiro_results = pd.DataFrame(columns=['P-Value', 'Normally Distributed'])

for column in parameters:
    stat, p_value = stats.shapiro(wq[column])

    shapiro_results.loc[column] = [str(p_value), int(p_value < .05)]

# To show table results on page
st.subheader("Shapiro-Wilk Results for Each Parameter")
st.dataframe(shapiro_results)


correlation_matrix = pd.DataFrame()
p_value_matrix = pd.DataFrame()

for col1 in parameters:
    for col2 in parameters:
        corr, p_value = stats.pearsonr(wq[col1], wq[col2])
        correlation_matrix.loc[col1, col2] = corr
        p_value_matrix.loc[col1, col2] = p_value

significant_correlation = correlation_matrix.where(p_value_matrix < .05, other=0)


anova_results = pd.DataFrame(columns=['P-Value', 'Reject Null Hypothesis'])
tukey_results = pd.DataFrame(columns=[])

for parameter in parameters:
    groups = [group[parameter].values for name, group in wq.groupby('Water Quality')]

    _, p_value = stats.f_oneway(*groups)
    if p_value < .05:
        sd = 'Yes'
    else:
        sd = 'No'

    anova_results.loc[parameter] = str(p_value), sd

    tukey = pairwise_tukeyhsd(endog=wq[parameter],
                               groups=wq['Quality'],
                               alpha=0.05)

st.dataframe(anova_results)


st.subheader("Parameters with Significant Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(significant_correlation, annot=True, ax=ax)

st.pyplot(fig)

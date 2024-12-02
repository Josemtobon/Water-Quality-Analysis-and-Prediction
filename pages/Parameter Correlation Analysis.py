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
shapiro_results = pd.DataFrame(columns=['Parameter', 'Group', 'P-Value', 'Normally Distributed'])

for column in parameters:
    for name, group in wq.groupby('Quality'):
        _, p_value = stats.shapiro(group[column])
        shapiro_results.loc[len(shapiro_results)] = [column, name, str(p_value), str(p_value < .05)]

# To show table results on page
st.subheader("Shapiro-Wilk Results for Each Parameter")
st.dataframe(shapiro_results)


# Homoscedasticity test
# DataFrame to store Bartlett results
bartlett_results = pd.DataFrame(columns=['Parameter', 'P-Value', 'Homoscedasticity'])

for column in parameters:
    groups = [group[column].values for name, group in wq.groupby('Water Quality')]
    _, p_value = stats.bartlett(*groups)
    bartlett_results.loc[len(bartlett_results)] = [column, str(p_value), str(p_value < .05)]

# To show table results on page
st.subheader("Bartlett Results for Each Parameter")
st.dataframe(bartlett_results)


anova_results = pd.DataFrame(columns=['Parameter', 'P-Value', 'significant Difference'])
tukey_results = pd.DataFrame(columns=[])

for parameter in parameters:
    groups = [group[parameter].values for name, group in wq.groupby('Water Quality')]

    _, p_value = stats.f_oneway(*groups)

    anova_results.loc[len(anova_results)] = [parameter, str(p_value), str(p_value < .05)]

    tukey = pairwise_tukeyhsd(endog=wq[parameter],
                               groups=wq['Quality'],
                               alpha=0.05)

st.dataframe(anova_results)

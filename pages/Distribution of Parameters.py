import streamlit as st
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns


wq = pd.read_csv("WQD.tsv", sep="\t")
wq = wq.rename(columns={wq.columns[4]: 'CO₂', wq.columns[5]: 'pH'})
wq['Quality'] = wq['Water Quality'].map({0: 'Excellent', 1: 'Good', 2: 'Poor'})

parameters = wq.columns[:-2]

# Page start
st.title("Parameter Distribution by Water Quality")

parameter = st.selectbox(
   'Select a Parameter',
    parameters
)

qualities = st.pills("Water Quality", ["Excellent", "Good", "Poor"], selection_mode="multi", default=["Excellent", "Good", "Poor"])
filtered_wq = wq[wq['Quality'].isin(qualities)]

fig, ax = plt.subplots(figsize=(10, 8))
if len(qualities) > 1:
    sns.histplot(filtered_wq, x=parameter, hue='Quality', fill=True, ax=ax, palette='Set2')
    ax.legend(title='Water Quality', labels=wq['Quality'].unique()[::-1])
else:
    sns.histplot(filtered_wq, x=parameter, fill=True, ax=ax)

if qualities:
    st.pyplot(fig)


st.header(f'{parameter} Resume')

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Excellent')
    st.table(wq[parameter][wq['Quality'] == 'Excellent'].describe())

with col2:
    st.subheader('Good')
    st.table(wq[parameter][wq['Quality'] == 'Good'].describe())

with col3:
    st.subheader('Poor')
    st.table(wq[parameter][wq['Quality'] == 'Poor'].describe())


st.header(f'{parameter} Resume')

st.write("For excellent water quality, the data shows well-defined ranges with a lower standard deviation compared to other water quality categories. Good quality is distributed near the edges of the range limits for excellent quality, while poor quality exhibits the highest standard deviation and includes atypical values.")


rows = []

for param in parameters:
    summary_row = {'Parameter': param}

    for quality, group in wq.groupby('Quality'):
        min_val = group[param].min()
        max_val = group[param].max()
        interval = f"{min_val} - {max_val}"

        summary_row[f"{quality} Quality"] = interval

    optimal_value = wq.loc[wq['Quality'] == 'Excellent', param].mean()
    summary_row['Optimal Value'] = optimal_value

    rows.append(summary_row)

summary_table = pd.DataFrame(rows)

st.table(summary_table.set_index('Parameter'))

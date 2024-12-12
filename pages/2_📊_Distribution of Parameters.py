import streamlit as st
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Distribution of Parameters",
    page_icon="📊"
)

wq = pd.read_csv("data/WQD.tsv", sep="\t")
wq = wq.rename(columns={wq.columns[4]: 'CO₂', wq.columns[5]: 'pH'})
wq['Quality'] = wq['Water Quality'].map({0: 'Excellent', 1: 'Good', 2: 'Poor'})

parameters = wq.columns[:-2]

# Page start
st.markdown("# Parameter Distribution by Water Quality")

parameter = st.selectbox(
   'Select a Parameter',
    parameters
)

qualities = st.pills("Water Quality", ["Excellent", "Good", "Poor"],
                     selection_mode="multi",
                     default=["Excellent", "Good", "Poor"])
filtered_wq = wq[wq['Quality'].isin(qualities)]


fig, ax = plt.subplots(figsize=(10, 6))

if len(qualities) > 1:
    sns.histplot(filtered_wq, x=parameter, hue='Quality', fill=True, ax=ax, palette='Set2')
    ax.legend(title='Water Quality', labels=wq['Quality'].unique()[::-1])
else:
    sns.histplot(filtered_wq, x=parameter, fill=True, ax=ax)

if qualities:
    st.pyplot(fig)


st.markdown('## Parameter Resume')
st.markdown(
    """
    This section provides a statistical summary for the selected parameter data
    categorized by water quality classifications: Excellent, Good, and Poor.
    For each classification, key descriptive statistics such as the number of
    observations (count), average value (mean), variability
    (standard deviation, std), and the range of values (min, max, and quartiles)
    are presented. These statistics highlight the distribution and variability of
    parameters within each quality category.
    """
)

param = st.select_slider("Parameter", parameters, label_visibility='collapsed')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('### Excellent')
    st.table(wq[param][wq['Quality'] == 'Excellent'].describe())

with col2:
    st.markdown('### Good')
    st.table(wq[param][wq['Quality'] == 'Good'].describe())

with col3:
    st.markdown('### Poor')
    st.table(wq[param][wq['Quality'] == 'Poor'].describe())

st.markdown(
"""
For excellent water quality, the data shows well-defined ranges with a
lower standard deviation compared to other water quality categories.
Good quality is distributed near the edges of the range limits for
excellent quality, while poor quality exhibits the highest standard
deviation and includes atypical values.
"""
)


st.markdown('## Water Quality Ranges')
st.markdown(
    """
    This table summarizes the ranges of various water quality parameters grouped
    by their respective classifications: Excellent Quality, Good Quality, and
    Poor Quality. Each range represents the minimum and maximum values observed
    for each parameter within a specific quality category.
    """
)

rows = []

for param in parameters:
    summary_row = {'Parameter': param}

    for quality, group in wq.groupby('Quality'):
        min_val = group[param].min()
        max_val = group[param].max()
        interval = f"{min_val} - {max_val}"

        summary_row[f"{quality} Quality"] = interval

    rows.append(summary_row)

summary_table = pd.DataFrame(rows)

st.table(summary_table.set_index('Parameter'))

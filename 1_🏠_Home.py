import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Home",
    page_icon="🏠"
)

st.title("Fish Pond Water Quality Dashboard")
st.write("""
    This dashboard is based on the **Aquaculture - Water Quality Dataset** (Veeramsetty, Venkataramana; Arabelli, Rajeshwarrao; Bernatin, T., 2024).
    The goal of this analysis is to assess the quality of water in fish ponds by examining the relationship between various physicochemical properties (e.g., Temperature, pH, Alkalinity, etc.) and the overall water quality classification.
""")


st.subheader("Dataset Description")
st.write("""
This dataset is useful for training and testing deep learning models developed to assess the quality of water in fish ponds based on parameters like Temperature, Turbidity, Dissolved Oxygen, Biochemical Oxygen Demand (BOD), CO₂, pH, Alkalinity, Hardness, Calcium, Ammonia, Nitrite, Phosphorus, H₂S, and Plankton. The quality of water in fish ponds is classified into three categories: Excellent, Good, and Poor quality.

This dataset was prepared based on the threshold values of each input feature, which represent the acceptable range, desirable range, and stress range for the growth of fish in ponds. The dataset consists of three different water quality samples: excellent quality is represented by 0, good quality is labeled with 1, and poor quality is labeled with 2.

The dataset contains a total of 4,300 samples, with 1,500 poor quality water samples, 1,400 excellent quality water samples, and 1,400 good quality water samples. It includes 14 input features and one output label column.
""")

wq = pd.read_csv("data/WQD.tsv", sep="\t")
wq["Quality"] = wq["Water Quality"].map({0: "Excellent", 1: "Good", 2: "Poor"})


# Use transparent background
plt.rcParams.update({
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "savefig.facecolor": "none",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "axes.edgecolor": "white",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelcolor": "white",
    "grid.color": "white",
    "legend.facecolor": "none",
})

# fig and ax to plot bars
fig, ax = plt.subplots(figsize=(8, 4))

sns.histplot(wq, x="Quality", color="#FF4B4B", edgecolor='none', shrink=.5, ax=ax)

for patch in ax.patches:
    height = patch.get_height()
    width = patch.get_width()
    x = patch.get_x() + width / 2
    y = height + 0.1
    ax.annotate(f'{int(height)}', (x, y), ha='center', va='bottom')

ax.set_xlabel("Water Quality")
ax.set_ylabel("Number of Samples")

with st.container():
    st.subheader("Distribution of Water Quality Categories")
    st.pyplot(fig)


data_summary = pd.DataFrame(columns=["Water Quality", "Total Records",
                                     "Missing Values (count)", "Missing Values (%)",
                                     "Duplicate Records (count)", "Duplicate Records (%)"])

for quality, group in wq.groupby('Quality'):
    total = group.shape[0]
    missing_count = group.isnull().sum().sum()
    missing_percentage = missing_count / total * 100
    duplicate_count = group.duplicated().sum()
    duplicate_percentage = duplicate_count / total * 100

    data_summary.loc[len(data_summary)] = [quality, total, missing_count, missing_percentage,
                                 duplicate_count, duplicate_percentage]

# Apply styling to hide the index
styled_table = data_summary.style.hide(axis="index").to_html()

# Center the table using CSS
table_html = f"""
<div style="display: flex; justify-content: center; align-items: center;">
    {styled_table}
</div>
"""

st.subheader("Missing Values and Duplicate Data")
st.markdown(table_html, unsafe_allow_html=True)


st.subheader("Dataset Citation")
st.write("""
Veeramsetty, Venkataramana; Arabelli, Rajeshwarrao; Bernatin, T. (2024), “Aquaculture - Water Quality Dataset”, Mendeley Data, V1, doi: 10.17632/y78ty2g293.1
""")

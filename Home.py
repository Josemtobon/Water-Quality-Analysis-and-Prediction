import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

wq = pd.read_csv("WQD.tsv", sep="\t")
wq["Quality"] = wq["Water Quality"].map({0: "Excellent", 1: "Good", 2: "Poor"})

fig, ax = plt.subplots()
sns.histplot(wq, x="Quality", color="#FF4B4B", shrink=.8, ax=ax)

ax.set_title("Distribution of Water Quality Categories")
ax.set_xlabel("Water Quality")
ax.set_ylabel("Number of Samples")

st.pyplot(fig)


st.subheader("Dataset Citation")
st.write("""
Veeramsetty, Venkataramana; Arabelli, Rajeshwarrao; Bernatin, T. (2024), “Aquaculture - Water Quality Dataset”, Mendeley Data, V1, doi: 10.17632/y78ty2g293.1
""")

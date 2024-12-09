import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.markdown("""
### Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** with **StratifiedKFold** to optimize **Random Forest**, **SVM**, and **KNN** classifiers. The goal was to find the best combination of parameters to maximize accuracy (`accuracy`) using cross-validation.

#### Parameters Evaluated
- **Random Forest**: `n_estimators`, `max_depth`, `bootstrap`.
- **SVM**: `C` and `kernel` (`linear`, `poly`, `rbf`).
- **KNN**: Range of values for `n_neighbors` from 1 to 29.

This process identified the best hyperparameters for each model, improving their performance and generalization ability.
""")

st.markdown("#### Best Hyperparameters Selected")
st.markdown(
"""
**For Random Forest:** The model uses 300 decision trees with no restriction on maximum depth. Bootstrap sampling was enabled to ensure robust training performance.

**For SVM:** The model uses an RBF kernel with 100 support vectors to optimize its decision boundary.

**For KNN:** A single nearest neighbor was used, indicating the simplicity of the classification boundary for this task.
"""
)

col1, col2 = st.columns(2)
col3, col4, col5 = st.columns([.25, .5, .25])

with col1:
    st.image("confusion_matrix_K-Nearest Neighbors.png")

with col2:
    st.image("confusion_matrix_Random Forest.png")

with col4:
    st.image("confusion_matrix_SVM.png")


results = pd.read_csv('performance_results.tsv', sep='\t')
styled_results = results.style.hide(axis='index').to_html()

table_html = f"""
<div style="display: flex; justify-content: center; align-items: center;">
    {styled_results}
</div>
"""

st.markdown(table_html, unsafe_allow_html=True)

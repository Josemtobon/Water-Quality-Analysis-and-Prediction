import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Analysis of Classifiers",
    page_icon="⚙️"
)

st.markdown("""
## Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** with **StratifiedKFold**
to optimize **Random Forest**, **SVM**, and **KNN** classifiers. The goal was to
find the best combination of parameters to maximize accuracy (`accuracy`) using
cross-validation.

### Parameters Evaluated
- **Random Forest**: `n_estimators`, `max_depth`, `bootstrap`.
- **SVM**: `C` and `kernel` (`linear`, `poly`, `rbf`).
- **KNN**: Range of values for `n_neighbors` from 1 to 29.

This process identified the best hyperparameters for each model, improving their
performance and generalization ability.
""")

st.markdown("### Best Hyperparameters Selected")
st.markdown(
"""
**For Random Forest:** The model uses 300 decision trees with no restriction on
maximum depth. Bootstrap sampling was enabled to ensure robust training performance.

**For SVM:** The model uses an RBF kernel with 100 support vectors to optimize its
decision boundary.

**For KNN:** A single nearest neighbor was used, indicating the simplicity of the
classification boundary for this task.
"""
)


st.markdown("## Model Performance Evaluation")
st.markdown("### Confusion Matrices")
st.markdown(
"""
Below are the confusion matrices for each classifier, summarizing their
performance in classifying water quality levels.
"""
)

col1, col2 = st.columns(2)
col3, col4, col5 = st.columns([.25, .5, .25])

with col1:
    st.markdown("#### K-Nearest Neighbors")
    st.image("images/confusion_matrix_K-Nearest Neighbors.png")

with col2:
    st.markdown("#### Random Forest")
    st.image("images/confusion_matrix_Random Forest.png")

with col4:
    st.markdown("#### Support Vector Machine")
    st.image("images/confusion_matrix_SVM.png")

st.markdown(
"""
**Random Forest** shows an overall better performance, with few false positives
and false negatives for all classes, though all are able to correctly identify
"Excellent" instances.
"""
)


st.markdown("### Receiver Operating Characteristic (ROC) Analysis")
st.markdown(
"""
ROC curves and the corresponding AUC scores were generated to evaluate the
classifiers’ ability to distinguish between classes. Higher AUC values indicate
better discriminatory performance.
"""
)

col6, col7 = st.columns(2)
col8, col9, col1 = st.columns([.25, .5, .25])

with col6:
    st.markdown("#### K-Nearest Neighbors")
    st.image("images/roc_curve_K-Nearest Neighbors.png")

with col7:
    st.markdown("#### Random Forest")
    st.image("images/roc_curve_Random Forest.png")

with col9:
    st.markdown("#### Support Vector Machine")
    st.image("images/roc_curve_SVM.png")

st.markdown(
"""
**Random Forest** demonstrates near-perfect capability to distinguish between
positive and negative classess.
However, all models efectively classify positive and negative instances for
"Excellent" quality.
"""
)


results = pd.read_csv('data/performance_results.tsv', sep='\t')
styled_results = results.style.hide(axis='index').to_html()

table_html = f"""
<div style="display: flex; justify-content: center; align-items: center;">
    {styled_results}
</div>
"""

st.markdown("### Performance Metrics for Random Forest, KNN, and SVM Classifiers")
st.markdown(
"""
The table below summarizes the performance metrics of Random Forest, KNN, and SVM
classifiers. These metrics include accuracy, precision, recall, F1 score, and ROC-AUC:
"""
)
st.markdown(table_html, unsafe_allow_html=True)
st.markdown(
"""
#### Insights

- The **Random Forest classifier** achieved the highest accuracy and ROC-AUC
score, demonstrating excellent performance and minimal overfitting.
- The **SVM model** showed strong performance, particularly in its ROC-AUC score.
- The **KNN model**, showed acceptable performance with a slightly lower
accuracy and AUC score.
"""
)

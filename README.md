# Water Quality Analysis and Prediction

This project uses the dataset [**Aquaculture - Water Quality Dataset** 14 (Veeramsetty, Venkataramana; Arabelli, Rajeshwarrao; Bernatin, T., 2024).](https://data.mendeley.com/datasets/y78ty2g293/1)
to train and test three different classifiers: Random Forest, Support Vector Machine and K-Nearest Neighbors.
Finally the best model is used to classify real samples of water.

---

├── best_model.joblib                  # Trained model with the best overall performance
├── clf_eval.py                        # Script to train and test models
├── data
│   ├── params.tsv                     # Parameters choose for models
│   ├── performance_results.tsv        # Performance metrics
│   └── WQD.tsv                        # Dataset file
├── Home.py                            # Main Streamlit app
├── images                             # Confusion matrices and ROC curve visualizations
│   ├── confusion_matrix_K-Nearest Neighbors.png
│   ├── confusion_matrix_Random Forest.png
│   ├── confusion_matrix_SVM.png
│   ├── roc_curve_K-Nearest Neighbors.png
│   ├── roc_curve_Random Forest.png
│   └── roc_curve_SVM.png
├── pages                              # Streamlit pages
│   ├── 2_📊_Distribution of Parameters.py
│   ├── 3_⚙️-_Analysis_of_Classifiers.py
│   └── 4_🧪_Classify Water Quality.py
├── requirements.txt                   # Dependecies
└── scaler.joblib                      # Preprocessing scaler file

---

# Online Version
You can access the app using this [Streamlit link](aquanalysis.streamlit.app).

# How to Run the Project Locally

1. Clone Repo
`
git clone https://github.com/Josemtobon/Water-Quality-Analysis-and-Prediction.git
cd Water-Quality-Analysis-and-Prediction
`

2. Install Dependecies
`
pip install -r requirements.txt
`

3. Run the Streamlit App
`
streamlit run Home.py
`

---

# Acknowledgements

Dataset: [Aquaculture - Water Quality Dataset](https://data.mendeley.com/datasets/y78ty2g293/1)
by Veeramsetty, Venkataramana; Arabelli, Rajeshwarrao; Bernatin, T. (2024).

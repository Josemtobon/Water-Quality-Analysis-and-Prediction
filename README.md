# Water Quality Analysis and Prediction

This project uses the dataset [**Aquaculture - Water Quality Dataset**](https://data.mendeley.com/datasets/y78ty2g293/1)  
(Veeramsetty, Venkataramana; Arabelli, Rajeshwarrao; Bernatin, T., 2024) to train and test three different classifiers:  
**Random Forest**, **Support Vector Machine (SVM)**, and **K-Nearest Neighbors (KNN)**. Finally, the best-performing model is used to classify real samples of water.

---

## Online Version

You can access the app directly using this [Streamlit link](https://aquanalysis.streamlit.app).

---

## How to Run the Project Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Josemtobon/Water-Quality-Analysis-and-Prediction.git
   cd Water-Quality-Analysis-and-Prediction
   ```


2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```


3. **Run the Streamlit App**

   ```bash
   streamlit run Home.py
   ```

---

## Repository Structure

```
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
```

---

## Acknowledgements

Dataset: [Aquaculture - Water Quality Dataset](https://data.mendeley.com/datasets/y78ty2g293/1)
by Veeramsetty, Venkataramana; Arabelli, Rajeshwarrao; Bernatin, T. (2024).

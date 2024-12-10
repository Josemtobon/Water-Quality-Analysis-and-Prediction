import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, RocCurveDisplay, auc
)
import matplotlib.pyplot as plt
from matplotlib import colormaps


# Water Quality Dataset
wq = pd.read_csv('WQD.tsv', sep='\t')

# Defining variables
X = wq.drop(columns=['Water Quality'])
y = wq['Water Quality'].values

#Normalizing features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Selection of best classifiers
# Cross Validation splitter to select best model
random_kf = StratifiedKFold(shuffle=True, random_state=6)

# Parameters to assess in Random Search
rf_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'bootstrap': [True, False]
}

svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf']
}

knn_params = {
    'n_neighbors': list(range(1, 30))
}


# GridSearchCV instances
rf_gridsearch = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=rf_params,
    scoring='accuracy',
    cv=random_kf,
    n_jobs=-1
)

svm_gridsearch = GridSearchCV(
    estimator=SVC(probability=True),
    param_grid=svm_params,
    scoring='accuracy',
    cv=random_kf,
    n_jobs=1
)

knn_gridsearch = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_params,
    scoring='accuracy',
    cv=random_kf,
    n_jobs=1
)

# Grid Search initialize
rf_gridsearch.fit(X_train, y_train)
svm_gridsearch.fit(X_train, y_train)
knn_gridsearch.fit(X_train, y_train)

# Save best params as tsv
rf_best_params = rf_gridsearch.best_params_
svm_best_params = svm_gridsearch.best_params_
knn_best_params = knn_gridsearch.best_params_

params_df = pd.DataFrame([
    {"Model": "Random Forest", **rf_best_params},
    {"Model": "SVM", **svm_best_params},
    {"Model": "KNN", **knn_best_params}
])

params_df.to_csv('params.tsv', sep='\t', index=False)

# Best stimators
rf_best_estimator = rf_gridsearch.best_estimator_
svm_best_estimator = svm_gridsearch.best_estimator_
knn_best_estimator = knn_gridsearch.best_estimator_

# Training selected models to compare
classifiers = {
    'SVM': svm_best_estimator,
    'Random Forest': rf_best_estimator,
    'K-Nearest Neighbors': knn_best_estimator
}

# df to store testing results
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC Score'])

# Color map for ROC curves and theme
color_map = colormaps['YlOrRd']
plt.style.use('dark_background')

# Iterate over each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovo', average='macro')

    # Make confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Excellent', 'Good', 'Poor'])
    disp.plot(cmap='YlOrRd', xticks_rotation='vertical')

    # Save the plot as aimages/n image file
    plt.savefig(f"images/confusion_matrix_{name}.png", bbox_inches='tight', transparent=True)
    plt.close()

    plt.figure(figsize=(8, 6))
    for class_label, i in {'Excellent': 0, 'Good': 1, 'Poor': 2}.items():
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_probs[:, i])
        auc_score = auc(fpr, tpr)
        color = color_map(i / 3)
        plt.plot(fpr, tpr, label=f"{class_label} (AUC: {auc_score:.2f})", color=color)

    plt.plot([0, 1], [0, 1], '--', label='Chance', color='White')
    plt.title(f"ROC Curve for {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", framealpha=0)

    # Save the ROC curve plot as an image file
    plt.savefig(f"images/roc_curve_{name}.png", bbox_inches='tight', transparent=True)
    plt.close()

    results.loc[len(results)] = [name, accuracy, precision, recall, f1, roc_auc]

# Store testing results as a tsv
results.to_csv('performance_results.tsv', index=False, sep='\t')

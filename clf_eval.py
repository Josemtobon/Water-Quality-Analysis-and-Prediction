import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


# RandomizedSearchCV instances
rf_random_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=rf_params,
    scoring='accuracy',
    cv=random_kf,
    n_jobs=-1
)

svm_random_search = GridSearchCV(
    estimator=SVC(),
    param_grid=svm_params,
    scoring='accuracy',
    cv=random_kf,
    n_jobs=1
)

knn_random_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_params,
    scoring='accuracy',
    cv=random_kf,
    n_jobs=1
)

rf_random_search.fit(X_train, y_train)
svm_random_search.fit(X_train, y_train)
knn_random_search.fit(X_train, y_train)

# Best stimators
rf_best_estimator = rf_random_search.best_estimator_
svm_best_estimator = svm_random_search.best_estimator_
knn_best_estimator = knn_random_search.best_estimator_

# Training selected models to compare
classifiers = {
    'SVM': svm_best_estimator,
    'Random Forest': rf_best_estimator,
    'K-Nearest Neighbors': knn_best_estimator
}

results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Excellent', 'Good', 'Poor'])
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    disp.ax_.set_title(f"Confusion Matrix for {name}")

    # Save the plot as an image file
    plt.savefig(f"confusion_matrix_{name}.png", bbox_inches='tight')
    plt.close()

    results.loc[len(results)] = [name, accuracy, precision, recall, f1]

results.to_csv('performance_results.tsv', index=False, sep='\t')

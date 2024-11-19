import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Water Quality Dataset
wq = pd.read_csv('WQD.tsv', sep='\t')

# Defining variables
X = wq.drop(columns=['Water Quality'])
y = wq['Water Quality'].values

#Normalizing features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

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
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=rf_params,
    n_iter=10,
    scoring='f1_macro',
    cv=random_kf,
    random_state=6,
    n_jobs=-1
)

svm_random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=svm_params,
    n_iter=10,
    scoring='f1_macro',
    cv=random_kf,
    random_state=6,
    n_jobs=1
)

knn_random_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=knn_params,
    n_iter=10,
    scoring='f1_macro',
    cv=random_kf,
    random_state=6,
    n_jobs=1
)

rf_random_search.fit(X, y)
svm_random_search.fit(X, y)
knn_random_search.fit(X, y)

# Best hyperparameters variables
rf_best_params = rf_random_search.best_params_
svm_best_params = svm_random_search.best_params_
knn_best_params = knn_random_search.best_params_

# Training selected models to compare
classifiers = {
    'SVM': SVC(**svm_best_params, random_state=6),
    'Random Forest': RandomForestClassifier(**rf_best_params, random_state=6),
    'K-Nearest Neighbors': KNeighborsClassifier(**knn_best_params)
}

kf = StratifiedKFold(n_splits=100, shuffle=True, random_state=6)
results = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1'])

for name, clf in classifiers.items():
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results.loc[len(results)] = [name, precision, recall, f1]

results.to_csv('performance_results.tsv', sep='\t')

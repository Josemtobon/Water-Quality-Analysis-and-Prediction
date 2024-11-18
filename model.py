import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from numpy import mean

wq = pd.read_csv('WQD.tsv', sep='\t')
parameters = [parameter for parameter in wq.columns if parameter != 'Water Quality']
normalized_wq = wq[parameters].apply(lambda col: (col - col.min()) / (col.max() - col.min()))

X = wq[parameters]
y = wq['Water Quality']

kf = StratifiedKFold(n_splits=100, shuffle=True, random_state=6)
clf = svm.SVC()

f1 = []
precision = []
recall = []

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1.append(f1_score(y_test, y_pred, average='macro'))
    precision.append(precision_score(y_test, y_pred, average='macro'))
    recall.append(precision_score(y_test, y_pred, average='macro'))

print(mean(f1))
print(mean(precision))
print(mean(recall))

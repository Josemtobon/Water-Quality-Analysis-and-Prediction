import pandas as pd
from sklearn.model_selection import StratifiedKFold

wq = pd.read_csv('WQD.tsv', sep='\t')
parameters = [parameter for parameter in wq.columns if parameter != 'Water Quality']
normalized_wq = wq[parameters].apply(lambda col: (col - col.min()) / (col.max() - col.min()))

X = wq[parameters]
y = wq['Water Quality']

kf = StratifiedKFold(n_splits=100, shuffle=True, random_state=6)

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import numpy as np

df_description = pd.read_csv('./data/data_definition.txt', sep='\t')

df_train_genba = pd.read_csv('./data/train_genba.tsv', sep='\t')
df_train_goto = pd.read_csv('./data/train_goto.tsv', sep='\t')

df_train = df_train_goto.merge(df_train_genba, on='pj_no', how='left')
df_train.drop('id', axis=1, inplace=True)

df_test_genba = pd.read_csv('./data/test_genba.tsv', sep='\t')
df_test_goto = pd.read_csv('./data/test_goto.tsv', sep='\t')

df_test = df_test_goto.merge(df_test_genba, on='pj_no', how='left')
df_test.drop('id', axis=1, inplace=True)

categorical_features = list(df_description[df_description['データ型'] != '数値']['項目名'])
continue_features = list(df_description[df_description['データ型'] == '数値']['項目名'])
objective = 'keiyaku_pr'

for col in categorical_features:
    dt = df_train[col].dtype
    if dt == int or dt == float:
        df_train[col].fillna(-1, inplace=True)
        df_test[col].fillna(-1, inplace=True)
    else:
        df_train[col].fillna('', inplace=True)
        df_test[col].fillna('', inplace=True)

for col in categorical_features:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    mapping = {v: i for i, v in enumerate(le.classes_)}
    df_test[col] = df_test[col].apply(lambda x: mapping[x] if x in mapping else -1)

splitter = KFold(n_splits=5, shuffle=True, random_state=28)
prediction_list = []
best_scores = []
for train_idx, valid_idx in splitter.split(df_train):
    train, valid = df_train.iloc[train_idx], df_train.iloc[valid_idx]
    X_train, y_train = train.drop('keiyaku_pr', axis=1), train['keiyaku_pr']
    X_valid, y_valid = valid.drop('keiyaku_pr', axis=1), valid['keiyaku_pr']
    regressor = lgb.LGBMRegressor(n_estimators=20000, silent=False, random_state=28, objective='MAPE')
    regressor.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=500)
    prediction_list.append(regressor.predict(df_test[df_train.drop(objective, axis=1).columns]))
    best_scores.append(regressor.best_score_['valid_0']['mape'])

print("5-fold cv mean mape %.8f" % np.mean(best_scores))

df_submission = pd.read_csv('./data/sample_submit.tsv', sep='\t', names=['id', 'pred'])

df_submission['pred'] = np.mean(prediction_list, axis=0)
df_submission.to_csv('submission.tsv', sep='\t', header=None, index=False)

# 0.08372451
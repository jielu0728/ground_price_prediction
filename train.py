import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(objective, axis=1), df_train[objective],
                                                      test_size=0.1, random_state=28)

regressor = lgb.LGBMRegressor(n_estimators=20000, silent=False, random_state=28, objective='MAPE')
regressor.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=1000)

prediction = regressor.predict(df_test[df_train.drop(objective, axis=1).columns])

df_submission = pd.read_csv('./data/sample_submit.tsv', sep='\t', names=['id', 'pred'])
df_submission['pred'] = prediction
df_submission.to_csv('submission.tsv', sep='\t', header=None, index=False)
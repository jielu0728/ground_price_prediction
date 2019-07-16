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

# preprocessing
def fill_city_name(name):
    if '市' not in name and '郡' not in name:
        name = '市' + name
    return name

def split_address(df):
    df['jukyo'] = df['jukyo'].str.slice(start=3).str.replace(r'[ヶｹ]', 'ケ')
    df['jukyo'] = df['jukyo'].apply(fill_city_name)
    city_split = df['jukyo'].str.split(r'[市郡]', n=1, expand=True)
    df['city'] = city_split[0]
    street_split = city_split[1].str.split(r'[町区]', n=1, expand=True)
    df['street'] = street_split[0]
    df['address_detail'] = street_split[1].str.strip().replace('', None)
    return df

df_train = split_address(df_train)
df_test = split_address(df_test)

df_train.drop(['kaoku_um'], axis=1, inplace=True)
df_test.drop(['kaoku_um'], axis=1, inplace=True)

# train
categorical_features = list(df_description[df_description['データ型'] != '数値']['項目名']) + ['city', 'street', 'address_detail']
for col in ['kaoku_um']:
    categorical_features.remove(col)
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
    X_train, y_train = train.drop('keiyaku_pr', axis=1), np.log(train['keiyaku_pr']+1)
    X_valid, y_valid = valid.drop('keiyaku_pr', axis=1), np.log(valid['keiyaku_pr']+1)
    regressor = lgb.LGBMRegressor(n_estimators=20000, silent=False, random_state=28)
    regressor.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=500)
    prediction_list.append(regressor.predict(df_test[df_train.drop(objective, axis=1).columns]))
    best_scores.append(regressor.best_score_['valid_0']['l2'])

print("5-fold cv mean l2 %.8f" % np.mean(best_scores))

df_submission = pd.read_csv('./data/sample_submit.tsv', sep='\t', names=['id', 'pred'])

df_submission['pred'] = np.exp(np.mean(prediction_list, axis=0))-1
df_submission.to_csv('submission.tsv', sep='\t', header=None, index=False)

# 0.01333821
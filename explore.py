import pandas as pd
from pprint import pprint
df_train_genba = pd.read_csv('./data/train_genba.tsv', sep='\t')
df_train_goto = pd.read_csv('./data/train_goto.tsv', sep='\t')
df_train = df_train_goto.merge(df_train_genba, on='pj_no', how='left')
df_train.drop('id', axis=1, inplace=True)

def fill_city_name(name):
    if '市' not in name and '郡' not in name:
        name = '市' + name
    return name

df_train['jukyo'] = df_train['jukyo'].str.slice(start=3).str.replace(r'[ヶｹ]', 'ケ')
df_train['jukyo'] = df_train['jukyo'].apply(fill_city_name)
city_split = df_train['jukyo'].str.split(r'[市郡]', n=1, expand=True)
df_train['city'] = city_split[0]
street_split = city_split[1].str.split(r'[町区]', n=1, expand=True)
df_train['street'] = street_split[0]
df_train['address_detail'] = street_split[1].str.strip().replace('', None)

pprint(df_train['city'])
pprint(df_train['street'])
pprint(df_train['address_detail'])

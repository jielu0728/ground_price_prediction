import pandas as pd
from pprint import pprint
df_train_genba = pd.read_csv('./data/train_genba.tsv', sep='\t')
df_train_goto = pd.read_csv('./data/train_goto.tsv', sep='\t')
df_train = df_train_goto.merge(df_train_genba, on='pj_no', how='left')
df_train.drop('id', axis=1, inplace=True)

hokakisei_replace = {'公有地拡大推進法': '公拡法',
                     '文化財保護法（埋蔵文化財）': '文化財保護法',
                     '埋蔵文化財': '文化財保護法',
                     '東日本震災復興特': '東日本大震災復興特別区域法',
                     '農地法届出要': '農地法'}

s = set()
for i in range(1, 5):
    s.update(df_train['hokakisei%d' % i])
pprint(s)
print(len(s))
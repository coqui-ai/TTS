import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('metadata.csv', sep='|', names=["paths", "text", "speaker", "style"])

from sklearn.utils import shuffle
df_shuffled = shuffle(df)
a = round(len(df_shuffled)*0.98)
df_train = df_shuffled.iloc[0:a]

df_valtest = df_shuffled.iloc[a:]
df_val = df_valtest.iloc[0:15]
df_test = df_valtest.iloc[15:]
print(df_val)
#metadata_test.to_csv('./metadata_test.csv', index=False, header=False, sep='|')

df_train.to_csv('./metadata_train.csv', index=False, header=False, sep='|')
df_val.to_csv('./metadata_val.csv', index=False, header=False, sep='|')
df_test.to_csv('./metadata_test.csv', index=False, header=False, sep='|')
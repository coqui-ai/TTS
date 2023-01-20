import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

parser = ArgumentParser(description="Metadata_type")
parser.add_argument('-type', action='store', dest='type', type=str, default="Neutral", required=True, help="Partition of the dataset to consider, can be a specific style or speaker")

args = parser.parse_args()

df = pd.read_csv('metadata_' + args.type + '.csv', sep='|', names=["paths", "text", "speaker", "style"])

from sklearn.utils import shuffle
df_shuffled = shuffle(df)
a = round(len(df_shuffled)*0.98)
df_train = df_shuffled.iloc[0:a]

df_valtest = df_shuffled.iloc[a:]
df_val = df_valtest.iloc[0:15]
df_test = df_valtest.iloc[15:]
print(df_val)
#metadata_test.to_csv('./metadata_test.csv', index=False, header=False, sep='|')

df_train.to_csv('./metadata_train_' + args.type + '.csv', index=False, header=False, sep='|')
df_val.to_csv('./metadata_val_' + args.type + '.csv', index=False, header=False, sep='|')
df_test.to_csv('./metadata_test_' + args.type + '.csv', index=False, header=False, sep='|')
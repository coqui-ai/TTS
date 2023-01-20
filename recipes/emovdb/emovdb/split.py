import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('metadata.csv', sep='|', names=["paths", "text", "speaker", "style"])

phrases = df.text.value_counts()[df.text.value_counts() == 18].index.to_numpy()
sizes = [len(i) for i in phrases]
max_size = max(sizes)
max_phrase = phrases[np.argmax(sizes)]
min_size = min(sizes)
min_phrase = phrases[np.argmin(sizes)]
med_size = np.sort(sizes)[int(np.ceil(len(sizes)/2))]
med_phrase = phrases[int(np.ceil(len(sizes)/2))]
med2_size = np.median(sizes)
med2_phrase = phrases[np.argsort(sizes)[len(sizes)//2]]

max_df = df[df.text == max_phrase]
min_df = df[df.text == min_phrase]
med_df1 = df[df.text == med_phrase]
med_df2 = df[df.text == med2_phrase]

metadata_test = pd.concat([max_df, min_df, med_df1, med_df2])
metadata_test.to_csv('./metadata_test.csv', index=False, header=False, sep='|')

train_val = df[~df.isin(metadata_test)].dropna().reset_index()

josh = train_val[train_val.speaker == 'josh']
bea = train_val[train_val.speaker == 'bea']
jenie = train_val[train_val.speaker == 'jenie']
sam = train_val[train_val.speaker == 'sam']

josh_train, josh_val = train_test_split(josh, test_size=0.01, stratify=josh['style'])
bea_train, bea_val = train_test_split(bea, test_size=0.01, stratify=bea['style'])
jenie_train, jenie_val = train_test_split(jenie, test_size=0.01, stratify=jenie['style'])
sam_train, sam_val = train_test_split(sam, test_size=0.01, stratify=sam['style'])

train = pd.concat([josh_train, bea_train, jenie_train, sam_train]).drop(columns=['index'])
train.to_csv('./metadata_train.csv', index=False, header=False, sep='|')

val = pd.concat([josh_val, bea_val, jenie_val, sam_val]).drop(columns=['index'])
val.to_csv('./metadata_val.csv', index=False, header=False, sep='|')

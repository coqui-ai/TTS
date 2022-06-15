import argparse
import json
import os
import pickle
import random
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm


def load_df(filename):
    df = pd.read_csv(filename, sep ='\t')
    return df

def analyze_df(df,label):
    label_dict = {}
    df_filtered = df[df[label].notnull() &  df[label].notna()]
    df_final = df_filtered[df_filtered[label]!="other"][label]
    for ac in df_final.unique():
        speakers = df[df[label]==ac]['client_id'].unique()
        no_speakers = len(speakers)
        if(no_speakers<50):
            continue
        label_dict[ac]=speakers
        print(Fore.YELLOW, f"\"{ac}\" unique speakers no.: {no_speakers}")
    print(Style.RESET_ALL)
    return label_dict

def train_test_split(df, label, label_dict, split=0.1):
    print(len(label_dict.keys()),label_dict.keys())
    train = pd.DataFrame()
    test = pd.DataFrame()
    for l in label_dict.keys():
        spkrs = label_dict[l]
        train_spkrs = spkrs[:int(len(spkrs)*(1-split))]
        test_spkrs = spkrs[int(len(spkrs)*(1-split)):]
        train = pd.concat([train,df[df.client_id.isin(train_spkrs)]])
        test = pd.concat([test,df[df.client_id.isin(test_spkrs)]])
    train = train[train[label]!="other"]
    test = test[test[label]!="other"]
    return train, test

def mp3_to_wav(mp3_list,data_path,data_split_path,json_file):
    waves = []
    for i in tqdm(mp3_list):
        sound = AudioSegment.from_mp3(f"{data_path}/{i}")
        wav = f'{data_path}/{i.split(".mp3")[0]}.wav'
        waves.append(wav)
        sound.export(wav, format="wav")

    ff = open(f"{data_split_path}",'w')
    ff.write("wav_filename|gender|text|speaker_name\n")
    for i,j in enumerate(waves):
        ff.write(f"{j}|m|blabla|ID_{i}\n")
    ff.close()
    write_config_dataset(data_path,data_split_path,json_file)

def write_config_dataset(data_path,data_split_path,json_path):
    data = {
    "model": "vits",
    "datasets": [
            {
            "name": "brspeech",
            "path": data_path,
            "meta_file_train": data_split_path,
            "language": "en",
            "meta_file_val": "null",
            "meta_file_attn_mask": ""
            }
        ]
    }
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)

def compute_speaker_emb(tts_root_dir,spkr_emb_model, spkr_emb_config, config_dataset, out_emb_json):
    cmd = ["python", f"{tts_root_dir}/TTS/bin/compute_embeddings.py", "--use_cuda", "True" ,
            "--no_eval", "True", spkr_emb_model,
            spkr_emb_config, config_dataset, out_emb_json]
    print(" ".join(cmd))
    print(subprocess.check_output(cmd).decode("utf-8"))

def compose_dataset(embeddings_json,df,label,split,out_array_path):
    f = open(embeddings_json)
    embs = json.load(f)
    e = []
    l = []
    for i in tqdm(df.path):
        id_=i.split('.mp3')[0]+".wav"
        e.append(embs[id_]['embedding'])
        l.append(df[df['path']==i][label].item())
    '''
    for i in tqdm(embs):
        id_ = i.split('/')[-1].split('.wav')[0]+".mp3"
        e.append(embs[i]['embedding'])
        l.append(df[df['path']==id_][label].item())
    '''
    #import pdb; pdb.set_trace()
    values = np.array(l)
    label_encoder = LabelEncoder()
    #print(f"{l} {label_encoder}")
    integer_encoded = label_encoder.fit_transform(values)
    print(np.unique(values,return_counts=True),np.unique(integer_encoded))
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot = onehot_encoder.fit_transform(integer_encoded)

    d = list(zip(e,onehot))
    random.shuffle(d)
    data , labels = zip(*d)
    data_name = f"{out_array_path}_data.npy"
    label_name = f"{out_array_path}_labels.npy"
    np.save(data_name, data)
    np.save(label_name,labels)
    uniq, counts = np.unique(values,return_counts=True)
    weight={}
    for i in np.unique(integer_encoded):
        weight[i]=(1/counts[i])*(len(values)/2.0)
    print(weight)
    with open(f'{out_array_path}-weights.pkl', 'wb') as f:
        pickle.dump(weight, f)
    print(f"Data: {np.array(data).shape} ,{data_name} \n Labels: {np.array(labels).shape} , {label_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="Full path of CV data in tsv format",
        required=True
    )
    parser.add_argument(
        "--label",
        required=True
    )
    parser.add_argument(
        "--out_dir",
        required=True
    )
    parser.add_argument(
        "--spkr_emb_model",
        required=True
    )
    parser.add_argument(
        "--spkr_emb_config",
        required=True
    )
    parser.add_argument(
        "--tts_root_dir",
        required=True
    )

    args = parser.parse_args()
    abs_path = '/'.join(args.data.split("/")[:-1])
    data_path = os.path.join(abs_path,"clips")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    args.out_dir = os.path.join(abs_path,args.out_dir)
    df = load_df(args.data)

    print(Fore.RED + f"Data header: {list(df)}")
    print(Style.RESET_ALL)
    assert args.label in list(df)
    label_dict = analyze_df(df,args.label)
    train_df, test_df = train_test_split(df, args.label, label_dict)
    for split in ["train", "test"]:
        if split=='train':
            df_subset = train_df
        else:
            df_subset = test_df
        tts_csv = os.path.join(args.out_dir,f"{args.label}_{split}_tts.csv")
        config_dataset =  os.path.join(args.out_dir, f"{args.label}_{split}_config_dataset.json")
        #mp3_to_wav(df_subset['path'],data_path,tts_csv,config_dataset)
        out_emb_json = "/datasets/cv/8.0/en/accent/filtered_spkr_embs.json" #os.path.join(args.out_dir,f"{args.label}_{split}_spkr_embs.json")
        #compute_speaker_emb(args.tts_root_dir, args.spkr_emb_model, args.spkr_emb_config, config_dataset, out_emb_json)
        out_array_path = os.path.join(args.out_dir, f"{args.label}_{args.label}_{split}")
        compose_dataset(out_emb_json,df_subset,args.label,split,out_array_path)

    print ("Done.")

if __name__ == "__main__":
    main()

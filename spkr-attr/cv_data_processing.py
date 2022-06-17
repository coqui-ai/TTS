import argparse
import json
import os
import pickle
import random
import subprocess
from argparse import RawTextHelpFormatter

import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from tqdm import tqdm


def load_df(filename, n):
    if n == "All":
        df = pd.read_csv(filename, sep="\t")
    else:
        df = shuffle(pd.read_csv(filename, sep="\t")).head(n=int(n))
    return df


def analyze_df(df, label):
    label_dict = {}
    df_filtered = df[df[label].notnull() & df[label].notna()]
    df_final = df_filtered[df_filtered[label] != "other"][label]
    for ac in df_final.unique():
        speakers = df[df[label] == ac]["client_id"].unique()
        no_speakers = len(speakers)
        label_dict[ac] = speakers
        print(f'"{ac}" unique speakers no.: {no_speakers}')
    return label_dict


def train_test_split(df, label, label_dict, split=0.1):
    print(len(label_dict.keys()), label_dict.keys())
    train = pd.DataFrame()
    test = pd.DataFrame()
    for l in label_dict.keys():
        spkrs = label_dict[l]
        train_spkrs = spkrs[: int(len(spkrs) * (1 - split))]
        test_spkrs = spkrs[int(len(spkrs) * (1 - split)) :]
        train = pd.concat([train, df[df.client_id.isin(train_spkrs)]])
        test = pd.concat([test, df[df.client_id.isin(test_spkrs)]])
    train = train[train[label] != "other"]
    test = test[test[label] != "other"]
    return train, test


def mp3_to_wav(mp3_list, data_path, data_split_path, json_file):
    waves = []
    for i in tqdm(mp3_list):
        sound = AudioSegment.from_mp3(f"{data_path}/{i}")
        wav = f'{data_path}/{i.split(".mp3")[0]}.wav'
        waves.append(wav)
        sound.export(wav, format="wav")

    with open(f"{data_split_path}", "w") as f:
        f.write("wav_filename|gender|text|speaker_name\n")
        for i, j in enumerate(waves):
            f.write(f"{j}|m|blabla|ID_{i}\n")
    write_config_dataset(data_path, data_split_path, json_file)


def write_config_dataset(data_path, data_split_path, json_path):
    cwd = os.getcwd()
    data_split_full_path = os.path.join(cwd, data_split_path)
    data = {
        "model": "vits",
        "datasets": [
            {
                "name": "brspeech",
                "path": data_path,
                "meta_file_train": data_split_full_path,
                "language": "en",
                "meta_file_val": "null",
                "meta_file_attn_mask": "",
            }
        ],
    }
    with open(json_path, "w") as outfile:
        json.dump(data, outfile)


def compute_speaker_emb(tts_root_dir, spkr_emb_model, spkr_emb_config, config_dataset, out_emb_json):
    cmd = [
        "python",
        f"{tts_root_dir}/TTS/bin/compute_embeddings.py",
        "--no_eval",
        "True",
        spkr_emb_model,
        spkr_emb_config,
        config_dataset,
        "--output_path",
        out_emb_json,
    ]
    print(" ".join(cmd))
    print(subprocess.check_output(cmd).decode("utf-8"))


def compose_dataset(embeddings_json, df, label, out_array_path):
    with open(embeddings_json) as f:
        embs = json.load(f)
    e = []
    l = []
    for i in tqdm(df.path):
        id_ = i.split(".mp3")[0] + ".wav"
        e.append(embs[id_]["embedding"])
        l.append(df[df["path"] == i][label].item())
    values = np.array(l)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(np.unique(values, return_counts=True), np.unique(integer_encoded))
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot = onehot_encoder.fit_transform(integer_encoded)

    d = list(zip(e, onehot))
    random.shuffle(d)
    data, labels = zip(*d)
    data_name = f"{out_array_path}_data.npy"
    label_name = f"{out_array_path}_labels.npy"
    np.save(data_name, data)
    np.save(label_name, labels)
    _, counts = np.unique(values, return_counts=True)
    weight = {}
    for i in np.unique(integer_encoded):
        weight[i] = (1 / counts[i]) * (len(values) / 2.0)
    print(weight)
    with open(f"{out_array_path}-weights.pkl", "wb") as f:
        pickle.dump(weight, f)
    print(f"Data: {np.array(data).shape} ,{data_name} \n Labels: {np.array(labels).shape} , {label_name}")


def main():
    parser = argparse.ArgumentParser(
        description="A scirpt to prepare CV data for speaker embedding classification.\n"
        "Example runs:\n"
        "python cv_data_processing.py --data /datasets/cv/8.0/en/train.tsv --attribute age --out_dir result --num_rec 100 --tts_root_dir /mount-storage/TTS/TTS --spkr_emb_model models/model_se.pth.tar --spkr_emb_config models/config_se.json",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--data", help="Full path of CV data in tsv format", required=True)
    parser.add_argument(
        "--num_rec", help="Number of records to use out of --data. Supply All to use all of the records", required=True
    )
    parser.add_argument("--attribute", help="Speaker attribute to sample from", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--spkr_emb_model", required=True)
    parser.add_argument("--spkr_emb_config", required=True)
    parser.add_argument("--tts_root_dir", required=True)

    args = parser.parse_args()

    abs_path = "/".join(args.data.split("/")[:-1])
    data_path = os.path.join(abs_path, "clips")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    df = load_df(args.data, args.num_rec)

    print(f"Data header: {list(df)}")
    assert args.attribute in list(df)
    label_dict = analyze_df(df, args.attribute)
    train_df, test_df = train_test_split(df, args.attribute, label_dict)
    for split in ["train", "test"]:
        if split == "train":
            df_subset = train_df
        else:
            df_subset = test_df
        tts_csv = os.path.join(args.out_dir, f"{args.attribute}_{split}_tts.csv")
        config_dataset = os.path.join(args.out_dir, f"{args.attribute}_{split}_config_dataset.json")
        mp3_to_wav(df_subset["path"], data_path, tts_csv, config_dataset)
        out_emb_json = os.path.join(args.out_dir, f"{args.attribute}_{split}_spkr_embs.json")
        compute_speaker_emb(args.tts_root_dir, args.spkr_emb_model, args.spkr_emb_config, config_dataset, out_emb_json)
        out_array_path = os.path.join(args.out_dir, f"{args.attribute}_{split}")
        compose_dataset(out_emb_json, df_subset, args.attribute, out_array_path)

    print("Done.")


if __name__ == "__main__":
    main()

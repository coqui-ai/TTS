import argparse
import pandas as pd
import json
from sklearn.utils import shuffle
from TTS.tts.utils.synthesis import synthesis
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from pathlib import Path
import numpy as np
import os
import scipy.io.wavfile
from tqdm import tqdm
import subprocess


source_path = Path(__file__).resolve()
ROOT_PATH = source_path.parent
out_path = os.path.join(ROOT_PATH,"output")
if not os.path.exists(out_path):
    os.makedirs(out_path)
tts_ckpt = os.path.join(ROOT_PATH, "models/model.pth")
tts_config_path = os.path.join(ROOT_PATH, "models/config.json")
if not os.path.exists("models"):
    os.mkdirs("models")
    subprocess.run(["aws", "s3", "cp", "s3://coqui-ai-models/TTS/published_models/YourTTSZeroShot/model.pth", "models/"])
    subprocess.run(["aws", "s3", "cp", "s3://coqui-ai-models/TTS/published_models/YourTTSZeroShot/config.json", "models/"])
tts_config = load_config(tts_config_path)
tts_model = setup_tts_model(config=tts_config)
tts_model.load_checkpoint(tts_config, tts_ckpt, eval=True)



def modify_df(data,embs):
    all = data.drop_duplicates(subset=['client_id'])
    gender = all[all[all.gender!="other"].notna()].gender
    age = all[all.age.notna()].age
    accents = all[all.accents.notna()].accents
    wavname = all[all.path.notna()].path
    #import pdb; pdb.set_trace()
    df = pd.concat((wavname,age,gender,accents),axis=1)
    return df

def generate_pseudo_spkr(embs,df,label,val,n=100):
    subset_df = df[df[label]==val]
    if len(subset_df)>n:
        subset_df = shuffle(subset_df.head(n=int(n)))
    wave_files = subset_df.path
    dvectors = []
    for i in wave_files:
        wav = i.split("mp3")[0]+'wav'
        if wav in embs.keys():
            dvectors.append(embs[wav]['embedding'])
    try:
        assert len(dvectors)>=2
    except AssertionError:
        return [] , len(dvectors)
    average_dvector = np.mean(np.array(dvectors), axis=0).reshape([512,1])
    return average_dvector, len(dvectors)
    
def synthesize_speech(embs,df,label,val,n=100):
    average_dvector, num_spkrs = generate_pseudo_spkr(embs,df,label,val,n=100)
    if len(average_dvector) == 0 :
        print(f"Not enough speakers for {val}, Skipping!")
        return 
    outputs = synthesis(model=tts_model,
                    text="Once upon a time, the King’s youngest son became filled with the desire to go abroad and see the world.",
                    CONFIG=tts_config, use_cuda=False,d_vector=average_dvector,language_id=0)
    waveform = outputs["wav"]
    scipy.io.wavfile.write(f"{out_path}/{label}-{val}-{num_spkrs}.wav", 44100, waveform)

def main():
    parser = argparse.ArgumentParser(
        description="A scirpt to generate psuedo speaker embedding based on some speaker attributes.\n"
        "Example run:\n"
        "python pool_selection.py --metadata metadata.tsv --embedding speakers.json --attribute age --out_dir result\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--metadata", help="metadata file in tsv format with these fields at least[path|gender|age|accent]", required=True)
    parser.add_argument("--embeddings", help="speaker embeddings in json format", required=True)
    parser.add_argument("--attribute", help="speaker attribute to condition on [gender or age or accent or all]", required=True)

    args = parser.parse_args()

    data = pd.read_csv(args.metadata, sep="\t")
    print("data loaded")
    f = open(args.embeddings)
    embs = json.load(f)
    print("embeddings loaded")

    df = modify_df(data,embs)   
    
    assert args.attribute in ['accents', 'age', 'gender', 'all']
    label = args.attribute
    
    if label != 'all':
        unique_labels = df[df[label].notna()].drop_duplicates(subset=[label])[label].tolist()
        val = input(f"Generate pseudo speaker based on {args.attribute} out of {unique_labels}: ")
        assert val in unique_labels
        synthesize_speech(embs,df,label,val,n=100)
    else:
        for label in tqdm(['age', 'gender']):
            unique_labels = df[df[label].notna()].drop_duplicates(subset=[label])[label].tolist()
            for val in tqdm(unique_labels):
                synthesize_speech(embs,df,label,val,n=100)

if __name__ == "__main__":
    main()


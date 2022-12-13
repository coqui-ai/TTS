import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

sys.path.insert(1, '/workspace/coqui-tts')

from TTS.trainer import Trainer, TrainingArgs

import warnings

from TTS.config import load_config, register_config
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.styles import StyleManager

warnings.filterwarnings('ignore')


# parser = argparse.ArgumentParser(
#     description="""Compute embedding vectors for each wav file in a dataset.\n\n"""
#     """
#     Example runs:
#     python TTS/bin/compute_embeddings.py speaker_encoder_model.pth.tar speaker_encoder_config.json  dataset_config.json embeddings_output_path/
#     """,
# )
# parser.add_argument("model_path", type=str, help="Path to model checkpoint file.")
# parser.add_argument(
#     "config_path",
#     type=str,
#     help="Path to model config file.",
# )

# parser.add_argument("train_df_path", type=str, help="path for train df.")
# parser.add_argument("val_df_path", type=str, help="path for eval df.")
# parser.add_argument("test_df_path", type=str, help="path for test df.")
# parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)

# args = parser.parse_args()


def numpy_to_torch(np_array, dtype, cuda=False):
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype)
    if cuda:
        return tensor.cuda()
    return tensor

def id_to_torch(speaker_id, cuda=False):
    if speaker_id is not None:
        speaker_id = np.asarray(speaker_id)
        speaker_id = torch.from_numpy(speaker_id).unsqueeze(0)
    if cuda:
        return speaker_id.cuda().type(torch.long)
    return speaker_id.type(torch.long)

import librosa
def compute_style_mel(style_wav, ap, cuda=False):
    style_mel = torch.FloatTensor(ap.melspectrogram(
        ap.load_wav(style_wav, sr=ap.sample_rate))).unsqueeze(0)
#     style_mel = torch.FloatTensor(ap.melspectrogram(
#         librosa.load(style_wav, sr=22050)[0])).unsqueeze(0)
    if cuda:
        return style_mel.cuda()
    return style_mel

# work only for CPQD files
def map_wavpath2style(wav_path):
    if('eps_acolhedor' in wav_path):
        return 'acolhedor'
    elif('eps_animado' in wav_path):
        return 'animado'
    elif('eps_rispido' in wav_path):
        return 'rispido'
    elif('eps_neutro' in wav_path):
        return 'neutro'
    elif('eps_teste_neutro' in wav_path):
        return 'neutro'
    elif('eps_teste_animado' in wav_path):
        return 'animado'
    elif('eps_teste_acolhedor' in wav_path):
        return 'acolhedor'
    elif('eps_teste_rispido' in wav_path):
        return 'rispido'
    else:
        return 'none'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    # Unused
    # parser.add_argument(
    #     '--config_path',
    #     type=str,
    #     help='Path to config file for training, if experiment folder is defined it is ignored.'
    # )
    parser.add_argument(
        '--train_df_path',
        type=str,
        help='Training dataframe path.'
    )
    parser.add_argument(
        '--val_df_path',
        type=str,
        help='Validation dataframe path.'
    )
    parser.add_argument(
        '--test_df_path',
        type=str,
        help='Test dataframe path.'
    )
    parser.add_argument('--experiment_path',
                        type=str,
                        default='',
                        help='Experiment path with the config.json file present and where all results will be stored.')

    parser.add_argument('--use_cuda',
                    type=str,
                    default='yes',
                    help='Whether use or not cuda "yes" or "no".')

    args = parser.parse_args()

    use_cuda = False
    if(args.use_cuda == 'yes'):
        use_cuda = True

    config = load_config(args.experiment_path + "/config.json")

    ## TO CHECK: Maybe we need to fix the data_path root

    train_args = TrainingArgs()

    # load training samples
    train_samples, eval_samples = load_tts_samples(config.datasets, eval_split=True)

    # setup audio processor
    ap = AudioProcessor(**config.audio)

    # init speaker manager
    if config.use_speaker_embedding:
        speaker_manager = SpeakerManager(data_items=train_samples + eval_samples)
    elif config.use_d_vector_file:
        speaker_manager = SpeakerManager(d_vectors_file_path=config.d_vector_file)
    else:
        speaker_manager = None

    # If, use style information, init the StyleManager
    if config.style_encoder_config.use_supervised_style:
        style_manager = StyleManager(data_items=train_samples + eval_samples)
        if hasattr(config, "model_args"):
            config.model_args.num_styles = style_manager.num_styles
        else:
            config.num_styles = style_manager.num_styles
    else:
        style_manager = None
        
    # init the model from config
    language_manager = None
    model = setup_model(config, speaker_manager, language_manager, style_manager)

    # init the trainer
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )

    # Load the checkpoint path model weights
    trainer.model, opt, scaler, restore_step = trainer.restore_model(config, args.model_checkpoint, trainer.model, trainer.optimizer, trainer.scaler)

    # Now lets init each dataframe, default is using delimiter = ';' , so its easier to preprocess the lists manually before running the script
    train_df = pd.read_csv(args.train_df_path, delimiter = ';', encoding = 'utf-8')
    val_df = pd.read_csv(args.val_df_path, delimiter = ';', encoding = 'utf-8')
    test_df = pd.read_csv(args.test_df_path, delimiter = ';', encoding = 'utf-8')


    # Setup dict mappings for style, this is used to plot graphs for each style
    style2id = style_manager.style_id_mapping
    id2style = {v: k for k, v in style_manager.style_id_mapping.items()}

    # Setup length of each dataset partition 
    N = config['style_encoder_config']['style_embedding_dim']
    train_feats = np.zeros((len(train_df), N))
    val_feats = np.zeros((len(val_df), N))
    test_feats = np.zeros((len(test_df), N))

    # Now, extracting style features
    ### TRAIN DATASET
    train_styles_id = []

    for i in range(len(train_df)):
        style_wav = train_df.wav_path.values[i]
        style_mel = compute_style_mel(style_wav, ap, cuda=True)
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)[0].T

        o_en, outputs = model.cuda().style_encoder_layer.forward([torch.rand(1,384,1).cuda(),style_mel.unsqueeze(0)], None)

        if(config['style_encoder_config']['se_type'] == 'vae'):
            outputs = outputs['z']
        elif(config['style_encoder_config']['se_type'] == 'diffusion'):
            outputs = outputs['style']

        train_feats[i] = outputs.squeeze(0).squeeze(0).detach().cpu().numpy()
        train_styles_id.append(style2id[train_df.style.values[i]])

    ### VAL DATASET
    val_styles_id = []

    for i in range(len(val_df)):
        style_wav = val_df.wav_path.values[i]
        style_mel = compute_style_mel(style_wav, ap, cuda=True)
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)[0].T

        o_en, outputs = model.cuda().style_encoder_layer.forward([torch.rand(1,384,1).cuda(),style_mel.unsqueeze(0)], None)

        if(config['style_encoder_config']['se_type'] == 'vae'):
            outputs = outputs['z']
        elif(config['style_encoder_config']['se_type'] == 'diffusion'):
            outputs = outputs['style']

        val_feats[i] = outputs.squeeze(0).squeeze(0).detach().cpu().numpy()
        val_styles_id.append(style2id[val_df.style.values[i]])

    ### TEST DATASET
    test_styles_id = []

    for i in range(len(test_df)):
        style_wav = test_df.wav_path.values[i]
        style_mel = compute_style_mel(style_wav, ap, cuda=True)
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)[0].T

        o_en, outputs = model.cuda().style_encoder_layer.forward([torch.rand(1,384,1).cuda(),style_mel.unsqueeze(0)], None)

        if(config['style_encoder_config']['se_type'] == 'vae'):
            outputs = outputs['z']
        elif(config['style_encoder_config']['se_type'] == 'diffusion'):
            outputs = outputs['style']

        test_feats[i] = outputs.squeeze(0).squeeze(0).detach().cpu().numpy()
        test_styles_id.append(style2id[test_df.style.values[i]])

    # Now lets get the embeddings spaces

    u = umap.UMAP(random_state = 42)
    train_embeddings = u.fit_transform(train_feats)
    # Saving pickle for this fitted umap "u" to be able to re-use it in other data partitions without running this script all again
    import pickle

    f_name = args.experiment_path + '/umap.pkl'
    pickle.dump(u, open(f_name, 'wb'))

    val_embeddings = u.transform(val_feats)

    test_embeddings = u.transform(test_feats)

    # Saving and exporting df's used for plots (to better control if want different style of plotting)
    train_plot_df = pd.DataFrame({'style': train_styles_id, 'dim1': train_embeddings[:,0], 'dim2': train_embeddings[:,1]})
    val_plot_df = pd.DataFrame({'style': val_styles_id, 'dim1': val_embeddings[:,0], 'dim2': val_embeddings[:,1]})
    test_plot_df = pd.DataFrame({'style': test_styles_id, 'dim1': test_embeddings[:,0], 'dim2': test_embeddings[:,1]})

    train_plot_df.to_csv(args.experiment_path + '/train_plot_df.csv', index = False)
    val_plot_df.to_csv(args.experiment_path + '/val_plot_df.csv', index = False)
    test_plot_df.to_csv(args.experiment_path + '/test_plot_df.csv', index = False)

    # Plotting and saving figs of generated style distributions 
    ### TRAIN
    plt.figure(figsize=(12,5))
    for i in range(train_plot_df.style.nunique()):
        df_filt = train_plot_df[train_plot_df['style'] == i]
        plt.scatter(df_filt['dim1'], df_filt['dim2'], label = id2style[i])
    plt.legend(fontsize=15)
    plt.xlabel('UMAP dim 1', fontsize = 20)
    plt.ylabel('UMAP dim 1', fontsize = 20)

    plt.savefig(args.experiment_path + "/training_style_distribution.png", dpi = 300)

    ### VAL
    plt.figure(figsize=(12,5))
    for i in range(val_plot_df.style.nunique()):
        df_filt = val_plot_df[val_plot_df['style'] == i]
        plt.scatter(df_filt['dim1'], df_filt['dim2'], label = id2style[i])
    plt.legend(fontsize=15)
    plt.xlabel('UMAP dim 1', fontsize = 20)
    plt.ylabel('UMAP dim 1', fontsize = 20)

    plt.savefig(args.experiment_path + "/validation_style_distribution.png", dpi = 300)


    ### TEST
    plt.figure(figsize=(12,5))
    for i in range(test_plot_df.style.nunique()):
        df_filt = test_plot_df[test_plot_df['style'] == i]
        plt.scatter(df_filt['dim1'], df_filt['dim2'], label = id2style[i])
    plt.legend(fontsize=15)
    plt.xlabel('UMAP dim 1', fontsize = 20)
    plt.ylabel('UMAP dim 1', fontsize = 20)

    plt.savefig(args.experiment_path + "/testing_style_distribution.png", dpi = 300)
import torch
import torchaudio


def read_audio(path):
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav.squeeze(0), sr


def resample_wav(wav, sr, new_sr):
    wav = wav.unsqueeze(0)
    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sr)
    wav = transform(wav)
    return wav.squeeze(0)


def map_timestamps_to_new_sr(vad_sr, new_sr, timestamps, just_begging_end=False):
    factor = new_sr / vad_sr
    new_timestamps = []
    if just_begging_end and timestamps:
        # get just the start and end timestamps
        new_dict = {"start": int(timestamps[0]["start"] * factor), "end": int(timestamps[-1]["end"] * factor)}
        new_timestamps.append(new_dict)
    else:
        for ts in timestamps:
            # map to the new SR
            new_dict = {"start": int(ts["start"] * factor), "end": int(ts["end"] * factor)}
            new_timestamps.append(new_dict)

    return new_timestamps


def get_vad_model_and_utils(use_cuda=False, use_onnx=False):
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True, onnx=use_onnx, force_onnx_cpu=True
    )
    if use_cuda:
        model = model.cuda()

    get_speech_timestamps, save_audio, _, _, collect_chunks = utils
    return model, get_speech_timestamps, save_audio, collect_chunks


def remove_silence(
    model_and_utils, audio_path, out_path, vad_sample_rate=8000, trim_just_beginning_and_end=True, use_cuda=False
):
    # get the VAD model and utils functions
    model, get_speech_timestamps, _, collect_chunks = model_and_utils

    # read ground truth wav and resample the audio for the VAD
    try:
        wav, gt_sample_rate = read_audio(audio_path)
    except:
        print(f"> ❗ Failed to read {audio_path}")
        return None, False

    # if needed, resample the audio for the VAD model
    if gt_sample_rate != vad_sample_rate:
        wav_vad = resample_wav(wav, gt_sample_rate, vad_sample_rate)
    else:
        wav_vad = wav

    if use_cuda:
        wav_vad = wav_vad.cuda()

    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav_vad, model, sampling_rate=vad_sample_rate, window_size_samples=768)

    # map the current speech_timestamps to the sample rate of the ground truth audio
    new_speech_timestamps = map_timestamps_to_new_sr(
        vad_sample_rate, gt_sample_rate, speech_timestamps, trim_just_beginning_and_end
    )

    # if have speech timestamps else save the wav
    if new_speech_timestamps:
        wav = collect_chunks(new_speech_timestamps, wav)
        is_speech = True
    else:
        print(f"> The file {audio_path} probably does not have speech please check it !!")
        is_speech = False

    # save
    torchaudio.save(out_path, wav[None, :], gt_sample_rate)
    return out_path, is_speech

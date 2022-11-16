import argparse
import glob
import os
import pathlib

from tqdm import tqdm

from TTS.utils.vad import get_vad_model_and_utils, remove_silence


def adjust_path_and_remove_silence(audio_path):
    output_path = audio_path.replace(os.path.join(args.input_dir, ""), os.path.join(args.output_dir, ""))
    # ignore if the file exists
    if os.path.exists(output_path) and not args.force:
        return output_path

    # create all directory structure
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # remove the silence and save the audio
    output_path, is_speech = remove_silence(
        model_and_utils,
        audio_path,
        output_path,
        trim_just_beginning_and_end=args.trim_just_beginning_and_end,
        use_cuda=args.use_cuda,
    )

    return output_path, is_speech


def preprocess_audios():
    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob), recursive=True))
    print("> Number of files: ", len(files))
    if not args.force:
        print("> Ignoring files that already exist in the output idrectory.")

    if args.trim_just_beginning_and_end:
        print("> Trimming just the beginning and the end with nonspeech parts.")
    else:
        print("> Trimming all nonspeech parts.")

    filtered_files = []
    if files:
        # create threads
        # num_threads = multiprocessing.cpu_count()
        # process_map(adjust_path_and_remove_silence, files, max_workers=num_threads, chunksize=15)
        for f in tqdm(files):
            output_path, is_speech = adjust_path_and_remove_silence(f)
            if not is_speech:
                filtered_files.append(output_path)

        # write files that do not have speech
        with open(os.path.join(args.output_dir, "filtered_files.txt"), "w", encoding="utf-8") as f:
            for file in filtered_files:
                f.write(file + "\n")
    else:
        print("> No files Found !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="python TTS/bin/remove_silence_using_vad.py -i=VCTK-Corpus/ -o=VCTK-Corpus-removed-silence/ -g=wav48_silence_trimmed/*/*_mic1.flac --trim_just_beginning_and_end True"
    )
    parser.add_argument("-i", "--input_dir", type=str, default="../VCTK-Corpus", help="Dataset root dir")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="../VCTK-Corpus-removed-silence", help="Output Dataset dir"
    )
    parser.add_argument("-f", "--force", default=False, action="store_true", help="Force the replace of exists files")
    parser.add_argument(
        "-g",
        "--glob",
        type=str,
        default="**/*.wav",
        help="path in glob format for acess wavs from input_dir. ex: wav48/*/*.wav",
    )
    parser.add_argument(
        "-t",
        "--trim_just_beginning_and_end",
        type=bool,
        default=True,
        help="If True this script will trim just the beginning and end nonspeech parts. If False all nonspeech parts will be trim. Default True",
    )
    parser.add_argument(
        "-c",
        "--use_cuda",
        type=bool,
        default=False,
        help="If True use cuda",
    )
    args = parser.parse_args()
    # load the model and utils
    model_and_utils = get_vad_model_and_utils(use_cuda=args.use_cuda)
    preprocess_audios()

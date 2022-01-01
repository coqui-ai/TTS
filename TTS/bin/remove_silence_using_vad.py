import argparse
import glob
import multiprocessing
import os
import pathlib

from tqdm.contrib.concurrent import process_map

from TTS.utils.vad import get_vad_speech_segments, read_wave, write_wave


def remove_silence(filepath):
    output_path = filepath.replace(os.path.join(args.input_dir, ""), os.path.join(args.output_dir, ""))
    # ignore if the file exists
    if os.path.exists(output_path) and not args.force:
        return

    # create all directory structure
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # load wave
    audio, sample_rate = read_wave(filepath)

    # get speech segments
    segments = get_vad_speech_segments(audio, sample_rate, aggressiveness=args.aggressiveness)

    segments = list(segments)
    num_segments = len(segments)
    flag = False
    # create the output wave
    if num_segments != 0:
        for i, segment in reversed(list(enumerate(segments))):
            if i >= 1:
                if not flag:
                    concat_segment = segment
                    flag = True
                else:
                    concat_segment = segment + concat_segment
            else:
                if flag:
                    segment = segment + concat_segment
                # print("Saving: ", output_path)
                write_wave(output_path, segment, sample_rate)
                return
    else:
        print("> Just Copying the file to:", output_path)
        # if fail to remove silence just write the file
        write_wave(output_path, audio, sample_rate)
        return


def preprocess_audios():
    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob), recursive=True))
    print("> Number of files: ", len(files))
    if not args.force:
        print("> Ignoring files that already exist in the output directory.")

    if files:
        # create threads
        num_threads = multiprocessing.cpu_count()
        process_map(remove_silence, files, max_workers=num_threads, chunksize=15)
    else:
        print("> No files Found !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="python remove_silence.py -i=VCTK-Corpus-bk/ -o=../VCTK-Corpus-removed-silence -g=wav48/*/*.wav -a=2"
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
        "-a",
        "--aggressiveness",
        type=int,
        default=2,
        help="set its aggressiveness mode, which is an integer between 0 and 3. 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.",
    )
    args = parser.parse_args()
    preprocess_audios()

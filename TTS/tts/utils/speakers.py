import os
import json


def make_speakers_json_path(out_path):
    """Returns conventional speakers.json location."""
    return os.path.join(out_path, "speakers.json")


def load_speaker_mapping(out_path):
    """Loads speaker mapping if already present."""
    try:
        if os.path.splitext(out_path)[1] == '.json':
            json_file = out_path
        else:
            json_file = make_speakers_json_path(out_path)
        with open(json_file) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_speaker_mapping(out_path, speaker_mapping):
    """Saves speaker mapping if not yet present."""
    speakers_json_path = make_speakers_json_path(out_path)
    with open(speakers_json_path, "w") as f:
        json.dump(speaker_mapping, f, indent=4)


def get_speakers(items):
    """Returns a sorted, unique list of speakers in a given dataset."""
    speakers = {e[2] for e in items}
    return sorted(speakers)

def parse_speakers(c, args, meta_data_train, OUT_PATH):
    """ Returns number of speakers, speaker embedding shape and speaker mapping"""
    if c.use_speaker_embedding:
        speakers = get_speakers(meta_data_train)
        if args.restore_path:
            if c.use_external_speaker_embedding_file: # if restore checkpoint and use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(prev_out_path)
                if not speaker_mapping:
                    print("WARNING: speakers.json was not found in restore_path, trying to use CONFIG.external_speaker_embedding_file")
                    speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
                    if not speaker_mapping:
                        raise RuntimeError("You must copy the file speakers.json to restore_path, or set a valid file in CONFIG.external_speaker_embedding_file")
                speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]['embedding'])
            elif not c.use_external_speaker_embedding_file: # if restore checkpoint and don't use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(prev_out_path)
                speaker_embedding_dim = None
                assert all([speaker in speaker_mapping
                            for speaker in speakers]), "As of now you, you cannot " \
                                                    "introduce new speakers to " \
                                                    "a previously trained model."
        elif c.use_external_speaker_embedding_file and c.external_speaker_embedding_file: # if start new train using External Embedding file
            speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
            speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]['embedding'])
        elif c.use_external_speaker_embedding_file and not c.external_speaker_embedding_file: # if start new train using External Embedding file and don't pass external embedding file
            raise "use_external_speaker_embedding_file is True, so you need pass a external speaker embedding file, run GE2E-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb or AngularPrototypical-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb notebook in notebooks/ folder"
        else: # if start new train and don't use External Embedding file
            speaker_mapping = {name: i for i, name in enumerate(speakers)}
            speaker_embedding_dim = None
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print(" > Training with {} speakers: {}".format(len(speakers),
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0
        speaker_embedding_dim = None
        speaker_mapping = None

    return num_speakers, speaker_embedding_dim, speaker_mapping
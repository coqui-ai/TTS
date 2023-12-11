import torch

class SpeakerManager():
    def __init__(self, speaker_file_path=None):
        self.speakers = torch.load(speaker_file_path)

    @property
    def name_to_id(self):
        return self.speakers.keys()

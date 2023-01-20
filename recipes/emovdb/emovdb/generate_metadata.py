import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Metadata_type")
parser.add_argument('-type', action='store', dest='type', type=str, default="Neutral", required=True, help="Partition of the dataset to consider, can be a specific style or speaker")


args = parser.parse_args()

names = []
sep = "|"
files_folder = './files'

file = open('transcriptions.txt')
content = file.readlines()

for dir in os.listdir(files_folder):
    info = dir.split('_')
    spk = info[0]
    style = info[1]
    for audio_file in os.listdir(files_folder + '/' + dir):
        file_path = dir + '/' + audio_file
        text_idx = int(audio_file.split('_')[-1].split('.')[0])
        assert text_idx <= 1132
        line = content[text_idx-1]
        line_split = line.split('"')
        text = line_split[1]
        write = file_path + sep + text + sep + spk + sep + style

        if args.type in ['Neutral', 'Sleepy', 'Angry', 'Amused', 'Disgusted']:
            if style == args.type: 
                names.append(write)
            else:
                pass
        elif args.type in ['bea', 'josh', 'sam', 'jenie']:
            if spk == args.type: 
                names.append(write)
            else:
                pass
        elif args.type == 'all':
            names.append(write)
        else:
            raise NotImplementedError


with open('metadata_' + args.type + '.csv', 'w') as f:
    for line in names:
        f.write(f"{line}\n")
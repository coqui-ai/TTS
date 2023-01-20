import os

os.rename('files/josh_Amused/amused_197-224_0023.wav', 'files/josh_Amused/amused_197-224_0223.wav')
os.rename('files/josh_Amused/amused_225_252_0026.wav', 'files/josh_Amused/amused_225_252_0226.wav')
os.rename('files/bea_Neutral/neutral_113-140_0014.wav', 'files/bea_Neutral/neutral_113-140_0114.wav')
os.rename('files/bea_Neutral/neutral_113-140_0015.wav', 'files/bea_Neutral/neutral_113-140_0115.wav')
os.rename('files/bea_Angry/anger_197-224_0024.wav','files/bea_Angry/anger_197-224_0224.wav')


# Possible Problems
'''
for dir in os.listdir(files_folder):
    for audio_file in os.listdir(files_folder + '/' + dir):
        file_path = '/' + dir + '/' + audio_file

        text_idx = int(audio_file.split('_')[-1].split('.')[0])

        file_split = audio_file.split('_')
        ran = file_split[1].split('-')
        if len(ran) < 2:
            high = file_split[2]
        else:
            high = ran[1]

        low = ran[0]

        if text_idx not in range(int(low), int(high)+1):
            print(file_path)
'''
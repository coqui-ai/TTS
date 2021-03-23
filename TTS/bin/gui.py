# GUI solution for Coqui TTS made by AceOfSpadesProduc100.
# WARNING: DO NOT run this from IDLE or double-clicking, it will be stuck loading. To be safe, you should enter "python mozilla-tts-gui.py" in your terminal such as command prompt or Powershell.
# Make sure to edit this script to add TTS models and vocoders on the ['values'] of ttsmodelbox and vocodermodelbox.
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter import messagebox
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
from pathlib import Path
import os
import errno
import io
import json
import simpleaudio as sa

def generate():
    if inputbox.get("1.0", "end-1c") == "":
        messagebox.showerror(message="TTS will give a division by zero error if the text field is blank.")
    else:
        if not os.path.exists('coqui-tts-output'):
            try:
                os.makedirs('coqui-tts-output')
            except OSError as e:
                if e.errno != e.errno.EEXIST:
                    raise
        generatebutton.config(state="disabled")
        exportbutton.config(state="disabled")
        model_path = None
        config_path = None
        vocoder_path = None
        vocoder_config_path = None
        path = Path(__file__).parent / "../.models.json"
        manager = ModelManager(path)
        model_name = 'tts_models/' + ttsmodelbox.get()
        print(f'model_name is {model_name}')
        # for dev
        model_path, config_path, model_item = manager.download_model(model_name)
        # for master
        #model_path, config_path = manager.download_model(model_name)
        vocoder_name = 'vocoder_models/' + vocodermodelbox.get()
        print(f'vocoder_name is {vocoder_name}')
        # for dev
        vocoder_path, vocoder_config_path, model_item = manager.download_model(vocoder_name)
        # for master
        #vocoder_path, vocoder_config_path = manager.download_model(vocoder_name)
        synthesizer = Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, cudacheckbutton.instate(['selected']))
        wav = synthesizer.tts(inputbox.get("1.0", "end-1c"))
        synthesizer.save_wav(wav, "coqui-tts-output/generated.wav")
        filename = "coqui-tts-output/generated.wav"
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until sound has finished playing
        generatebutton.config(state="enabled")
        exportbutton.config(state="enabled")
        if os.path.exists("coqui-tts-output/generated.wav"):
            os.remove("coqui-tts-output/generated.wav")
        print("All done!")

def savetext():
    f = filedialog.asksaveasfile(mode='w', defaultextension=".txt", filetypes=[("Text files", ".txt")])
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    text2save = str(inputbox.get("1.0", "end-1c")) # starts from `1.0`, not `0.0`
    f.write(text2save)
    f.close() # `()` was missing.
    inputbox.edit_modified(False)

def savetextandopen():
    f = filedialog.asksaveasfile(mode='w', defaultextension=".txt", filetypes=[("Text files", ".txt")])
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    text2save = str(inputbox.get("1.0", "end-1c")) # starts from `1.0`, not `0.0`
    f.write(text2save)
    f.close() # `()` was missing.
    inputbox.edit_modified(False)
    opentext()

def exportaudio():
    if inputbox.get("1.0", "end-1c") == "":
        messagebox.showerror(message="TTS will give a division by zero error if the text field is blank.")
    else:
        f = filedialog.asksaveasfile(mode='a', defaultextension=".wav", filetypes=[("Wave files", ".wav")])
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
    generatebutton.config(state="disabled")
    exportbutton.config(state="disabled")
    model_path = None
    config_path = None
    vocoder_path = None
    vocoder_config_path = None
    path = Path(__file__).parent / "TTS/.models.json"
    manager = ModelManager(path)
    model_name = 'tts_models/' + ttsmodelbox.get()
    print(f'model_name is {model_name}')
    # for dev
    model_path, config_path, model_item = manager.download_model(model_name)
    # for master
    #model_path, config_path = manager.download_model(model_name)
    vocoder_name = 'vocoder_models/' + vocodermodelbox.get()
    print(f'vocoder_name is {vocoder_name}')
    # for dev
    vocoder_path, vocoder_config_path, model_item = manager.download_model(vocoder_name)
    # for master
    #vocoder_path, vocoder_config_path = manager.download_model(vocoder_name)
    synthesizer = Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, cudacheckbutton.instate(['selected']))
    wav = synthesizer.tts(inputbox.get("1.0", "end-1c"))
    synthesizer.save_wav(wav, str(f.name))
    generatebutton.config(state="enabled")
    exportbutton.config(state="enabled")
    print("All done!")

def opentext():
    file = filedialog.askopenfile(mode='rt', defaultextension=".txt", filetypes=[("Text files", ".txt")])
    if file is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    contents = file.read()
    inputbox.insert('1.0', contents)
    inputbox.edit_modified(False)
    file.close()

def checkopentext():
    if inputbox.edit_modified() == True:
        response = messagebox.askyesnocancel(message="You have unsaved changes. Do you want to save?")
        if response == True:
            savetextandopen()
        elif response == False:
            opentext()
        else:
            return
    else:
        opentext()

# Creating tkinter window
window = tk.Tk()
window.geometry('')
window.title("Coqui TTS GUI")
window.resizable(False, False)

# Label
ttk.Label(window, text="Enter text here", font=("Tahoma", 10)).grid(column=0, columnspan=3, row=12, padx=10, pady=12)

# Text
inputbox = scrolledtext.ScrolledText(window, height=15, width=70, undo=True)
inputbox.grid(column=0, columnspan=3, row=13, padx=10, pady=12)

# Label
ttk.Label(window, text="Select the tts_model", font=("Tahoma", 10)).grid(column=0, row=14, padx=10, pady=12)
n = tk.StringVar()
ttsmodelbox = ttk.Combobox(window, width=32, textvariable=n, state="readonly")

tts_models_name_list = []
vocoder_models_name_list = []
moddict = ModelManager().models_dict
for lang in moddict["tts_models"]:
    for dataset in moddict["tts_models"][lang]:
        for model in moddict["tts_models"][lang][dataset]:
            model_full_name = f"tts_models--{lang}--{dataset}--{model}"
            output_path = os.path.join(ModelManager().output_prefix, model_full_name)
            tts_models_name_list.append(f'{lang}/{dataset}/{model}')

for lang in moddict["vocoder_models"]:
    for dataset in moddict["vocoder_models"][lang]:
        for model in moddict["vocoder_models"][lang][dataset]:
            model_full_name = f"vocoder_models--{lang}--{dataset}--{model}"
            output_path = os.path.join(ModelManager().output_prefix, model_full_name)
            vocoder_models_name_list.append(f'{lang}/{dataset}/{model}')

# Adding combobox drop down list
ttsmodelbox['values'] = (tts_models_name_list)
ttsmodelbox.grid(column=0, row=15, padx=10, pady=12)
ttsmodelbox.current(0)

# Label
ttk.Label(window, text="Select the vocoder_model", font=("Tahoma", 10)).grid(column=2, row=14, padx=10, pady=12)
r = tk.StringVar()
vocodermodelbox = ttk.Combobox(window, width=32, textvariable=r, state="readonly")

# Checkbutton
# Label
ttk.Label(window, text="Use CUDA (Nvidia GPUs only)", font=("Tahoma", 10)).grid(column=1, columnspan=1, row=14, padx=10, pady=12)
cudacheckbutton = ttk.Checkbutton(window)
cudacheckbutton.grid(column=1, columnspan=1, row=15, padx=10, pady=12)
cudacheckbutton.state(['!alternate'])
cudacheckbutton.state(['!selected'])

# Adding combobox drop down list
vocodermodelbox['values'] = (vocoder_models_name_list)
vocodermodelbox.grid(column=2, row=15, padx=10, pady=12)
vocodermodelbox.current(0)

# Button
opentextbutton = ttk.Button(window, width=20, text="Open text", command=checkopentext)
opentextbutton.grid(column=0, columnspan=1, row=16, padx=10, pady=12)
generatebutton = ttk.Button(window, width=20, text="Generate", command=generate)
generatebutton.grid(column=1, columnspan=1, row=17, padx=10, pady=12)
savetextbutton = ttk.Button(window, width=20, text="Save text", command=savetext)
savetextbutton.grid(column=1, columnspan=1, row=16, padx=10, pady=12)
exportbutton = ttk.Button(window, width=20, text="Export audio", command=exportaudio)
exportbutton.grid(column=2, columnspan=1, row=16, padx=10, pady=12)
def loadgui():
    window.mainloop()

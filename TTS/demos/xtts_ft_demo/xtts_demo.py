import os
import sys
import tempfile

import gradio as gr
import librosa.display
import numpy as np

import os
import torch
import torchaudio
from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list, list_audios

import logging

PORT = 5003



def run_tts(lang, tts_text, state_vars, temperature, rms_norm_output=False):
    return None

# define a logger to redirect 
class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

# redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout


def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()


with gr.Blocks() as demo:
    with gr.Tab("XTTS"):
        state_vars = gr.State(
        )
        with gr.Row():
            with gr.Column() as col1:       
                upload_file = gr.Audio(
                    sources="upload",
                    label="Select here the audio files that you want to use for XTTS trainining !",
                    type="filepath",
                )
                lang = gr.Dropdown(
                    label="Dataset Language",
                    value="en",
                    choices=[
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "pl",
                        "tr",
                        "ru",
                        "nl",
                        "cs",
                        "ar",
                        "zh",
                        "hu",
                        "ko",
                        "ja"
                    ],
                )
                voice_ready = gr.Label(
                    label="Progress."
                )
                logs = gr.Textbox(
                    label="Logs:",
                    interactive=False,
                )
                demo.load(read_logs, None, logs, every=1)

                prompt_compute_btn = gr.Button(value="Step 1 - Create dataset.")

            with gr.Column() as col2:

                tts_text = gr.Textbox(
                    label="Input Text.",
                    value="This model sounds really good and above all, it's reasonably fast.",
                )
                temperature = gr.Slider(
                    label="temperature", minimum=0.00001, maximum=1.0, step=0.05, value=0.75
                )
                rms_norm_output = gr.Checkbox(
                    label="RMS norm output.", value=True, interactive=True
                )
                tts_btn = gr.Button(value="Step 2 - TTS")

            with gr.Column() as col3:
                tts_output_audio_no_enhanced = gr.Audio(label="HiFi-GAN.")
                tts_output_audio_no_enhanced_ft = gr.Audio(label="HiFi-GAN new.")
                reference_audio = gr.Audio(label="Reference Speech used.")

        def preprocess_dataset(audio_path, language, state_vars, progress=gr.Progress(track_tqdm=True)):
            # create a temp directory to save the dataset
            out_path = tempfile.TemporaryDirectory().name
            if audio_path is None:
                # ToDo: raise an error
                pass
            else:
                
                train_meta, eval_meta = format_audio_list([audio_path], target_language=language, out_path=out_path, gradio_progress=progress)

            state_vars = {}
            state_vars["train_csv"] = train_meta
            state_vars["eval_csv"] = eval_meta
            return "Dataset Processed!", state_vars

        prompt_compute_btn.click(
            fn=preprocess_dataset,
            inputs=[
                upload_file,
                lang,
                state_vars,
            ],
            outputs=[
                voice_ready,
                state_vars,
            ],
        )

        tts_btn.click(
            fn=run_tts,
            inputs=[
                lang,
                tts_text,
                state_vars,
                temperature,
                rms_norm_output,
            ],
            outputs=[tts_output_audio_no_enhanced, tts_output_audio_no_enhanced_ft],
        )

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        server_port=PORT,
        server_name="0.0.0.0"
    )

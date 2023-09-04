(docker_images)=
## Docker images
We provide docker images to be able to test TTS without having to setup your own environment.

### Using premade images
You can use premade images built automatically from the latest TTS version.

#### CPU version
```bash
docker pull ghcr.io/coqui-ai/tts-cpu
```
#### GPU version
```bash
docker pull ghcr.io/coqui-ai/tts
```

### Building your own image
```bash
docker build -t tts .
```

## Basic inference
Basic usage: generating an audio file from a text passed as argument.
You can pass any tts argument after the image name.

### CPU version
```bash
docker run --rm -v ~/tts-output:/root/tts-output ghcr.io/coqui-ai/tts-cpu --text "Hello." --out_path /root/tts-output/hello.wav
```
### GPU version
For the GPU version, you need to have the latest NVIDIA drivers installed.
With `nvidia-smi` you can check the CUDA version supported, it must be >= 11.8

```bash
docker run --rm --gpus all -v ~/tts-output:/root/tts-output ghcr.io/coqui-ai/tts --text "Hello." --out_path /root/tts-output/hello.wav --use_cuda true
```

## Start a server
Starting a TTS server:
Start the container and get a shell inside it.

### CPU version
```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --list_models #To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits
```

### GPU version
```bash
docker run --rm -it -p 5002:5002 --gpus all --entrypoint /bin/bash ghcr.io/coqui-ai/tts
python3 TTS/server/server.py --list_models #To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits --use_cuda true
```

Click [there](http://[::1]:5002/) and have fun with the server!
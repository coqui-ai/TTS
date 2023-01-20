while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container tts_container_$number on gpu $gpu and port $port";

nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=$gpu --runtime=nvidia --userns=host --shm-size 64G -v /work/leonardo.boulitreau/TTS:/workspace/coqui-tts/ -p $port --name tts_container$number multimodal-research-group-tts:latest /bin/bash
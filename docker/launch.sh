cd ..
CUDA_VISIBLE_DEVICES='2'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/GloVe,type=bind \
    --mount src=/media/user/data2,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /GloVe \
    litcoderr/glove:latest \
    bash -c "bash" \

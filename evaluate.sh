#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1 

tasks="task1 task2"
model=example/whisper/whisper_encoder.py
gpu_id=0
mode=train # test

. parse_options.sh

for task in $tasks; do
    echo "------- Run $task $mode --------"
    if [ $mode == 'test' ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id python3 -m xares_llm.run $model $task $task --mode test
    elif [ $mode == 'train' ]; then
        pkill -f xares
        accelerate launch --num_processes=8 --mixed-precision='bf16' -m xares_llm.run $model $task $task --mode train
    fi
done

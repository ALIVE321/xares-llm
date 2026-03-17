#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

# export NCCL_DEBUG=INFO

tasks="task1 task2"
model=example/whisper/whisper_encoder.py
gpu_id=0
mode=train # test
model_name=
exp_name=
benchmark_type=trainable-encoder  # freeze-encoder

. parse_options.sh

extra_args=""

if [ -n "$model_name" ]; then
    extra_args="$extra_args --model_args '{\"model_name\": \"$model_name\"}'"
fi
if [ -n "$exp_name" ]; then
    extra_args="$extra_args --exp_name $exp_name"
fi
if [ -n "$benchmark_type" ]; then
    extra_args="$extra_args --benchmark_type $benchmark_type"
fi

for task in $tasks; do
    echo "------- Run $task $mode --------"
    if [ $mode == 'test' ]; then
        eval CUDA_VISIBLE_DEVICES=$gpu_id python3 -m xares_llm.run $model $task $task --mode test $extra_args
    elif [ $mode == 'train' ]; then
        pkill -f xares
        eval accelerate launch --num_processes=8 --mixed-precision='bf16' -m xares_llm.run $model $task $task --mode train $extra_args
    fi
done

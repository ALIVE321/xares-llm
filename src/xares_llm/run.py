# Copyright 2025 Horizon Team, MiLM Plus, Xiaomi Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from typing import Dict, Any, List

import pandas as pd
from loguru import logger
from pathlib import Path

from xares_llm.task import (
    XaresLLMTask,
    XaresLLMTrainConfig,
    XaresLLMEvaluationConfig,
    AVAILABLE_EVALUATION_CONFIGS,
    AVAILABLE_TRAINING_CONFIGS,
)


def main(args):
    #Training
    overwrite_kwargs = args.args or {}
    if args.exp_name:
        overwrite_kwargs["exp_name"] = args.exp_name
    if args.benchmark_type:
        overwrite_kwargs["benchmark_type"] = args.benchmark_type
    train_config = XaresLLMTrainConfig.from_file_or_key(
        args.train_config, 
        encoder_path=args.encoder_path,
        model_kwargs=args.model_args,
        overwrite_kwargs=overwrite_kwargs,
    )
    eval_configs = XaresLLMEvaluationConfig.configs_from_file_or_key(args.eval_configs)
    if args.benchmark:
        logger.info("Using deterministic mode.")
        import torch
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Training with Train config \n{train_config}\n Eval config: {eval_configs}")
    runner = XaresLLMTask(train_config)

    mode = args.mode  # "all", "train", "test"

    if mode in ("all", "train"):
        runner.train_mlp()
        logger.info("Training completed.")

    if mode == "train":
        logger.info("Mode is 'train', skipping evaluation.")
        return

    # mode == "all" or "test"
    scores: List[Dict[str, Any]] = runner.run(eval_configs)

    logger.info("Scoring completed: All tasks scored.")

    # Print results
    df = pd.DataFrame(scores)
    df.sort_values(by="Task", inplace=True)

    new_row = pd.DataFrame(
        [
            {
                "Task": "Overall",
                "score": (df["score"] * df["weight"]).sum() / df["weight"].sum(),
                "weight": df["weight"].sum(),
            }
        ]
    )
    df = pd.concat((df, new_row), ignore_index=True)
    logger.info(f"\nResults:\n{df.to_string(index=False, float_format='%.3f')}")
    df.to_csv(Path(runner.output_dir) / "scores.tsv", sep="\t", index=False, float_format='%.3f')
    logger.info(f"\nFile saved: {Path(runner.output_dir) / 'scores.tsv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XARES-LLM")
    parser.add_argument(
        "encoder_path",
        type=str,
        help="Encoder path. e.g.: example/dummy/dummyencoder.py or path to the module (including class) e.g., example.dummy.dummyencoder.DummyEncoder",
    )
    parser.add_argument(
        "train_config",
        type=str,
        help=f"Tasks .yaml or predefined dataset. Datasets are: {list(AVAILABLE_TRAINING_CONFIGS.keys())}",
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "eval_configs",
        type=str,
        nargs="?",
        help=f"Evaluation Task .yaml. One Yaml can specify multiple datasets. By default we use the XARES-LLM datasets. Datasets are : {list(AVAILABLE_EVALUATION_CONFIGS.keys())} ",
        default="all",
    )
    parser.add_argument(
        "--model_args",
        type=lambda arg: json.loads(arg),
        help="Additional args passed to the encoder model. Format is JSON like: --model_args {'my_paramter1':2, 'my_paramter2':30}",
        default={},
    )
    parser.add_argument(
        "--args",
        type=lambda arg: json.loads(arg),
        help="Additional training args. Format is JSON like: --args {'per_device_train_batch_size':16, 'save_steps':30}",
        default={},
    )
    parser.add_argument(
        "--benchmark",
        action='store_true',
        help="Using deterministic mode for training/evaluation. Slows down training, but is reproducible",
        default=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name for output directory. If not set, derived from encoder path.",
    )
    parser.add_argument(
        "--benchmark_type",
        type=str,
        choices=["freeze-encoder", "trainable-encoder"],
        default="freeze-encoder",
        help="Mode for training: 'freeze-encoder' (frozen encoder + LoRA LLM), 'trainable-encoder' (trainable encoder + frozen LLM)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "train", "test"],
        default="all",
        help="Run mode: 'all' (train+test), 'train' (train only), 'test' (test only, requires existing checkpoint)",
    )
    args = parser.parse_args()
    main(args)

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

from loguru import logger
from transformers import Trainer, ProgressCallback
from xares_llm.audiowebdataset import AudioTextTokenWebdataset


class LoguruMetricsCallback(ProgressCallback):
    def __init__(self):
        super().__init__()
        self._predict_step_count = 0
        self._predict_log_interval = 100

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, float):
                    shallow_logs[k] = f"{v:.4g}"
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            log = ", ".join([f"{key} = {value}" for key, value in shallow_logs.items()])
            logger.info(str(log))

    def on_prediction_step(self, args, state, control, **kwargs):
        self._predict_step_count += 1
        if self._predict_step_count % self._predict_log_interval == 0:
            logger.info(f"Predict step {self._predict_step_count}")


class XaresLLMTrainerEvaluator(Trainer):
    def __init__(self, *args, **kwargs):
        self.train_data_object: AudioTextTokenWebdataset = kwargs.pop("train_data_object", None)
        self.eval_data_object: AudioTextTokenWebdataset = kwargs.pop("eval_data_object", None)
        train_dataset = self.train_data_object.create_dataset() if self.train_data_object else None
        super().__init__(train_dataset=train_dataset, *args, **kwargs)
        self.remove_callback(ProgressCallback)
        self.add_callback(LoguruMetricsCallback)

    def get_train_dataloader(self):
        return self.train_data_object.create_dataloader()

    def get_test_dataloader(self, eval_dataset: AudioTextTokenWebdataset, *args, **kwargs):
        return eval_dataset.create_dataloader()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        generated_ids = model.generate(**inputs, repetition_penalty=1.05, max_new_tokens=150, do_sample=False, temperature=1.0, top_k=50, top_p=1.0)
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.to(generated_ids.device)
        return (None, generated_ids, labels)

    def _save(self, output_dir, state_dict=None):
        """Override _save to handle models with shared tensors gracefully."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        # Try saving with safe_serialization first (default, modern format)
        try:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
            )
        except RuntimeError as e:
            # If we get a shared tensor error, fall back to safe_serialization=False
            if "shared tensors" in str(e):
                logger.warning(
                    "Model contains shared tensors (e.g., nn.Sequential wrapping existing modules). "
                    "Saving with safe_serialization=False to handle this."
                )
                self.model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=False,
                )
            else:
                # Re-raise if it's a different error
                raise
        
        # Save the tokenizer if present
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

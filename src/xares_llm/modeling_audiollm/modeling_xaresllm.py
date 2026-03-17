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

import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType

from xares_llm.audio_encoder_checker import check_audio_encoder
from xares_llm.modeling_audiollm.configuration_xaresllm import XaresLLMModelConfig
from xares_llm.utils import attr_from_module, attr_from_py_path

# 目标输出帧率 6.25Hz，即帧间隔 160ms
TARGET_HOP_SIZE_IN_MS = 160


class EncoderProjector(nn.Module):
    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 1):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.size()
        if self.downsample_rate > 1:
            discard = seq_len % self.downsample_rate
            if discard > 0:
                x = x[:, :-discard, :]
                seq_len = x.size(1)
            x = x.contiguous().view(
                batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
            )
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class XaresLLMModel(PreTrainedModel, nn.Module):
    config_class = XaresLLMModelConfig

    def __init__(self, config: XaresLLMModelConfig) -> None:
        super().__init__(config)
        self.config = config
        is_trainable_encoder = config.benchmark_type == "trainable-encoder"

        if Path(self.config.audio_encoder_name).is_file():
            audio_encoder = attr_from_py_path(self.config.audio_encoder_name, endswith="Encoder")(
                **self.config.audio_encoder_params, train=is_trainable_encoder)
        else:
            audio_encoder = attr_from_module(self.config.audio_encoder_name)(
                **self.config.audio_encoder_params, train=is_trainable_encoder)
        try:
            audio_encoder_parameters = list(audio_encoder.parameters())
            if len(audio_encoder_parameters) > 0:
                device_type = audio_encoder_parameters[0].device.type
                if device_type != "meta":  # When using .from_pretrained, device is meta
                    check_audio_encoder(audio_encoder)
        except Exception as e:
            logger.exception(e)
            return  # Error is raised inside
        self.audio_encoder = audio_encoder

        if is_trainable_encoder:
            self.audio_encoder.train()
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        else:
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        decoder = AutoModelForCausalLM.from_pretrained(config.decoder_type)
        if is_trainable_encoder:
            for param in decoder.parameters():
                param.requires_grad = False
            self.decoder = decoder
            logger.info("[trainable-encoder] freeze LLM with no LoRA.")
        else:
            peft_config = LoraConfig(
                target_modules="all-linear",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
            )
            self.decoder = get_peft_model(decoder, peft_config)
            self.decoder.print_trainable_parameters()

        downsample_rate = max(1, round(TARGET_HOP_SIZE_IN_MS / self.audio_encoder.hop_size_in_ms))
        logger.info(f"EncoderProjector: encoder_hop={self.audio_encoder.hop_size_in_ms}ms, "
                    f"target_hop={TARGET_HOP_SIZE_IN_MS}ms, downsample_rate={downsample_rate}")
        self.audio_projector = EncoderProjector(
            self.audio_encoder.output_dim, self.decoder.config.hidden_size, downsample_rate
        )

    def merge_and_unload(self):
        self.decoder = self.decoder.merge_and_unload()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration as e:
            logger.error("Rerun the script with 'accelerate launch -m xares_llm.run'")
            raise e

    def _prepare_multimodal_inputs(self, audio, audio_attention_mask, input_ids, attention_mask, labels=None):
        audio = audio.to(self.device)
        audio_attention_mask = audio_attention_mask.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        final_audio_attention_mask = None
        if self.config.benchmark_type == "trainable-encoder":
            audio_feature, final_audio_attention_mask = self.audio_encoder(audio, audio_attention_mask)
            audio_feature = audio_feature.to(self.device)
        else:
            with torch.no_grad():
                audio_feature, final_audio_attention_mask = self.audio_encoder(audio, audio_attention_mask)
                audio_feature = audio_feature.to(self.device)  # returned tensor might be on cpu
        audio_feature = self.audio_projector(audio_feature)
        # 下采样后更新 attention_mask 的时间维度
        ds = self.audio_projector.downsample_rate
        if ds > 1 and final_audio_attention_mask is not None:
            T = final_audio_attention_mask.shape[1]
            discard = T % ds
            if discard > 0:
                final_audio_attention_mask = final_audio_attention_mask[:, :-discard]
            final_audio_attention_mask = final_audio_attention_mask[:, ::ds]
        if final_audio_attention_mask is None:
            final_audio_attention_mask = torch.ones(*audio_feature.shape[:2], device=attention_mask.device)
        # An error occurs if .get_input_embeddings() is used with self.input_embeds = ...
        input_embeds = self.decoder.get_input_embeddings()(input_ids)  # Int -> Float

        # concatenate all data: [AUDIO, TEXT]
        input_embeds = torch.cat((audio_feature, input_embeds), dim=1)
        zero_audio_targets = torch.full(
            audio_feature.shape[:2], device=audio_feature.device, dtype=torch.int, fill_value=-100
        )
        if labels is not None:
            labels = torch.cat((zero_audio_targets, labels), dim=1)
        else:
            labels = None
        attention_mask = torch.cat(
            (final_audio_attention_mask, attention_mask),
            dim=1,
        )
        return input_embeds, attention_mask, labels

    def forward(self, audio, audio_attention_mask, input_ids, attention_mask, labels, **kwargs):
        input_embeds, attention_mask, labels = self._prepare_multimodal_inputs(
            audio, audio_attention_mask, input_ids, attention_mask, labels
        )
        return self.decoder(input_ids=None, inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask)

    @torch.no_grad()
    def generate(self, audio, audio_attention_mask, input_ids, attention_mask, **gen_kwargs):
        input_embeds, attention_mask, _ = self._prepare_multimodal_inputs(
            audio, audio_attention_mask, input_ids, attention_mask, labels=None
        )
        return self.decoder.generate(
            input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask, **gen_kwargs
        )

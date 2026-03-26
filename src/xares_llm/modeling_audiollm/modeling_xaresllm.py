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
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType

from xares_llm.audio_encoder_checker import check_audio_encoder
from xares_llm.modeling_audiollm.configuration_xaresllm import XaresLLMModelConfig
from xares_llm.utils import attr_from_module, attr_from_py_path

# 目标输出帧率 6.25Hz，即帧间隔 160ms
TARGET_HOP_SIZE_IN_MS = 160

# 音频占位符：在 chat template 的 user message 中标记音频位置，模型前向时会将其替换为音频 embedding
AUDIO_PLACEHOLDER = "<audio>"


class EncoderProjector(nn.Module):
    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 1):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(encoder_dim, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)
        if self.downsample_rate > 1:
            self.avgpool = nn.AvgPool1d(kernel_size=self.downsample_rate, stride=self.downsample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if self.downsample_rate > 1:
            x = x.transpose(1, 2)  # (B, D, T)
            x = self.avgpool(x)     # (B, D, T//ds)
            x = x.transpose(1, 2)  # (B, T//ds, D)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def get_num_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trained


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
        # self.audio_projector = nn.Linear(self.audio_encoder.output_dim, self.decoder.config.hidden_size)

        encoder_params, encoder_train_params = get_num_params(self.audio_encoder)
        adapter_params, adapter_train_params = get_num_params(self.audio_projector)
        decoder_params, decoder_train_params = get_num_params(self.decoder)
        logger.info(f"[encoder] params={encoder_params/1e6:.3f}M trainable={encoder_train_params/1e6:.3f}M")
        logger.info(f"[adapter] params={adapter_params/1e6:.3f}M trainable={adapter_train_params/1e6:.3f}M")
        logger.info(f"[decoder] params={decoder_params/1e6:.3f}M trainable={decoder_train_params/1e6:.3f}M")

    def train(self, mode: bool = True):
        """重写 train()，freeze-encoder 模式下始终保持 audio_encoder 为 eval()。
        
        HuggingFace Trainer 在训练循环中会调用 model.train()，递归将所有子模块
        切换到 training 模式。对于 frozen encoder，这会导致其内部的 Dropout/BatchNorm
        等层意外激活，因此需要在此处强制恢复 eval() 状态。
        """
        super().train(mode)
        if self.config.benchmark_type != "trainable-encoder":
            self.audio_encoder.eval()
        return self

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

        ds = getattr(self.audio_projector, 'downsample_rate', 1)
        if ds > 1 and final_audio_attention_mask is not None:
            mask_float = final_audio_attention_mask.float().unsqueeze(1)  # (B, 1, T)
            mask_float = nn.functional.avg_pool1d(mask_float, kernel_size=ds, stride=ds)  # (B, 1, T//ds)
            final_audio_attention_mask = (mask_float.squeeze(1) > 0).long()  # (B, T//ds)
        if final_audio_attention_mask is None:
            final_audio_attention_mask = torch.ones(*audio_feature.shape[:2], device=attention_mask.device)

        # 获取文本 embedding
        input_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B, T_text, D)

        # 查找 AUDIO_PLACEHOLDER token 并替换为音频 embedding
        # 如果 input_ids 中包含占位符 token，则在占位符位置插入音频 embedding
        # 否则 fallback 到原始的 [AUDIO, TEXT] concat 模式
        if hasattr(self, '_audio_placeholder_token_ids') and self._audio_placeholder_token_ids is not None:
            placeholder_ids = self._audio_placeholder_token_ids  # List[int]
            placeholder_len = len(placeholder_ids)
            new_embeds_list, new_labels_list, new_attn_list = [], [], []
            B = input_ids.shape[0]
            for b in range(B):
                # 先截掉 tokenizer padding 的尾部，避免模型层二次 padding 后序列冗余
                valid_len = attention_mask[b].sum().item()
                b_ids = input_ids[b, :valid_len].tolist()
                b_embeds = input_embeds[b, :valid_len]
                b_attn = attention_mask[b, :valid_len]
                b_labels = labels[b, :valid_len] if labels is not None else None

                # 查找占位符序列的起始位置
                placeholder_pos = -1
                for i in range(len(b_ids) - placeholder_len + 1):
                    if b_ids[i:i + placeholder_len] == placeholder_ids:
                        placeholder_pos = i
                        break

                if placeholder_pos >= 0:
                    # 在占位符位置插入音频 embedding
                    # [前缀文本 embedding] + [音频 embedding] + [后缀文本 embedding]
                    pre_embed = b_embeds[:placeholder_pos]  # 占位符之前
                    post_embed = b_embeds[placeholder_pos + placeholder_len:]  # 占位符之后
                    audio_embed = audio_feature[b]  # (T_audio, D)
                    merged = torch.cat([pre_embed, audio_embed, post_embed], dim=0)
                    new_embeds_list.append(merged)

                    # attention_mask: 前缀 + 音频mask + 后缀
                    pre_attn = b_attn[:placeholder_pos]
                    post_attn = b_attn[placeholder_pos + placeholder_len:]
                    audio_attn = final_audio_attention_mask[b]
                    new_attn_list.append(torch.cat([pre_attn, audio_attn, post_attn], dim=0))

                    # labels: 前缀(-100) + 音频(-100) + 后缀(原labels)
                    if b_labels is not None:
                        pre_labels = b_labels[:placeholder_pos]
                        post_labels = b_labels[placeholder_pos + placeholder_len:]
                        audio_labels = torch.full((audio_embed.shape[0],), -100, device=labels.device, dtype=labels.dtype)
                        new_labels_list.append(torch.cat([pre_labels, audio_labels, post_labels], dim=0))
                else:
                    # fallback: 没找到占位符，使用 [AUDIO, TEXT] concat
                    new_embeds_list.append(torch.cat([audio_feature[b], b_embeds], dim=0))
                    audio_attn = final_audio_attention_mask[b]
                    new_attn_list.append(torch.cat([audio_attn, b_attn], dim=0))
                    if b_labels is not None:
                        audio_labels = torch.full((audio_feature[b].shape[0],), -100, device=labels.device, dtype=labels.dtype)
                        new_labels_list.append(torch.cat([audio_labels, b_labels], dim=0))

            # padding 到同一长度
            max_len = max(e.shape[0] for e in new_embeds_list)
            D = input_embeds.shape[-1]
            padded_embeds = torch.zeros(B, max_len, D, device=input_embeds.device, dtype=input_embeds.dtype)
            padded_attn = torch.zeros(B, max_len, device=attention_mask.device, dtype=attention_mask.dtype)
            padded_labels = torch.full((B, max_len), -100, device=input_ids.device, dtype=torch.long) if labels is not None else None

            for b in range(B):
                L = new_embeds_list[b].shape[0]
                padded_embeds[b, :L] = new_embeds_list[b]
                padded_attn[b, :L] = new_attn_list[b]
                if padded_labels is not None:
                    padded_labels[b, :L] = new_labels_list[b]

            return padded_embeds, padded_attn, padded_labels
        else:
            # 原始模式：简单 concat [AUDIO, TEXT]
            input_embeds = torch.cat((audio_feature, input_embeds), dim=1)
            zero_audio_targets = torch.full(
                audio_feature.shape[:2], device=audio_feature.device, dtype=torch.long, fill_value=-100
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
        return self.decoder(input_ids=None, inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask, **kwargs)

    @torch.no_grad()
    def generate(self, audio, audio_attention_mask, input_ids, attention_mask, **gen_kwargs):
        input_embeds, attention_mask, _ = self._prepare_multimodal_inputs(
            audio, audio_attention_mask, input_ids, attention_mask, labels=None
        )
        return self.decoder.generate(
            input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask, **gen_kwargs
        )

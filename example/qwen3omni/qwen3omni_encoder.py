from pathlib import Path

import torch
import torch.nn as nn
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()


def _get_feat_extract_output_lengths(input_lengths):
    """计算卷积下采样后的输出长度（与官方 modeling_qwen3omni_moe.py 第145行一致）。"""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3OmniEncoder(nn.Module):
    """从 Qwen3-Omni 提取 audio encoder（Whisper-based），适配 xares-llm 评测接口。

    提取 audio_tower 并禁用投影层（proj1/act/proj2），
    输出维度为 encoder 内部的 d_model。
    """

    def __init__(self, model_name="model/Qwen3-Omni-30B-A3B-Instruct", train=True):
        super().__init__()

        if not Path(model_name).exists():
            self.download_from_hub(model_name)

        full_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu" if train else None,
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_name, fix_mistral_regex=True)

        # 提取 audio encoder
        self.model = full_model.thinker.audio_tower
        # 禁用投影层，只取 encoder 的 d_model 维度表征
        self.model.proj1 = nn.Identity()
        self.model.proj2 = nn.Identity()
        self.model.act = nn.Identity()

        # 释放 CPU 上的完整模型，再将 audio_tower 搬到 GPU
        del full_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # self.model = self.model.to("cuda")
        self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.output_dim = self.model.config.d_model
        self.hop_size_in_ms = (
            self.processor.feature_extractor.hop_length / self.sampling_rate * 1000
        )

    @property
    def device(self):
        return self.model.device

    @classmethod
    def download_from_hub(cls, model_name: str, output_root: str = "."):
        """下载模型到本地，避免多进程下载冲突。"""
        output_dir = Path(output_root) / model_name
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.float32, device_map="cpu",
            low_cpu_mem_usage=False)
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_name, fix_mistral_regex=True)
        processor.save_pretrained(output_dir)
        model.save_pretrained(output_dir)

    def _extract_features(self, audio_np):
        """用 processor 提取 mel 频谱特征，返回 (input_features, feature_lens)。"""
        features = self.processor.feature_extractor(
            audio_np,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        input_features = features["input_features"].to(self.device)
        attention_mask = features["attention_mask"]
        feature_lens = attention_mask.sum(dim=-1).long().to(self.device)
        return input_features, feature_lens

    def _forward_encoder(self, input_features, feature_attention_mask):
        """严格按照官方 get_audio_features + audio_tower.forward 的调用方式。

        官方流程：
        1. 用 feature_attention_mask 将 batch 中所有有效帧拼成 (D, total_T) 的 2D tensor
        2. 将 feature_lens（每条音频的真实 mel 长度）传入 audio_tower
        3. audio_tower 返回 (total_output_T, hidden_dim) 的 2D tensor
        4. 按 _get_feat_extract_output_lengths 算出每条音频的输出长度，拆分并 pad 成 batch
        """
        # 与官方 get_audio_features 完全一致：拼成 (D, total_T)
        audio_feature_lengths = feature_attention_mask.sum(dim=1)
        merged_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        merged_features = merged_features.to(self.model.dtype)

        audio_outputs = self.model(
            merged_features,
            feature_lens=audio_feature_lengths,
        )
        # audio_tower 返回 (total_output_T, hidden_dim) 的 2D tensor
        hidden_states = audio_outputs.last_hidden_state

        # 按 _get_feat_extract_output_lengths 拆分成各条音频的输出
        output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
        output_list = hidden_states.split(output_lengths.tolist(), dim=0)

        # pad 到相同长度后组成 batch
        max_t = output_lengths.amax().item()
        padded = []
        for o in output_list:
            if o.shape[0] < max_t:
                pad = torch.zeros(max_t - o.shape[0], o.shape[1],
                                  dtype=o.dtype, device=o.device)
                o = torch.cat([o, pad], dim=0)
            padded.append(o.unsqueeze(0))
        return torch.cat(padded, dim=0), output_lengths

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(audio, torch.Tensor)
        audio = audio.cpu().numpy()
        if audio.ndim == 1:
            audio_list = [audio]
        elif audio.ndim == 2:
            audio_list = [a for a in audio]
        else:
            raise ValueError("Audio tensor must be 1D (single sequence) or 2D (batch of sequences).")

        input_features, feature_lens = self._extract_features(audio)
        # 构造 feature_attention_mask：(B, T_padded)，1 表示有效帧
        feature_attention_mask = length_to_mask(feature_lens)

        output, feature_lengths = self._forward_encoder(input_features, feature_attention_mask)

        trim_length = feature_lengths.amax()
        attention_mask = length_to_mask(feature_lengths)
        return output[:, :trim_length, :], attention_mask


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    enc = Qwen3OmniEncoder(model_name=model_name)
    print("Loading Qwen3Omni audio encoder ---- ")
    q, mask = enc(torch.randn(2, 32000), length_to_mask(torch.tensor([32000, 16000])))
    print(f"output: {q.shape}, mask: {mask.shape}, output_dim: {enc.output_dim}")

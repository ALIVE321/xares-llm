import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor


def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()


class Qwen3ASREncoder(nn.Module):
    """从 Qwen3-ASR 提取 audio encoder，适配 xares-llm 评测接口。

    提取 audio_tower 并禁用投影层（proj1/act/proj2），
    输出维度为 encoder 内部的 d_model。
    """

    def __init__(self, model_name="model/Qwen3-ASR-1.7B", train=True, **kwargs):
        super().__init__()

        if not Path(model_name).exists():
            self.download_from_hub(model_name)

        full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=None,
        )
        self.processor = Qwen3ASRProcessor.from_pretrained(
            model_name, fix_mistral_regex=True)

        # 提取 audio encoder
        self.model = full_model.thinker.audio_tower
        # 禁用投影层，只取 encoder 的 d_model 维度表征
        self.model.proj1 = nn.Identity()
        self.model.proj2 = nn.Identity()
        self.model.act = nn.Identity()

        del full_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if not train:
            self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.output_dim = self.model.config.d_model
        self.hop_size_in_ms = (
            self.processor.feature_extractor.hop_length / self.sampling_rate * 1000
        ) * 8   # 8x downsampling

    @property
    def device(self):
        return self.model.device

    @classmethod
    def download_from_hub(cls, model_name: str, output_root: str = "."):
        """下载模型到本地，避免多进程下载冲突。"""
        output_dir = Path(output_root) / model_name
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.float32, device_map="cpu")
        processor = Qwen3ASRProcessor.from_pretrained(
            model_name, fix_mistral_regex=True)
        processor.save_pretrained(output_dir)
        if hasattr(model, "generation_config"):
            model.generation_config.temperature = 1.0
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

    def _forward_encoder(self, input_features, feature_lens):
        """逐条送入 encoder（Qwen3ASR audio_tower 不支持 batch 推理）。"""
        outputs = []
        for feat, flen in zip(input_features, feature_lens):
            feat_trimmed = feat[:, :flen].to(self.model.dtype)
            audio_output = self.model(
                feat_trimmed,
                feature_lens=flen.unsqueeze(0),
            )
            o = audio_output.last_hidden_state
            if o.ndim == 2:
                o = o.unsqueeze(0)
            outputs.append(o)

        # 各条输出时间维度不同，需 pad 到最大长度后才能 cat
        max_t = max(o.shape[1] for o in outputs)
        padded = []
        for o in outputs:
            if o.shape[1] < max_t:
                pad = torch.zeros(o.shape[0], max_t - o.shape[1], o.shape[2],
                                  dtype=o.dtype, device=o.device)
                o = torch.cat([o, pad], dim=1)
            padded.append(o)
        return torch.cat(padded, dim=0)

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio_attention_mask is not None:
            audio_lengths = audio_attention_mask.sum(dim=-1).long()
        else:
            audio_lengths = torch.tensor([audio.shape[-1]] * audio.shape[0], dtype=torch.long)
        audio_list = [audio[i, :audio_lengths[i]].cpu().numpy() for i in range(audio.shape[0])]

        input_features, feature_lens = self._extract_features(audio_list)
        output = self._forward_encoder(input_features, feature_lens)

        # 用真实 mel 长度算下采样后长度（与 _get_feat_extract_output_lengths 一致）
        remainder = feature_lens % 100
        feat_lengths = (remainder - 1) // 2 + 1
        feature_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13

        trim_length = feature_lengths.amax()
        attention_mask = length_to_mask(feature_lengths)
        return output[:, :trim_length, :], attention_mask


if __name__ == "__main__":
    model_name = "model/Qwen3-ASR-1.7B"
    print("Loading Qwen3ASR audio encoder ---- ")
    enc = Qwen3ASREncoder(model_name=model_name)
    q, mask = enc(torch.randn(2, 32000), length_to_mask(torch.tensor([32000, 16000])))
    print(f"output: {q.shape}, mask: {mask.shape}, output_dim: {enc.output_dim}, len: {mask.sum(-1)}")

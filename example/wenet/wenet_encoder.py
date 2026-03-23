import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from src import WenetTransformerEncoderModel
except Exception as e:
    raise RuntimeError(f"Failed to load wenet model files due to:\n{e}")


def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()


class WenetEncoder(nn.Module):
    def __init__(
        self,
        model_name="model/ssl_hubert_v2.0.1",
        train=True,
        num_mel_bins=128,
        **kwargs
    ):
        super().__init__()

        self.model = WenetTransformerEncoderModel.from_pretrained(
            model_name, map_location='cpu')
        if not train:
            self.model.eval()

        self.sampling_rate = 16000
        self.output_dim = self.model.encoder_out_dim
        self.num_mel_bins = num_mel_bins

        self.n_fft = 400         # 25ms @ 16kHz
        self.hop_length = 160    # 10ms @ 16kHz
        self.hop_size_in_ms = 40 # 25Hz

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _compute_log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Whisper 风格 log mel spectrogram 提取（参考 openai-whisper 实现）。

        Args:
            waveform: (T,) 单条音频的 1D tensor

        Returns:
            log_spec: (T_feat, num_mel_bins) mel 特征
        """
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(waveform, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = torch.from_numpy(
            librosa.filters.mel(
                sr=self.sampling_rate, n_fft=self.n_fft,
                n_mels=self.num_mel_bins)
        ).to(dtype=magnitudes.dtype, device=magnitudes.device)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.transpose(0, 1)  # (T_feat, num_mel_bins)

    def _extract_features(self, audio: torch.Tensor, audio_attention_mask=None):
        """从 raw waveform 提取 whisper 风格的 log mel spectrogram 特征。

        Args:
            audio: (B, T) raw waveform tensor
            audio_attention_mask: (B, T) attention mask，标记有效采样点

        Returns:
            features: (B, T_feat, D) mel 特征，padding 部分为 0
            feature_lens: (B,) 每条音频的真实帧数
        """
        B = audio.shape[0]
        if audio_attention_mask is not None:
            audio_lengths = audio_attention_mask.sum(dim=-1).long()
        else:
            audio_lengths = torch.tensor([audio.shape[-1]] * B, dtype=torch.long)

        feat_list = []
        for i in range(B):
            valid_len = audio_lengths[i].item()
            wav = audio[i, :valid_len]
            feat = self._compute_log_mel_spectrogram(wav)  # (T_feat, D)
            feat_list.append(feat)

        feature_lens = torch.tensor([f.shape[0] for f in feat_list], dtype=torch.long)
        features = pad_sequence(feat_list, batch_first=True, padding_value=0.0)

        return features, feature_lens

    def _forward_encoder(self, features, feature_lens):
        """送入 wenet encoder，返回 (encoder_out, encoder_out_lens)。

        Args:
            features: (B, T_feat, D) mel 特征
            feature_lens: (B,) 每条音频的真实帧数

        Returns:
            encoder_out: (B, T', output_dim)
            encoder_out_lens: (B,) 每条音频的输出长度
        """
        enc_out = self.model.forward(
            features.to(self.device), feature_lens.to(self.device))
        return enc_out["encoder_out"], enc_out["encoder_out_lens"]

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        features, feature_lens = self._extract_features(audio, audio_attention_mask)
        output, output_lens = self._forward_encoder(features, feature_lens)

        trim_length = output_lens.amax()
        attention_mask = length_to_mask(output_lens)
        return output[:, :trim_length, :], attention_mask


if __name__ == "__main__":
    model_name = "model/ssl_hubert_v2.0.1"
    print("Loading WenetEncoder ---- ")
    enc = WenetEncoder(model_name=model_name)
    q, mask = enc(
        torch.randn(2, 32000),
        length_to_mask(torch.tensor([32000, 16000]), max_len=32000),
    )
    print(f"output: {q.shape}, mask: {mask.shape}, output_dim: {enc.output_dim}, len: {mask.sum(-1)}")

import torch
from transformers import WhisperModel, WhisperProcessor


def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()


class WhisperEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-medium", train=True, **kwargs):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).get_encoder()
        self.output_dim = self.model.config.d_model
        self.hop_size_in_ms = 20    # 50Hz
        if not train:
            self.model.eval()

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Since feature extraction is on cpu this is super slow
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio_attention_mask is not None:
            audio_lengths = audio_attention_mask.sum(dim=-1).long()
        else:
            audio_lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long, device=audio.device)
        audio_list = [audio[i, :audio_lengths[i]].cpu().numpy() for i in range(audio.shape[0])]

        hop_length = self.processor.feature_extractor.hop_length
        # Whisper processor 会将音频截断到 n_samples（默认 480000，即 30 秒），
        # 因此 mel 帧数上限为 n_samples // hop_length = 3000，需要 clamp 防止越界。
        max_mel_length = self.processor.feature_extractor.n_samples // hop_length
        mel_lengths = (audio_lengths // hop_length).clamp(max=max_mel_length)
        feature_lengths = (mel_lengths - 1) // 2 + 1
        trim_length = feature_lengths.amax().item()
        attention_mask = length_to_mask(feature_lengths)

        features = self.processor(audio_list, sampling_rate=16000, return_tensors="pt").to(self.model.device)
        output = self.model(**features).last_hidden_state
        return output[:, :trim_length, :], attention_mask


if __name__ == "__main__":
    print("Loading WhisperEncoder ---- ")
    model_name = 'model/whisper/whisper-medium'
    enc = WhisperEncoder(model_name)
    q, mask = enc(
        torch.randn(2, 32000),
        length_to_mask(torch.tensor([32000, 16000]), max_len=32000),
    )
    print(f"output: {q.shape}, mask: {mask.shape}, output_dim: {enc.output_dim}, len: {mask.sum(-1)}")

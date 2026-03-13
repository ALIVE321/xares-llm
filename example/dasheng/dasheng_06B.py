import torch
from transformers import AutoModel, AutoFeatureExtractor


class DashengBaseEncoder(torch.nn.Module):
    def __init__(self, model_name="mispeech/dasheng-0.6B"):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.output_dim = self.model.config.encoder_kwargs["embed_dim"]

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        features = self.processor(audio, return_tensors="pt")
        output = self.model(**features).hidden_states
        return output, None


if __name__ == "__main__":
    mdl = DashengBaseEncoder()
    q = mdl(torch.randn(1, 160000))
    print(q.shape)

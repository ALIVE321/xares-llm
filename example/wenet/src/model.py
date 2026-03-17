"""
独立的 WeNet Transformer/Conformer encoder 模型封装。

仅依赖:
- torch
- wenet (用于 TransformerEncoder / ConformerEncoder 网络结构定义)
- safetensors (可选, 用于加载 .safetensors 格式权重)

不依赖 auden 框架。

使用方法:
    from model import WenetTransformerEncoderModel
    model = WenetTransformerEncoderModel.from_pretrained("/path/to/model_dir")
    out = model(features, feature_lens)  # -> dict{"encoder_out", "encoder_out_lens"}
"""

import logging
import os

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .model_config import WenetTransformerConfig
except ImportError:
    from model_config import WenetTransformerConfig

# 直接复用 wenet 官方的 encoder 实现
from wenet.transformer.encoder import (
    ConformerEncoder,
    TransformerEncoder,
)


class WenetTransformerEncoderModel(nn.Module):
    """WeNet Transformer/Conformer encoder 的独立封装。

    直接使用 wenet 官方的 TransformerEncoder / ConformerEncoder，
    无需 auden 框架，未来 wenet 更新后只需升级 wenet 包即可。

    - forward(x, x_lens) 返回 dict: {'encoder_out': Tensor, 'encoder_out_lens': Tensor}
    - encoder_out_dim: encoder 输出维度
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        module_name: str | None = None,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
        dtype=None,
        device=None,
    ) -> "WenetTransformerEncoderModel":
        """从预训练 checkpoint 加载 WeNet encoder。

        Args:
            model_path: 包含权重和 config 的目录，或直接指向 .pt/.safetensors 文件。
            module_name: 如果从复合模型加载，指定包含 encoder 权重的子模块名称
                (例如 "audio_encoder" 或 "encoder")。为 None 时自动检测。
            strict: 传递给 load_state_dict。
            map_location: 传递给 torch.load。
            dtype: 加载后可选的 dtype 转换。
            device: 加载后可选的 device 转移。
        """
        # 解析模型目录和权重文件
        if os.path.isdir(model_path):
            model_dir = model_path
            weight_path = None
            for ext in (".safetensors", ".pt"):
                for name in ("pretrained", "model"):
                    p = os.path.join(model_dir, f"{name}{ext}")
                    if os.path.exists(p):
                        weight_path = p
                        break
                if weight_path is not None:
                    break
            assert weight_path is not None, (
                f"在 {model_dir} 下未找到权重文件 "
                f"['pretrained.safetensors','model.safetensors','pretrained.pt','model.pt']"
            )
        else:
            weight_path = model_path
            model_dir = os.path.dirname(model_path)

        config = WenetTransformerConfig.from_pretrained(model_dir)

        # 加载权重
        ext = os.path.splitext(weight_path)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file as safe_load_file
            device_arg = str(map_location) if isinstance(map_location, torch.device) else map_location
            state_obj = safe_load_file(weight_path, device=device_arg)
        else:
            state_obj = torch.load(weight_path, map_location=map_location)

        if isinstance(state_obj, dict) and "state_dict" in state_obj:
            state_dict = state_obj["state_dict"]
        else:
            state_dict = state_obj

        logging.info(f"已加载权重 type={config.model_type} from {weight_path}")

        # 自动检测子模块前缀
        detected_module = module_name
        if detected_module is None:
            for candidate_name in ("audio_encoder", "speech_encoder", "encoder"):
                if any(k.startswith(f"{candidate_name}.") for k in state_dict.keys()):
                    # 仅当 config 中有对应 sub_config 属性时才检测
                    if hasattr(config, f"{candidate_name}_config"):
                        detected_module = candidate_name
                        break

        if detected_module is not None:
            sub_config = getattr(config, f"{detected_module}_config", None)
            assert sub_config is not None, (
                f"Config 中不包含 '{detected_module}_config'"
            )
        else:
            sub_config = config

        model = cls(sub_config)

        # 剥离可能的前缀: DDP 'module.' 和 父模块前缀
        def _strip_prefix(d: dict, prefix: str):
            if any(k.startswith(prefix) for k in d.keys()):
                return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
            return None

        candidate = state_dict
        stripped = _strip_prefix(candidate, "module.")
        if stripped:
            candidate = stripped

        if detected_module is not None:
            for pfx in [f"{detected_module}.", "audio_encoder.", "speech_encoder.", "encoder."]:
                stripped = _strip_prefix(candidate, pfx)
                if stripped:
                    candidate = stripped
                    break
        state_dict = candidate

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if not strict and (missing or unexpected):
            logging.warning(f"from_pretrained missing keys: {missing}, unexpected keys: {unexpected}")

        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model

    def __init__(self, config: WenetTransformerConfig) -> None:
        super().__init__()
        self.config = config

        encoder_type = config.encoder_type

        # 公共参数
        common_params = {
            "input_size": config.input_size,
            "output_size": config.output_size,
            "attention_heads": config.attention_heads,
            "linear_units": config.linear_units,
            "num_blocks": config.num_blocks,
            "dropout_rate": config.dropout_rate,
            "positional_dropout_rate": config.positional_dropout_rate,
            "attention_dropout_rate": config.attention_dropout_rate,
            "input_layer": config.input_layer,
            "pos_enc_layer_type": config.pos_enc_layer_type,
            "normalize_before": config.normalize_before,
            "static_chunk_size": config.static_chunk_size,
            "use_dynamic_chunk": config.use_dynamic_chunk,
            "global_cmvn": config.global_cmvn,
            "use_dynamic_left_chunk": config.use_dynamic_left_chunk,
            "query_bias": config.query_bias,
            "key_bias": config.key_bias,
            "value_bias": config.value_bias,
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_sdpa": config.use_sdpa,
            "layer_norm_type": config.layer_norm_type,
            "norm_eps": config.norm_eps,
            "n_kv_head": config.n_kv_head,
            "head_dim": config.head_dim,
            "mlp_type": config.mlp_type,
            "mlp_bias": config.mlp_bias,
            "n_expert": config.n_expert,
            "n_expert_activated": config.n_expert_activated,
        }

        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                **common_params,
                positionwise_conv_kernel_size=config.positionwise_conv_kernel_size,
                macaron_style=config.macaron_style,
                selfattention_layer_type=config.selfattention_layer_type,
                activation_type=config.activation_type,
                use_cnn_module=config.use_cnn_module,
                cnn_module_kernel=config.cnn_module_kernel,
                causal=config.causal,
                cnn_module_norm=config.cnn_module_norm,
                conv_bias=config.conv_bias,
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                **common_params,
                activation_type=config.activation_type,
                selfattention_layer_type=config.selfattention_layer_type,
            )
        else:
            raise ValueError(f"未知的 encoder_type: {encoder_type}")

        self.encoder_out_dim = self.encoder.output_size()

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        return_dict: bool = True,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> dict | tuple[Tensor, Tensor]:
        """
        Args:
            x: (N, T, C) 特征张量 (如 log-Mel / FBANK)。
            x_lens: (N,) 每个样本的有效帧数。
            return_dict: True 返回 dict，False 返回 tuple。
            decoding_chunk_size: 解码 chunk 大小。
                0: 训练默认，使用随机动态 chunk。
                <0: 解码时使用全 chunk。
                >0: 解码时使用固定 chunk。
            num_decoding_left_chunks: 左 chunk 数量。
                >=0: 使用指定数量。
                <0: 使用全部左 chunk。

        Returns:
            dict: {'encoder_out': Tensor, 'encoder_out_lens': Tensor}
            或 tuple: (encoder_out, encoder_out_lens)
        """
        encoder_out, chunk_masks = self.encoder(
            xs=x,
            xs_lens=x_lens,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
        )

        # 从 chunk_masks 计算输出长度
        if chunk_masks.dim() == 3 and chunk_masks.size(1) == 1:
            encoder_out_lens = chunk_masks.squeeze(1).sum(dim=1).long()
        elif chunk_masks.dim() == 2:
            encoder_out_lens = chunk_masks.sum(dim=1).long()
        else:
            encoder_out_lens = chunk_masks[:, -1, :].sum(dim=1).long()

        if return_dict:
            return {
                "encoder_out": encoder_out,
                "encoder_out_lens": encoder_out_lens,
            }
        else:
            return encoder_out, encoder_out_lens

    def save_pretrained(
        self,
        save_directory: str,
        *,
        filename: str | None = None,
        use_safetensors: bool = True,
    ) -> str:
        """保存模型权重和配置到目录。

        Args:
            save_directory: 目标目录。
            filename: 权重文件名。None 时自动选择 model.safetensors 或 model.pt。
            use_safetensors: 优先使用 safetensors 格式。

        Returns:
            保存的权重文件路径。
        """
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        chosen_filename = filename
        if chosen_filename is None:
            if use_safetensors:
                try:
                    from safetensors.torch import save_file as safe_save_file  # noqa: F401
                    chosen_filename = "model.safetensors"
                except Exception:
                    chosen_filename = "model.pt"
            else:
                chosen_filename = "model.pt"

        weight_path = os.path.join(save_directory, chosen_filename)
        state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()}

        ext = os.path.splitext(weight_path)[1].lower()
        if ext == ".safetensors":
            try:
                from safetensors.torch import save_file as safe_save_file
                safe_save_file(state_dict, weight_path)
            except Exception as e:
                logging.warning(f"safetensors 保存失败 ({e}); 回退到 .pt")
                weight_path = os.path.splitext(weight_path)[0] + ".pt"
                torch.save(state_dict, weight_path)
        else:
            torch.save(state_dict, weight_path)

        logging.info(f"模型已保存到 {weight_path}")
        return weight_path

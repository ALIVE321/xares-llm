"""
独立的 WeNet Transformer/Conformer encoder 配置类。

不依赖 auden 框架，仅依赖标准库 json。
兼容原始 auden WenetTransformerConfig 的 config.json 格式。

使用方法:
    config = WenetTransformerConfig.from_pretrained("/path/to/model_dir")
    config.save_pretrained("/path/to/output_dir")
"""

import json
import logging
import os
from typing import Optional


class WenetTransformerConfig:
    """WeNet Transformer/Conformer encoder 配置。

    支持两种 encoder 类型:
    - "transformer": 标准 Transformer encoder
    - "conformer": Conformer encoder (Transformer + 卷积模块)
    """

    model_type: str = "wenet-transformer"

    def __init__(
        self,
        encoder_type: str = "conformer",
        input_size: int = 80,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: Optional[str] = None,
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn=None,
        use_dynamic_left_chunk: bool = False,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        # Conformer 特有参数
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        # Attention bias 参数
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        # 高级参数
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = "position_wise_feed_forward",
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
        **kwargs,
    ):
        assert encoder_type in ("conformer", "transformer"), (
            f"encoder_type 必须是 'conformer' 或 'transformer', 得到 '{encoder_type}'"
        )

        self.model_type = "wenet-transformer"
        self.encoder_type = encoder_type
        self.input_size = input_size
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.positional_dropout_rate = positional_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.input_layer = input_layer

        # 根据 encoder_type 设定默认值
        self.pos_enc_layer_type = pos_enc_layer_type or (
            "rel_pos" if encoder_type == "conformer" else "abs_pos"
        )
        self.selfattention_layer_type = selfattention_layer_type or (
            "rel_selfattn" if encoder_type == "conformer" else "selfattn"
        )
        self.activation_type = activation_type or (
            "swish" if encoder_type == "conformer" else "relu"
        )

        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.global_cmvn = global_cmvn
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa
        self.layer_norm_type = layer_norm_type
        self.norm_eps = norm_eps

        # Conformer 特有
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.macaron_style = macaron_style
        self.use_cnn_module = use_cnn_module
        self.cnn_module_kernel = cnn_module_kernel
        self.causal = causal
        self.cnn_module_norm = cnn_module_norm

        # Attention bias
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.conv_bias = conv_bias

        # 高级参数
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.mlp_type = mlp_type
        self.mlp_bias = mlp_bias
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated

        # 保留未知参数以保持前向兼容
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """返回可 JSON 序列化的字典表示。"""
        output = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            if hasattr(value, "to_dict"):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output

    def save_pretrained(self, output_dir: str, filename: str = "config.json"):
        """保存配置到 JSON 文件。"""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, filename)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logging.info(f"配置已保存到: {config_path}")

    @classmethod
    def from_pretrained(cls, config_path: str, filename: str = "config.json"):
        """从目录或 JSON 文件加载配置。"""
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, filename)
        else:
            config_file = config_path
        with open(config_file, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({json.dumps(self.to_dict(), indent=2)})"

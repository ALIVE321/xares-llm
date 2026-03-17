"""WeNet Transformer/Conformer 独立 encoder 模型包。

使用方法 (两种方式均可):

1. 直接 sys.path 导入:
    import sys
    sys.path.insert(0, "/path/to/wenet_ssl_hubert_v2.0.1_wchunk")
    from model import WenetTransformerEncoderModel
    model = WenetTransformerEncoderModel.from_pretrained("/path/to/wenet_ssl_hubert_v2.0.1_wchunk")

2. 作为包导入 (需要将父目录加入 sys.path):
    from wenet_ssl_hubert_v2.0.1_wchunk.model import WenetTransformerEncoderModel

依赖:
    - torch
    - wenet (PYTHONPATH 需包含 wenet 源码根目录, 如 /path/to/asr3/wenet)
    - safetensors (可选, 用于 .safetensors 权重格式)
"""

from .model_config import WenetTransformerConfig
from .model import WenetTransformerEncoderModel

__all__ = ["WenetTransformerConfig", "WenetTransformerEncoderModel"]

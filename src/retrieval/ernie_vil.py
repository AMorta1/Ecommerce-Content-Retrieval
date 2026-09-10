"""ERNIE-ViL 2.0 中文图文向量提取。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .windows_cuda import configure_windows_cuda_dll_paths


DEFAULT_MODEL_NAME = "PaddlePaddle/ernie_vil-2.0-base-zh"


class ErnieVilEmbedder:
    """把中文文本和商品图片编码到同一个 768 维向量空间。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "gpu:0",
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size 必须大于或等于 1。")
        configure_windows_cuda_dll_paths()

        # 必须先补全 Windows DLL 路径，再导入 Paddle。
        import paddle
        from paddlenlp import Taskflow

        if device.startswith("gpu") and not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("当前 Paddle 不是 GPU 版本，无法使用 GPU 推理。")

        paddle.set_device(device)
        self._paddle = paddle
        self._encoder: Any = Taskflow(
            "feature_extraction",
            model=model_name,
            batch_size=batch_size,
            is_static_model=False,
        )
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """编码一组非空中文查询，并返回已归一化的二维数组。"""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("texts 必须包含至少一条非空文本。")
        features = self._encoder(list(texts))["features"].numpy()
        return normalize_rows(features)

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        """编码一组本地图片，并返回已归一化的二维数组。"""
        if not image_paths:
            raise ValueError("image_paths 不能为空。")

        images: list[Image.Image] = []
        try:
            for path in image_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            features = self._encoder(images)["features"].numpy()
        finally:
            for image in images:
                image.close()
        return normalize_rows(features)

    def gpu_memory_allocated_mib(self) -> float | None:
        """返回当前已分配显存；CPU 模式或接口不可用时返回 None。"""
        if not self.device.startswith("gpu"):
            return None
        memory_bytes = self._paddle.device.cuda.memory_allocated()
        return round(memory_bytes / 1024**2, 2)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    """对每个向量做 L2 归一化，便于用点积计算余弦相似度。"""
    if features.ndim != 2:
        raise ValueError(f"features 必须是二维数组，实际维度为 {features.ndim}。")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("模型返回了零向量，无法归一化。")
    return features / norms

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinerUResult:
    markdown: str
    raw_blocks: list[Any]
    elapsed_seconds: float


class MinerUBackend(Protocol):
    def extract_image(self, image_path: Path) -> MinerUResult:
        ...


class DisabledMinerUBackend:
    def extract_image(self, image_path: Path) -> MinerUResult:
        return MinerUResult(markdown="", raw_blocks=[], elapsed_seconds=0.0)


class TransformersMinerUBackend:
    def __init__(
        self,
        model_id: str,
        dtype: str = "float32",
        device_map: str = "cpu",
        image_analysis: bool = False,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.image_analysis = image_analysis
        self._client: Any | None = None
        self._json2md: Any | None = None

    def _load(self) -> None:
        if self._client is not None:
            return

        try:
            import torch
            from mineru_vl_utils import MinerUClient
            from mineru_vl_utils.post_process import json2md
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "MinerU transformers backend dependencies are not installed. "
                "Install them with 'pip install -r requirements.txt'. "
                "For Windows CPU, install PyTorch CPU wheels first as shown in README.md."
            ) from exc

        dtype_value: Any
        if self.dtype == "float32":
            dtype_value = torch.float32
        elif self.dtype == "float16":
            dtype_value = torch.float16
        elif self.dtype == "bfloat16":
            dtype_value = torch.bfloat16
        else:
            dtype_value = "auto"

        if self.device_map == "auto":
            if torch.cuda.is_available():
                log.info("CUDA detected for MinerU: %s", torch.cuda.get_device_name(0))
            else:
                log.warning("MinerU device_map=auto but CUDA is not available; transformers may fall back to CPU.")

        log.info(
            "Loading MinerU model: %s (device_map=%s, dtype=%s)",
            self.model_id,
            self.device_map,
            self.dtype,
        )
        model_kwargs = {
            "device_map": {"": "cpu"} if self.device_map == "cpu" else self.device_map,
            "dtype": dtype_value,
        }
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except TypeError:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        model.eval()

        processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self._client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
            image_analysis=self.image_analysis,
        )
        self._json2md = json2md

    def extract_image(self, image_path: Path) -> MinerUResult:
        self._load()
        assert self._client is not None
        assert self._json2md is not None

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for MinerU image input.") from exc

        started = time.perf_counter()
        with Image.open(image_path) as image:
            blocks = self._client.two_step_extract(image.convert("RGB"))
        markdown = str(self._json2md(blocks)).strip()
        elapsed = time.perf_counter() - started
        return MinerUResult(markdown=markdown, raw_blocks=blocks, elapsed_seconds=elapsed)


def create_mineru_backend(
    backend: str,
    model_id: str,
    dtype: str,
    device_map: str,
    image_analysis: bool,
) -> MinerUBackend:
    if backend == "disabled":
        return DisabledMinerUBackend()
    if backend == "transformers":
        return TransformersMinerUBackend(
            model_id=model_id,
            dtype=dtype,
            device_map=device_map,
            image_analysis=image_analysis,
        )
    raise ValueError(f"Unsupported MinerU backend: {backend}")


def serialize_mineru_blocks(blocks: list[Any]) -> list[Any]:
    serialized = []
    for block in blocks:
        if hasattr(block, "model_dump"):
            serialized.append(block.model_dump())
        elif hasattr(block, "__dict__"):
            serialized.append(dict(block.__dict__))
        else:
            serialized.append(block)
    return serialized


def save_mineru_result(result: MinerUResult, md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(result.markdown + ("\n" if result.markdown else ""), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "elapsed_seconds": result.elapsed_seconds,
                "blocks": serialize_mineru_blocks(result.raw_blocks),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

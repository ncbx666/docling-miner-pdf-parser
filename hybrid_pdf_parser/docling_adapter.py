from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


def create_docling_converter(image_scale: float, cpu_threads: int, device: str) -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorOptions,
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = image_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    if hasattr(pipeline_options, "generate_table_images"):
        pipeline_options.generate_table_images = True
    pipeline_options.do_table_structure = True
    pipeline_options.enable_remote_services = False
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=device,
        num_threads=cpu_threads,
    )
    if hasattr(pipeline_options, "table_structure_options"):
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )


def save_docling_artifacts(document: Any, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        document.save_as_json(artifacts_dir / "docling_document.json")
    except Exception as exc:
        log.warning("Could not save Docling JSON artifact: %s", exc)

    try:
        document.save_as_markdown(artifacts_dir / "docling_document.md")
    except Exception as exc:
        log.warning("Could not save Docling Markdown artifact: %s", exc)


def save_docling_page_images(document: Any, pages_dir: Path) -> dict[int, Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for page_no, page in sorted(document.pages.items()):
        image_ref = getattr(page, "image", None)
        pil_image = getattr(image_ref, "pil_image", None) if image_ref is not None else None
        if pil_image is None:
            continue
        path = pages_dir / f"page_{page_no:04d}.png"
        pil_image.save(path, format="PNG")
        paths[page_no] = path
    return paths

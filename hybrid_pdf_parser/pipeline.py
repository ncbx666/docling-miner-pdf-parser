from __future__ import annotations

import logging
import re
from hashlib import sha1
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .docling_adapter import (
    create_docling_converter,
    save_docling_artifacts,
    save_docling_page_images,
)
from .markdown import (
    fallback_complex_markdown,
    item_bbox,
    item_page_no,
    join_blocks,
    label_value,
    render_text_item,
)
from .mineru_backend import create_mineru_backend, save_mineru_result
from .pdf_inspection import inspect_pdf_pages, render_page


log = logging.getLogger(__name__)


COMPLEX_LABELS = {"table", "picture", "chart", "formula"}


@dataclass(frozen=True)
class HybridParserConfig:
    artifacts_dir: Path
    mineru_backend: str = "transformers"
    mineru_model: str = "opendatalab/MinerU2.5-Pro-2604-1.2B"
    mineru_dtype: str = "float32"
    mineru_device_map: str = "cpu"
    mineru_image_analysis: bool = True
    low_text_threshold: int = 80
    image_scale: float = 2.0
    docling_threads: int = 4
    docling_device: str = "cpu"
    runtime_profile: str = "local-cpu"


@dataclass(frozen=True)
class ComplexItemResult:
    markdown: str
    image_path: Path | None
    used_mineru: bool


class HybridPdfParser:
    def __init__(self, config: HybridParserConfig) -> None:
        self.config = config
        self._docling_converter: Any | None = None
        self._mineru_backend = create_mineru_backend(
            backend=config.mineru_backend,
            model_id=config.mineru_model,
            dtype=config.mineru_dtype,
            device_map=config.mineru_device_map,
            image_analysis=config.mineru_image_analysis,
        )

    def parse_pdf(self, pdf_path: Path, output_path: Path) -> None:
        started = time.perf_counter()
        pdf_path = pdf_path.resolve()
        artifact_dir = self.config.artifacts_dir / self._artifact_name(pdf_path)
        pages_dir = artifact_dir / "pages"
        regions_dir = artifact_dir / "regions"
        mineru_dir = artifact_dir / "mineru"

        log.info("Inspecting pages for scanned/low-text detection")
        page_stats = inspect_pdf_pages(pdf_path)
        low_text_pages = {
            page_no
            for page_no, stats in page_stats.items()
            if stats.text_chars < self.config.low_text_threshold
        }
        if low_text_pages:
            log.info("Low-text pages routed to MinerU full-page parsing: %s", sorted(low_text_pages))

        converter = self._get_docling_converter()
        docling_started = time.perf_counter()
        conversion = converter.convert(pdf_path)
        document = conversion.document
        log.info("Docling conversion finished in %.2f seconds", time.perf_counter() - docling_started)

        save_docling_artifacts(document, artifact_dir)
        page_images = save_docling_page_images(document, pages_dir)

        markdown_blocks: list[str] = [
            f"<!-- source: {pdf_path.name} -->",
            "<!-- parser: Docling main structure + MinerU complex visual regions -->",
        ]

        for page_no in sorted(document.pages):
            page_md = self._render_page(
                pdf_path=pdf_path,
                document=document,
                page_no=page_no,
                page_images=page_images,
                page_stats=page_stats,
                low_text_pages=low_text_pages,
                regions_dir=regions_dir,
                mineru_dir=mineru_dir,
                markdown_base_dir=output_path.parent,
            )
            if page_md.strip():
                markdown_blocks.append(f"<!-- page {page_no} -->\n\n{page_md.strip()}")

        output_path.write_text(join_blocks(markdown_blocks), encoding="utf-8")
        elapsed = time.perf_counter() - started
        log.info("Wrote %s", output_path)
        log.info("PDF pipeline finished in %.2f seconds", elapsed)

    def _get_docling_converter(self) -> Any:
        if self._docling_converter is None:
            self._docling_converter = create_docling_converter(
                image_scale=self.config.image_scale,
                cpu_threads=self.config.docling_threads,
                device=self.config.docling_device,
            )
        return self._docling_converter

    def _render_page(
        self,
        pdf_path: Path,
        document: Any,
        page_no: int,
        page_images: dict[int, Path],
        page_stats: dict[int, Any],
        low_text_pages: set[int],
        regions_dir: Path,
        mineru_dir: Path,
        markdown_base_dir: Path,
    ) -> str:
        if page_no in low_text_pages:
            full_page = self._full_page_mineru(
                pdf_path=pdf_path,
                page_no=page_no,
                page_images=page_images,
                mineru_dir=mineru_dir,
            )
            if full_page:
                return full_page

        blocks: list[str] = []
        for item, level in document.iterate_items(page_no=page_no):
            label = label_value(item)
            if label in COMPLEX_LABELS:
                complex_result = self._complex_item_mineru(
                    item=item,
                    document=document,
                    page_no=page_no,
                    regions_dir=regions_dir,
                    mineru_dir=mineru_dir,
                    markdown_base_dir=markdown_base_dir,
                )
                blocks.append(complex_result.markdown)
            else:
                blocks.append(render_text_item(item, level))

        if not blocks:
            stats = page_stats.get(page_no)
            detail = f"text_chars={stats.text_chars}" if stats else "no Docling page items"
            blocks.append(f"<!-- empty page after Docling iteration: {detail} -->")
        return join_blocks(blocks)

    def _full_page_mineru(
        self,
        pdf_path: Path,
        page_no: int,
        page_images: dict[int, Path],
        mineru_dir: Path,
    ) -> str:
        page_image = page_images.get(page_no)
        if page_image is None:
            page_image = render_page(
                pdf_path=pdf_path,
                page_no=page_no,
                output_path=mineru_dir / f"page_{page_no:04d}_rendered.png",
                scale=self.config.image_scale,
            )
        if self.config.mineru_backend == "disabled":
            return ""

        try:
            result = self._mineru_backend.extract_image(page_image)
        except Exception as exc:
            log.warning("MinerU failed on page %s; falling back to Docling items: %s", page_no, exc)
            return ""

        md_path = mineru_dir / f"page_{page_no:04d}_full.md"
        json_path = mineru_dir / f"page_{page_no:04d}_full.json"
        save_mineru_result(result, md_path, json_path)
        if not result.markdown:
            return ""
        log.info("MinerU full-page parse page %d finished in %.2f seconds", page_no, result.elapsed_seconds)
        return f"<!-- MinerU full-page result for low-text page {page_no} -->\n\n{result.markdown}"

    def _complex_item_mineru(
        self,
        item: Any,
        document: Any,
        page_no: int,
        regions_dir: Path,
        mineru_dir: Path,
        markdown_base_dir: Path,
    ) -> ComplexItemResult:
        label = label_value(item)
        item_ref = self._safe_filename(
            str(getattr(item, "self_ref", "")).strip("#/").replace("/", "_") or "item"
        )
        actual_page = item_page_no(item) or page_no
        image_path = self._save_item_image(
            item=item,
            document=document,
            page_no=actual_page,
            label=label,
            item_ref=item_ref,
            regions_dir=regions_dir,
        )

        markdown_image_path = (
            self._markdown_path(image_path, markdown_base_dir) if image_path is not None else None
        )
        fallback = fallback_complex_markdown(item, document, markdown_image_path)
        if self.config.mineru_backend == "disabled" or image_path is None:
            return ComplexItemResult(markdown=fallback, image_path=image_path, used_mineru=False)

        try:
            result = self._mineru_backend.extract_image(image_path)
        except Exception as exc:
            log.warning("MinerU failed on %s %s page %s: %s", label, item_ref, actual_page, exc)
            return ComplexItemResult(markdown=fallback, image_path=image_path, used_mineru=False)

        base_name = f"page_{actual_page:04d}_{label}_{item_ref}"
        save_mineru_result(
            result,
            mineru_dir / f"{base_name}.md",
            mineru_dir / f"{base_name}.json",
        )
        if not result.markdown:
            return ComplexItemResult(markdown=fallback, image_path=image_path, used_mineru=False)

        bbox = item_bbox(item)
        bbox_note = f" bbox={bbox}" if bbox else ""
        log.info(
            "MinerU parsed %s on page %d in %.2f seconds",
            label,
            actual_page,
            result.elapsed_seconds,
        )
        markdown = (
            f"<!-- MinerU enhancement: {label} page={actual_page}{bbox_note} -->\n\n"
            f"{result.markdown}"
        )
        return ComplexItemResult(markdown=markdown, image_path=image_path, used_mineru=True)

    def _save_item_image(
        self,
        item: Any,
        document: Any,
        page_no: int,
        label: str,
        item_ref: str,
        regions_dir: Path,
    ) -> Path | None:
        if not hasattr(item, "get_image"):
            return None
        try:
            image = item.get_image(document)
        except Exception as exc:
            log.debug("Docling could not render item image for %s: %s", item_ref, exc)
            return None
        if image is None:
            return None
        regions_dir.mkdir(parents=True, exist_ok=True)
        image_path = regions_dir / f"page_{page_no:04d}_{label}_{item_ref}.png"
        image.save(image_path, format="PNG")
        return image_path

    @staticmethod
    def _artifact_name(pdf_path: Path) -> str:
        digest = sha1(str(pdf_path).encode("utf-8")).hexdigest()[:10]
        return f"{HybridPdfParser._safe_filename(pdf_path.stem)}_{digest}"

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value)
        cleaned = "".join("_" if ord(char) < 32 else char for char in cleaned).strip(" ._")
        return cleaned[:120] or "item"

    @staticmethod
    def _markdown_path(path: Path, base_dir: Path) -> Path:
        try:
            return path.resolve().relative_to(base_dir.resolve())
        except ValueError:
            return path

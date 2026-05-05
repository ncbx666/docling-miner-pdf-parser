from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageStats:
    page_no: int
    text_chars: int
    image_count: int

    @property
    def is_low_text(self) -> bool:
        return self.text_chars == 0


def inspect_pdf_pages(pdf_path: Path) -> dict[int, PageStats]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for page inspection. Install requirements.txt.") from exc

    stats: dict[int, PageStats] = {}
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            images = page.get_images(full=True)
            stats[index] = PageStats(
                page_no=index,
                text_chars=len(text.strip()),
                image_count=len(images),
            )
    return stats


def render_page(pdf_path: Path, page_no: int, output_path: Path, scale: float) -> Path:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for fallback page rendering.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        page = doc[page_no - 1]
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(output_path)
    return output_path

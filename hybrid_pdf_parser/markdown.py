from __future__ import annotations

import html
from pathlib import Path
from typing import Any


TEXT_LABELS = {
    "caption",
    "field_value",
    "footnote",
    "handwritten_text",
    "page_footer",
    "page_header",
    "paragraph",
    "reference",
    "text",
}


def label_value(item: Any) -> str:
    label = getattr(item, "label", "")
    return getattr(label, "value", str(label)).lower()


def item_page_no(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    return getattr(prov[0], "page_no", None)


def item_bbox(item: Any) -> list[float] | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return None
    values = []
    for attr in ("l", "t", "r", "b"):
        if hasattr(bbox, attr):
            values.append(float(getattr(bbox, attr)))
    return values or None


def render_text_item(item: Any, level: int) -> str:
    label = label_value(item)
    text = str(getattr(item, "text", "") or "").strip()
    if not text:
        return ""

    escaped = html.escape(text, quote=False)
    if label == "title":
        return f"# {escaped}"
    if label == "section_header":
        heading_level = max(2, min(6, level + 1))
        return f"{'#' * heading_level} {escaped}"
    if label == "list_item":
        return f"- {escaped}"
    if label == "code":
        return f"```\n{text}\n```"
    if label == "formula":
        return f"$$\n{text}\n$$"
    if label in {"caption", "page_footer", "page_header", "footnote"}:
        return f"*{escaped}*"
    return escaped


def fallback_complex_markdown(item: Any, document: Any, image_path: Path | None) -> str:
    label = label_value(item)
    if label == "table" and hasattr(item, "export_to_markdown"):
        try:
            return str(item.export_to_markdown(doc=document)).strip()
        except Exception:
            pass

    if label == "formula":
        return render_text_item(item, 0)

    caption = ""
    if hasattr(item, "caption_text"):
        try:
            caption = str(item.caption_text(document)).strip()
        except Exception:
            caption = ""

    if image_path is not None:
        alt = caption or label or "visual element"
        return f"![{alt}]({image_path.as_posix()})"
    if caption:
        return f"*{html.escape(caption, quote=False)}*"
    return f"<!-- {label or 'visual element'} -->"


def join_blocks(blocks: list[str]) -> str:
    cleaned = [block.strip() for block in blocks if block and block.strip()]
    return "\n\n".join(cleaned).strip() + ("\n" if cleaned else "")

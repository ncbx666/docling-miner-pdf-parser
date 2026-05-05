from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from hybrid_pdf_parser.pipeline import HybridParserConfig, HybridPdfParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid PDF parser: Docling for main structure, MinerU for complex visual regions.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input PDF file or a folder containing PDF files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Markdown file for one PDF, or output folder for an input folder.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Folder for debug artifacts. Defaults to '<output>_artifacts' for a file or '<output>/artifacts' for a folder.",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Use Google Colab/T4-friendly CUDA autodetection. Does not install dependencies.",
    )
    parser.add_argument(
        "--mineru-backend",
        choices=("transformers", "disabled"),
        default="transformers",
        help="MinerU backend. Use 'disabled' for Docling-only debugging.",
    )
    parser.add_argument(
        "--mineru-model",
        default="opendatalab/MinerU2.5-Pro-2604-1.2B",
        help="Hugging Face model id for the MinerU transformers backend.",
    )
    parser.add_argument(
        "--mineru-dtype",
        choices=("float32", "float16", "bfloat16", "auto"),
        default=None,
        help="Torch dtype for MinerU inference. Defaults to float32 locally and auto with --colab.",
    )
    parser.add_argument(
        "--disable-mineru-image-analysis",
        action="store_true",
        help="Disable MinerU image/chart analysis mode for faster text-only visual-region parsing.",
    )
    parser.add_argument(
        "--low-text-threshold",
        type=int,
        default=80,
        help="A page with fewer extracted text characters is sent to MinerU as a scanned/low-text page.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=2.0,
        help="Docling/PyMuPDF page image scale. 2.0 is roughly 144 DPI.",
    )
    parser.add_argument(
        "--docling-threads",
        type=int,
        default=4,
        help="Number of Docling CPU inference threads.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def collect_inputs(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file must be a PDF: {input_path}")
        return [input_path]

    pdfs = sorted(p for p in input_path.rglob("*.pdf") if p.is_file())
    if not pdfs:
        raise ValueError(f"No PDF files found under folder: {input_path}")
    return pdfs


def resolve_outputs(input_path: Path, output_path: Path, pdfs: list[Path]) -> dict[Path, Path]:
    if len(pdfs) == 1 and input_path.is_file():
        if output_path.exists() and output_path.is_dir():
            return {pdfs[0]: output_path / f"{pdfs[0].stem}.md"}
        return {pdfs[0]: output_path}

    output_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[Path, Path] = {}
    for pdf in pdfs:
        relative = pdf.relative_to(input_path) if input_path.is_dir() else Path(pdf.name)
        out_file = (output_path / relative).with_suffix(".md")
        outputs[pdf] = out_file
    return outputs


def default_artifacts_dir(output_path: Path, input_is_dir: bool) -> Path:
    if input_is_dir:
        return output_path / "artifacts"
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_artifacts")
    return output_path / "artifacts"


def build_config(args: argparse.Namespace, artifacts_dir: Path) -> HybridParserConfig:
    mineru_dtype = args.mineru_dtype
    if mineru_dtype is None:
        mineru_dtype = "auto" if args.colab else "float32"

    return HybridParserConfig(
        artifacts_dir=artifacts_dir,
        mineru_backend=args.mineru_backend,
        mineru_model=args.mineru_model,
        mineru_dtype=mineru_dtype,
        mineru_device_map="auto" if args.colab else "cpu",
        mineru_image_analysis=not args.disable_mineru_image_analysis,
        low_text_threshold=args.low_text_threshold,
        image_scale=args.image_scale,
        docling_threads=args.docling_threads,
        docling_device="auto" if args.colab else "cpu",
        runtime_profile="colab" if args.colab else "local-cpu",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("parse_pdf")

    total_start = time.perf_counter()
    try:
        pdfs = collect_inputs(args.input)
        outputs = resolve_outputs(args.input, args.output, pdfs)
        artifacts_dir = args.artifacts_dir or default_artifacts_dir(args.output, args.input.is_dir())

        config = build_config(args, artifacts_dir)
        log.info(
            "Runtime profile: %s; Docling device=%s; MinerU device_map=%s; MinerU dtype=%s",
            config.runtime_profile,
            config.docling_device,
            config.mineru_device_map,
            config.mineru_dtype,
        )
        parser = HybridPdfParser(config)

        log.info("Found %d PDF(s)", len(pdfs))
        for index, pdf in enumerate(pdfs, start=1):
            out_file = outputs[pdf]
            out_file.parent.mkdir(parents=True, exist_ok=True)
            log.info("Parsing %d/%d: %s", index, len(pdfs), pdf)
            parser.parse_pdf(pdf, out_file)

        elapsed = time.perf_counter() - total_start
        log.info("Whole pipeline finished in %.2f seconds", elapsed)
        return 0
    except Exception as exc:
        log.error("%s", exc)
        if args.log_level == "DEBUG":
            log.exception("Full traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())

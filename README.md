# Hybrid PDF Parser: Docling + MinerU 2604

This project parses PDFs locally on Windows 11 with a Docling-first pipeline:

- Docling handles the main PDF conversion, document structure, layout, tables, figures, and page images.
- MinerU 2604 is called only for complex visual regions detected by Docling (`table`, `picture`, `chart`, `formula`) and for scanned/low-text pages detected with PyMuPDF.
- Intermediate Docling, image, and MinerU artifacts are saved for debugging.
- The whole pipeline is timed in the logs.

Target MinerU model: `opendatalab/MinerU2.5-Pro-2604-1.2B`.

The default profile is Windows/local CPU. Use `--colab` in Google Colab with a T4 GPU.

## Setup in PowerShell

Use Python 3.10 or 3.11 on Windows 11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The first run downloads Docling and MinerU model artifacts. MinerU CPU inference for a 1.2B VLM is slow and needs several GB of RAM.

## Setup in Google Colab T4

In Colab, select `Runtime > Change runtime type > T4 GPU`, then run:

```python
!git clone <your-repository-url>
%cd <repository-folder>
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

Upload PDFs into the current Colab filesystem, for example `/content/input.pdf` or `/content/input_folder`. The `--colab` flag does not install anything and does not assume Google Drive mounts.

Run one PDF:

```python
!python parse_pdf.py /content/input.pdf --output /content/result.md --colab
```

Run a folder:

```python
!python parse_pdf.py /content/input_folder --output /content/result_folder --colab
```

`--colab` keeps accuracy-oriented settings such as Docling accurate table mode and the default `--image-scale 2.0`, but changes device handling:

- Docling uses `device="auto"` so it can use CUDA when available.
- MinerU transformers uses `device_map="auto"`.
- MinerU dtype defaults to `auto`, matching the official transformers example for the model.

If Colab reports CUDA out-of-memory on a large or image-heavy PDF, retry one PDF at a time and lower image scale only as a fallback:

```python
!python parse_pdf.py /content/input.pdf --output /content/result.md --colab --image-scale 1.5
```

## Parse One PDF

```powershell
python parse_pdf.py .\input.pdf --output .\result.md
```

## Parse A Folder

```powershell
python parse_pdf.py .\input_folder --output .\result_folder
```

Each input PDF produces one Markdown file under the output folder.

## Useful Options

```powershell
python parse_pdf.py .\input.pdf --output .\result.md --log-level DEBUG
python parse_pdf.py .\input.pdf --output .\result.md --mineru-backend disabled
python parse_pdf.py .\input.pdf --output .\result.md --low-text-threshold 120
python parse_pdf.py .\input.pdf --output .\result.md --mineru-dtype float32
python parse_pdf.py .\input.pdf --output .\result.md --disable-mineru-image-analysis
python parse_pdf.py input.pdf --output result.md --colab
```

`--mineru-backend disabled` is for debugging the Docling path only.

## Artifacts

For `--output result.md`, artifacts default to `result_artifacts/<pdf-stem>_<path-hash>/`:

- `docling_document.json`
- `docling_document.md`
- `pages/page_0001.png`
- `regions/page_0001_table_...png`
- `mineru/page_0001_table_...json`
- `mineru/page_0001_table_...md`

## Architecture

`parse_pdf.py` is the CLI entry point.

The package modules are:

- `hybrid_pdf_parser.docling_adapter`: documented Docling conversion setup and artifact export.
- `hybrid_pdf_parser.pdf_inspection`: PyMuPDF page text inspection and fallback rendering.
- `hybrid_pdf_parser.mineru_backend`: replaceable MinerU backend. The default implementation uses `mineru-vl-utils` with the transformers backend on CPU, or CUDA autodetection with `--colab`.
- `hybrid_pdf_parser.pipeline`: orchestration, routing, merge, logging, and timing.
- `hybrid_pdf_parser.markdown`: Markdown rendering helpers.

The MinerU backend is intentionally isolated so another local backend can replace it without changing the Docling pipeline.

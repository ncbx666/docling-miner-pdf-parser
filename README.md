# Гибридный PDF-парсер: Docling + MinerU

Локальный парсер PDF в Markdown. Основную работу делает Docling, а MinerU
подключается только там, где Docling не справляется.

- **Docling** конвертирует документ целиком: структура, разметка страниц,
  таблицы, рисунки, изображения страниц.
- **MinerU** (`opendatalab/MinerU2.5-Pro-2604-1.2B`) получает только сложные
  визуальные области, найденные Docling (`table`, `picture`, `chart`,
  `formula`), и страницы-сканы с малым количеством текста (их находит PyMuPDF).
- Промежуточные результаты Docling, изображения и ответы MinerU сохраняются для
  отладки, время каждого этапа пишется в лог.

По умолчанию парсер настроен на локальный CPU под Windows. Для Google Colab с
GPU T4 есть флаг `--colab`.

Тот же подход, но с GLM-OCR вместо MinerU:
[docling-glm-ocr-parser](https://github.com/ncbx666/docling-glm-ocr-parser).

## Установка (Windows, PowerShell)

Нужен Python 3.10 или 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

При первом запуске скачиваются модели Docling и MinerU. MinerU — VLM на 1,2
млрд параметров, на CPU работает медленно и требует несколько ГБ RAM.

## Установка в Google Colab (T4)

Выберите `Runtime > Change runtime type > T4 GPU` и выполните:

```python
!git clone https://github.com/ncbx666/docling-miner-pdf-parser.git
%cd docling-miner-pdf-parser
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

Загрузите PDF в файловую систему Colab, например `/content/input.pdf` или
`/content/input_folder`. Флаг `--colab` ничего не устанавливает и не
подключает Google Drive.

```python
!python parse_pdf.py /content/input.pdf --output /content/result.md --colab
!python parse_pdf.py /content/input_folder --output /content/result_folder --colab
```

С `--colab` настройки точности остаются прежними (точный режим таблиц Docling,
`--image-scale 2.0`), меняется только работа с устройством:

- Docling использует `device="auto"` и подхватывает CUDA;
- MinerU запускается с `device_map="auto"`;
- тип данных MinerU по умолчанию `auto`, как в официальном примере модели.

Если на большом или насыщенном картинками PDF не хватает видеопамяти,
обрабатывайте файлы по одному и только в крайнем случае уменьшайте масштаб:

```python
!python parse_pdf.py /content/input.pdf --output /content/result.md --colab --image-scale 1.5
```

## Запуск

Один PDF:

```powershell
python parse_pdf.py .\input.pdf --output .\result.md
```

Папка с PDF (на каждый файл — свой Markdown в выходной папке):

```powershell
python parse_pdf.py .\input_folder --output .\result_folder
```

## Параметры

| Флаг | По умолчанию | Что делает |
|---|---|---|
| `--output` | — | Markdown-файл для одного PDF или папка для папки |
| `--artifacts-dir` | `<output>_artifacts`, для папки `<output>/artifacts` | папка для отладочных артефактов |
| `--colab` | выкл. | автоопределение CUDA для Colab/T4 |
| `--mineru-backend` | `transformers` | `disabled` — только Docling, для отладки |
| `--mineru-model` | `opendatalab/MinerU2.5-Pro-2604-1.2B` | id модели на Hugging Face |
| `--mineru-dtype` | `float32` локально, `auto` с `--colab` | `float32`, `float16`, `bfloat16`, `auto` |
| `--disable-mineru-image-analysis` | выкл. | не анализировать изображения и графики, быстрее |
| `--low-text-threshold` | `80` | страница с меньшим числом символов считается сканом и уходит в MinerU |
| `--image-scale` | `2.0` | масштаб рендеринга страниц (2.0 ≈ 144 DPI) |
| `--docling-threads` | `4` | число CPU-потоков Docling |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Артефакты

Для `--output result.md` артефакты по умолчанию лежат в
`result_artifacts/<имя-pdf>_<хеш-пути>/`:

- `docling_document.json`, `docling_document.md` — результат Docling;
- `pages/page_0001.png` — изображения страниц;
- `regions/page_0001_table_...png` — вырезанные области;
- `mineru/page_0001_table_...json`, `.md` — ответы MinerU.

## Устройство

`parse_pdf.py` — точка входа CLI. Модули пакета `hybrid_pdf_parser`:

- `docling_adapter` — настройка конвертации Docling и экспорт артефактов;
- `pdf_inspection` — проверка текста страниц через PyMuPDF и запасной рендеринг;
- `mineru_backend` — бэкенд MinerU на `mineru-vl-utils` с transformers;
  вынесен отдельно, чтобы его можно было заменить, не трогая пайплайн Docling;
- `pipeline` — маршрутизация, склейка результатов, логирование и тайминги;
- `markdown` — сборка итогового Markdown.

## Лицензия

[MIT](LICENSE)

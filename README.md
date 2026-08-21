# PDF Translator Studio - OpenRouter Multimodal Release

A layout-first PDF translation application for native PDFs, mixed PDFs, and scanned/image PDFs.

## What changed

- Replaced DeepInfra with OpenRouter using the OpenAI-compatible API at `https://openrouter.ai/api/v1`.
- The Streamlit UI loads the OpenRouter model catalog and marks image-capable models with a `🖼️` badge.
- Normal PDF text continues through deterministic PDF-coordinate extraction and local rendering.
- Embedded images are preserved at their original PDF size and position.
- Text inside embedded images is sent to the selected OpenRouter vision model as an image input. The model returns OCR text, translation, and image-pixel bounding boxes.
- Full-page scanned/image-dominant pages automatically use the OpenRouter vision path when image processing is enabled.
- The renderer maps image-pixel boxes back to PDF coordinates and fits translated text only inside those boxes.
- No page reflow is performed. Images are not resized or replaced.
- A post-render QA report is still generated for native text geometry.

OpenRouter supports image understanding through Chat Completions by sending a user message with both text and an `image_url`/base64 image part. Vision-capable models advertise image input in the model catalog. See the OpenRouter documentation for the current modality requirements. 

## Install

```powershell
pip install -r requirements.txt
```

Local OCR extras are still available as a fallback for non-image-dominant pages:

```powershell
pip install -r requirements-ocr.txt
```

## Configure

Copy `.env.example` to `.env` and set:

```text
OPENROUTER_API_KEY=your_key
OPENROUTER_HTTP_REFERER=http://localhost:8501
OPENROUTER_APP_NAME=PDF Translator Studio
MAX_IMAGE_EDGE=2400
```

Do not commit `.env` or publish your API key.

## Run the UI

```powershell
python -m streamlit run ui\app.py
```

Open `http://localhost:8501`.

## Recommended UI settings

Choose a `🖼️` vision-capable OpenRouter model when your PDF contains scanned pages or text embedded inside images. Keep `Use OpenRouter vision for image text` enabled for those files.

For normal text PDFs, the same model handles translation while PyMuPDF preserves the existing coordinate layout, tables, drawings, and images.

## Image handling details

The source image object is never resized or replaced. The vision copy may be resized only for API efficiency; returned OCR coordinates are mapped back to the original image pixel dimensions and then to the exact PDF image rectangle.

When translating image text, the application paints a local background patch over the original glyph area and inserts the translated text inside the same rectangle. This keeps the surrounding image content and page geometry intact. For photographs or highly textured image backgrounds, visual replacement of text can require a dedicated image-inpainting model; this release intentionally does not alter the underlying image object.

## CLI

```powershell
python -m src.cli translate input.pdf --model google/gemini-2.5-flash --source Japanese --target English
```

Disable image OCR/translation with:

```powershell
python -m src.cli translate input.pdf --model <model> --no-image-vision
```

## Output

The app creates:

- translated PDF
- `.report.json` with extraction mode, image text regions, font fitting, warnings, and QA

The PDF is generated with the original page dimensions preserved.

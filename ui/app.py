
from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.pipeline import translate_document_with_options
from src.excel_pipeline import translate_excel_workbook


st.set_page_config(
    page_title="Document Translator Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fa; }
    .hero { padding:22px 28px; border-radius:18px;
            background:linear-gradient(135deg,#111827 0%,#263449 100%);
            color:white; margin-bottom:22px; }
    .hero h1 { margin:0; font-size:34px; letter-spacing:-.8px; }
    .hero p { margin:8px 0 0; color:#cbd5e1; font-size:15px; }
    .section-title { font-size:21px; font-weight:700; margin:10px 0 12px; color:#111827; }
    .status-card { padding:14px 16px; border:1px solid #e5e7eb;
                   border-radius:14px; background:white; min-height:82px; }
    .status-label { color:#6b7280; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
    .status-value { color:#111827; font-size:16px; font-weight:650; margin-top:7px; }
    .hint { color:#6b7280; font-size:13px; margin-top:-4px; margin-bottom:12px; }
    div[data-testid="stFileUploader"] { background:white; border:1px solid #e5e7eb;
                                        border-radius:14px; padding:8px; }
    .stButton > button { min-height:48px; border-radius:12px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌐 Document Translator Studio</h1>
        <p>OpenRouter translation for PDF and Excel with native-content preservation and image-aware processing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    st.error("OPENROUTER_API_KEY is missing. Put your OpenRouter key in the project's .env file.")
    st.stop()


@st.cache_data(ttl=600)
def get_models(api_key: str):
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = client.models.list()
    rows = []
    for item in response.data:
        raw = item.model_dump() if hasattr(item, "model_dump") else vars(item)
        arch = raw.get("architecture") or {}
        inputs = set(arch.get("input_modalities") or [])
        outputs = set(arch.get("output_modalities") or [])
        model_id = raw.get("id") or getattr(item, "id", None)
        if not model_id:
            continue
        if "embedding" in model_id.lower() and "text" not in outputs:
            continue
        if (not inputs or "text" in inputs) and (not outputs or "text" in outputs):
            rows.append({
                "id": model_id,
                "vision": "image" in inputs,
            })
    rows.sort(key=lambda x: (not x["vision"], x["id"].lower()))
    return rows


@st.cache_data(ttl=600)
def get_image_models(api_key: str):
    response = requests.get(
        "https://openrouter.ai/api/v1/images/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    rows = []
    for item in response.json().get("data", []):
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        arch = item.get("architecture") or {}
        inputs = set(arch.get("input_modalities") or [])
        outputs = set(arch.get("output_modalities") or [])
        if model_id and "image" in inputs and "image" in outputs:
            rows.append({
                "id": model_id,
                "name": item.get("name") or model_id,
            })
    rows.sort(key=lambda x: (
        x["id"] != "google/gemini-3.1-flash-image-preview",
        x["id"].lower(),
    ))
    return rows


try:
    model_catalog = get_models(api_key)
    model_error = None
except Exception as exc:
    model_catalog = []
    model_error = str(exc)

try:
    image_catalog = get_image_models(api_key)
    image_error = None
except Exception as exc:
    image_catalog = []
    image_error = str(exc)


LANGUAGES = [
    "Japanese", "English", "Chinese", "Korean", "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian", "Arabic", "Hindi", "Tamil", "Telugu", "Malayalam", "Kannada",
    "Thai", "Vietnamese", "Dutch", "Polish", "Ukrainian",
]

st.markdown('<div class="section-title">⚙️ Translation setup</div>', unsafe_allow_html=True)

model_by_display = {}
for row in model_catalog:
    display = f"{row['id']}{' 🖼️' if row['vision'] else ''}"
    model_by_display[display] = row

model_options = ["Select an OpenRouter model..."] + list(model_by_display)
source_options = ["Select source language..."] + LANGUAGES

c1, c2, c3 = st.columns(3)
with c1:
    selected_model_display = st.selectbox("🤖 AI Model", model_options, index=0)
    selected_model = model_by_display.get(selected_model_display, {}).get("id")
with c2:
    selected_source = st.selectbox("🌏 Source Language", source_options, index=0)
    source_language = None if selected_source == source_options[0] else selected_source
with c3:
    target_options = ["Select target language..."] + [x for x in LANGUAGES if x != source_language]
    selected_target = st.selectbox("🌎 Target Language", target_options, index=0)
    target_language = None if selected_target == target_options[0] else selected_target

img_model_by_display = {
    f"{row['id']} 🖼️": row for row in image_catalog
}
img_options = list(img_model_by_display)
if not img_options:
    selected_image_model = None
else:
    default_img = next(
        (i for i, x in enumerate(img_options)
         if x.startswith("google/gemini-3.1-flash-image-preview")),
        0,
    )
    selected_image_model_display = st.selectbox(
        "🖼️ Excel image model",
        img_options,
        index=default_img,
        help="Used only for images embedded in Excel workbooks. The generated image is written back using the original Excel image anchor and displayed size.",
    )
    selected_image_model = img_model_by_display[selected_image_model_display]["id"]

st.markdown('<div class="section-title">📁 Document</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload PDF or Excel",
    type=["pdf", "xlsx", "xls"],
    help="PDF, .xlsx and legacy .xls are supported.",
)

file_type = Path(uploaded.name).suffix.lower() if uploaded else ""

if model_error:
    st.warning(f"OpenRouter model catalog error: {model_error}")
if image_error and file_type in {".xlsx", ".xls"}:
    st.warning(f"OpenRouter image model catalog error: {image_error}")

if file_type in {".xlsx", ".xls"}:
    st.markdown('<div class="section-title">📊 Excel processing</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        translate_cells = st.checkbox(
            "Translate native Excel cells",
            value=True,
            help="Reads actual cell values and translates only text cells. Formulas, formatting, merged cells, widths and heights are preserved.",
        )
    with e2:
        translate_excel_images = st.checkbox(
            "Translate embedded images",
            value=True,
            help="Each embedded Excel image is sent to the selected image model and replaced at the exact original Excel anchor and displayed size.",
        )
    with e3:
        preserve_formulas = st.checkbox(
            "Preserve formulas",
            value=True,
            disabled=True,
            help="Formulas are always preserved by this pipeline.",
        )
else:
    st.markdown('<div class="section-title">🖼️ PDF image / OCR behavior</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        use_vision_images = st.checkbox(
            "Use OpenRouter vision for image text",
            value=True,
            help="Keeps the existing PDF image/OCR workflow.",
        )
    with p2:
        max_image_edge = st.selectbox(
            "Vision image size",
            [1600, 2000, 2400, 3000],
            index=2,
        )

    st.markdown('<div class="section-title">🧠 PDF extraction</div>', unsafe_allow_html=True)
    p3, p4 = st.columns(2)
    with p3:
        extraction_mode = st.selectbox(
            "Text extraction mode",
            ["Auto (MuPDF → OCR when needed)", "Native MuPDF only", "OCR"],
            index=0,
        )
    with p4:
        ocr_dpi = st.selectbox("Local OCR quality", [200, 300, 400], index=1)

    st.markdown('<div class="section-title">📐 PDF layout</div>', unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    with l1:
        min_scale = st.slider("Minimum font scale", 0.55, 1.00, 0.60, 0.01)
    with l2:
        margin = st.slider("Text fitting margin", 0.0, 5.0, 0.0, 0.25)

if uploaded:
    st.success(f"Loaded: {uploaded.name}")

    if st.button("🚀 Translate", type="primary", use_container_width=True):
        missing = []
        if not selected_model:
            missing.append("AI model")
        if not source_language:
            missing.append("source language")
        if not target_language:
            missing.append("target language")
        if file_type in {".xlsx", ".xls"} and translate_excel_images and not selected_image_model:
            missing.append("Excel image model")

        if missing:
            st.warning("Please select " + ", ".join(missing) + " before translating.")
            st.stop()

        tmpdir = Path(tempfile.mkdtemp(prefix="document_translator_"))
        input_path = tmpdir / uploaded.name
        input_path.write_bytes(uploaded.getvalue())

        progress = st.progress(0)
        status = st.empty()

        try:
            if file_type in {".xlsx", ".xls"}:
                status.info("Analyzing workbook sheets, native cells, and embedded images...")
                progress.progress(15)

                output_path = tmpdir / f"{input_path.stem}_{target_language.lower().replace(' ', '_')}.xlsx"

                result = translate_excel_workbook(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    api_key=api_key,
                    model=selected_model,
                    image_model=selected_image_model,
                    source_language=source_language,
                    target_language=target_language,
                    translate_native_cells=translate_cells,
                    translate_images=translate_excel_images,
                )

                progress.progress(100)
                status.success("Excel translation completed.")

                st.download_button(
                    "⬇️ Download translated Excel",
                    data=Path(result["output"]).read_bytes(),
                    file_name=Path(result["output"]).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                status.info("Analyzing PDF structure, native text, and images...")
                progress.progress(15)

                output_path = tmpdir / f"{input_path.stem}_{target_language.lower().replace(' ', '_')}.pdf"

                result = translate_document_with_options(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    api_key=api_key,
                    model=selected_model,
                    source_language=source_language,
                    target_language=target_language,
                    min_font_scale=min_scale,
                    text_margin=margin,
                    use_cache=True,
                    extraction_mode=extraction_mode,
                    ocr_dpi=ocr_dpi,
                    use_vision_images=use_vision_images,
                    max_image_edge=max_image_edge,
                    # Keep your existing PDF image model argument for compatibility.
                    image_model=locals().get("selected_image_model"),
                )

                progress.progress(100)
                status.success("PDF translation completed.")

                st.download_button(
                    "⬇️ Download translated PDF",
                    data=output_path.read_bytes(),
                    file_name=output_path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )

            with st.expander("Translation report"):
                st.json(result)

        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Translation failed: {exc}")
            with st.expander("Technical details"):
                st.code(str(exc))
        finally:
            gc.collect()
            time.sleep(0.15)
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                st.caption(f"Temporary working files retained for debugging: {tmpdir}")

st.divider()
st.caption(
    "OpenRouter handles text translation and vision/image processing. "
    "Excel native cells keep workbook structure; embedded Excel images are "
    "translated by a Gemini image model and written back to the original image area."
)

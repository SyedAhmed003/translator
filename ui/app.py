import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.pipeline import translate_document_with_options  # noqa: E402

st.set_page_config(
    page_title="PDF Translator Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fa; }
    .hero { padding: 22px 28px; border-radius: 18px; background: linear-gradient(135deg,#111827 0%,#263449 100%); color:white; margin-bottom:22px; }
    .hero h1 { margin:0; font-size:34px; letter-spacing:-.8px; }
    .hero p { margin:8px 0 0; color:#cbd5e1; font-size:15px; }
    .section-title { font-size:21px; font-weight:700; margin:10px 0 12px; color:#111827; }
    .status-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:white; min-height:82px; }
    .status-label { color:#6b7280; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
    .status-value { color:#111827; font-size:16px; font-weight:650; margin-top:7px; }
    .hint { color:#6b7280; font-size:13px; margin-top:-4px; margin-bottom:12px; }
    div[data-testid="stFileUploader"] { background:white; border:1px solid #e5e7eb; border-radius:14px; padding:8px; }
    .stButton > button { min-height:48px; border-radius:12px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌐 PDF Translator Studio</h1>
        <p>OpenRouter text + vision translation with fixed PDF geometry and image-preserving layout.</p>
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
        input_modalities = set(arch.get("input_modalities") or [])
        output_modalities = set(arch.get("output_modalities") or [])
        model_id = raw.get("id") or getattr(item, "id", None)
        if not model_id:
            continue
        # Exclude catalog entries that are clearly not chat/text generation models.
        if "embedding" in model_id.lower() and "text" not in output_modalities:
            continue
        vision = "image" in input_modalities
        text_input = not input_modalities or "text" in input_modalities
        text_output = not output_modalities or "text" in output_modalities
        if text_input and text_output:
            rows.append({"id": model_id, "vision": vision})
    rows.sort(key=lambda x: (not x["vision"], x["id"].lower()))
    return rows


try:
    model_catalog = get_models(api_key)
    model_error = None
except Exception as exc:
    model_catalog = []
    model_error = str(exc)

st.markdown('<div class="section-title">⚙️ Translation setup</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hint">Choose the OpenRouter model, source language, and target language. Vision-capable models can also OCR text embedded inside PDF images and full-page scans.</div>',
    unsafe_allow_html=True,
)

LANGUAGES = [
    "Japanese", "English", "Chinese", "Korean", "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian", "Arabic", "Hindi", "Tamil", "Telugu", "Malayalam", "Kannada",
    "Thai", "Vietnamese", "Dutch", "Polish", "Ukrainian",
]

model_by_display = {}
for row in model_catalog:
    badge = " 🖼️" if row["vision"] else ""
    display = f"{row['id']}{badge}"
    model_by_display[display] = row

model_options = ["Select an OpenRouter model..."] + list(model_by_display.keys())
source_options = ["Select source language..."] + LANGUAGES

c1, c2, c3 = st.columns(3)
with c1:
    selected_model_display = st.selectbox("🤖 AI Model", model_options, index=0, key="model_selector")
    selected_model = model_by_display.get(selected_model_display, {}).get("id")
    model_is_vision = bool(model_by_display.get(selected_model_display, {}).get("vision"))
with c2:
    selected_source = st.selectbox("🌏 Source Language", source_options, index=0, key="source_selector")
    source_language = None if selected_source == source_options[0] else selected_source
with c3:
    target_options = ["Select target language..."] + [x for x in LANGUAGES if x != source_language]
    selected_target = st.selectbox("🌎 Target Language", target_options, index=0, key="target_selector")
    target_language = None if selected_target == target_options[0] else selected_target

if model_error:
    st.warning(f"OpenRouter model list could not be loaded: {model_error}")
elif not model_catalog:
    st.warning("No text-capable OpenRouter models were returned. Check the API key and network connection.")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        f'<div class="status-card"><div class="status-label">Selected model</div><div class="status-value">{selected_model or "Waiting for selection"}</div></div>',
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        f'<div class="status-card"><div class="status-label">Translation direction</div><div class="status-value">{source_language or "—"} → {target_language or "—"}</div></div>',
        unsafe_allow_html=True,
    )
with s3:
    capability = "Vision + text" if model_is_vision else "Text only"
    st.markdown(
        f'<div class="status-card"><div class="status-label">Model capability</div><div class="status-value">{capability} · {len(model_catalog)} models</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">📄 Document</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload your PDF", type=["pdf"], help="Native text and scanned/image PDFs are supported.")

st.markdown('<div class="section-title">🖼️ Image / OCR behavior</div>', unsafe_allow_html=True)
q1, q2 = st.columns(2)
with q1:
    use_vision_images = st.checkbox(
        "Use OpenRouter vision for image text",
        value=True,
        help="Reads text embedded inside PDF images and full-page scans with the selected model, then overlays the translation inside the exact image-text boxes.",
    )
with q2:
    max_image_edge = st.selectbox(
        "Vision image size",
        [1600, 2000, 2400, 3000],
        index=2,
        help="The original PDF image is never resized in the output. This only controls the copy sent to OpenRouter.",
    )

st.markdown('<div class="section-title">🧠 Extraction</div>', unsafe_allow_html=True)
e1, e2 = st.columns(2)
with e1:
    extraction_mode = st.selectbox(
        "Text extraction mode",
        ["Auto (MuPDF → OCR when needed)", "Native MuPDF only", "OCR"],
        index=0,
        help="Native PDF text is kept on the coordinate-based path. Image-dominant pages use OpenRouter vision when enabled.",
    )
with e2:
    ocr_dpi = st.selectbox("Local OCR quality", [200, 300, 400], index=1, help="Fallback OCR quality for pages that are not handled as image-dominant vision pages.")

st.markdown('<div class="section-title">📐 Layout controls</div>', unsafe_allow_html=True)
l1, l2 = st.columns(2)
with l1:
    min_scale = st.slider("Minimum font scale", 0.55, 1.00, 0.60, 0.01, help="Reduce font size only when translated text cannot fit inside the original text geometry.")
with l2:
    margin = st.slider("Text fitting margin", 0.0, 5.0, 0.0, 0.25, help="Keep at 0 for maximum source-PDF geometry fidelity.")

if use_vision_images and selected_model and not model_is_vision:
    st.warning("The selected OpenRouter model is not marked as image-capable. Choose a model with the 🖼️ badge to translate text embedded inside images/scans.")

if uploaded:
    st.success(f"Loaded: {uploaded.name}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Source", source_language or "Not selected")
    with col2:
        st.metric("Target", target_language or "Not selected")
    with col3:
        st.metric("Model", selected_model or "Not selected")

    if st.button("🚀 Translate PDF", type="primary", use_container_width=True):
        missing = []
        if not selected_model:
            missing.append("AI model")
        if not source_language:
            missing.append("source language")
        if not target_language:
            missing.append("target language")
        if missing:
            st.warning("Please select " + ", ".join(missing) + " before translating.")
            st.stop()
        if use_vision_images and not model_is_vision:
            st.error("Image OCR is enabled, but the selected model does not advertise image input. Pick a 🖼️ vision-capable OpenRouter model.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / uploaded.name
            input_path.write_bytes(uploaded.getvalue())
            output_path = tmpdir / f"{input_path.stem}_{target_language.lower().replace(' ', '_')}.pdf"
            progress = st.progress(0)
            status = st.empty()
            try:
                status.info("Analyzing PDF structure, native text, and embedded images...")
                progress.progress(15)
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
                )
                progress.progress(100)
                status.success("Translation completed.")
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

st.divider()
st.caption("OpenRouter is used for text translation and, when enabled, multimodal OCR/translation of text embedded inside PDF images. The output keeps the original image objects at their source size and position.")

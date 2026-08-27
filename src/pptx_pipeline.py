from __future__ import annotations

import base64
import io
import mimetypes
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from openai import OpenAI, RateLimitError, APIStatusError

def _client(api_key: str) -> OpenAI:
    if not api_key or not api_key.strip():
        raise RuntimeError("OpenRouter API key is missing.")
    return OpenAI(
        api_key=api_key.strip(),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://translator.local",
            "X-Title": "Document Translator Studio",
        },
        max_retries=0,
    )



SOURCE_SCRIPT_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]"
)

# Slide XML is the authoritative location for visible shape/table/text-box
# text. SmartArt text is commonly stored in ppt/diagrams/*.xml.
TEXT_PART_RE = re.compile(
    rb"<a:t(?:\s[^>]*)?>(.*?)</a:t>",
    re.DOTALL,
)


@dataclass
class PptxTranslationReport:
    text_nodes_found: int = 0
    text_nodes_translated: int = 0
    slides_processed: int = 0
    images_found: int = 0
    images_translated: int = 0
    skipped_image_types: int = 0
    slide_layout_parts_preserved: int = 0


def _contains_source(text: str) -> bool:
    return bool(SOURCE_SCRIPT_RE.search(text or ""))


def _message_content(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                out.append(str(text))
        return "".join(out)
    return str(content or "")


def _translate_batch(
    client,
    model: str,
    source_language: str,
    target_language: str,
    rows: list[tuple[str, str]],
) -> dict[str, str]:
    if not rows:
        return {}

    expected = {k for k, _ in rows}

    payload_lines = []
    for key, value in rows:
        safe = (
            str(value)
            .replace("\t", " ")
            .replace("\r", " ")
            .replace("\n", " ")
        )
        payload_lines.append(f"{key}\t{safe}")

    prompt = (
        f"Translate the PowerPoint text from {source_language} to {target_language}.\n\n"
        "This is a real PowerPoint deck. Translate only the supplied text.\n"
        "Preserve technical identifiers, numbers, units, filenames, URLs, "
        "model numbers, reference designators and codes.\n"
        "Keep the translation concise enough for the existing shape/text box.\n"
        "Do not redesign the slide.\n"
        "Return ONLY:\n"
        "KEY<TAB>TRANSLATION\n"
        "No JSON. No Markdown. No explanations.\n\n"
        "INPUT:\n"
        + "\n".join(payload_lines)
    )

    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise PowerPoint document translator.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=16000,
            )

            if not response.choices:
                raise RuntimeError("OpenRouter returned no choices.")

            text = _message_content(response.choices[0].message)
            result = {}

            for line in text.splitlines():
                if "\t" not in line:
                    continue
                key, value = line.split("\t", 1)
                key = key.strip()
                value = value.strip()
                if key in expected and value:
                    result[key] = value

            return result

        except (RateLimitError, APIStatusError) as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            retryable = (
                isinstance(exc, RateLimitError)
                or status in (429, 500, 502, 503, 504)
            )
            if retryable and attempt < 2:
                time.sleep(min(8, 2 ** attempt))
                continue
            raise

        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(min(6, 2 ** attempt))
                continue
            raise RuntimeError(f"PPTX text translation failed: {exc}") from exc

    raise RuntimeError(f"PPTX text translation failed: {last_error}")


def _read_package(path: str):
    with zipfile.ZipFile(path, "r") as zin:
        files = {
            info.filename: zin.read(info.filename)
            for info in zin.infolist()
            if not info.is_dir()
        }
        infos = {
            info.filename: info
            for info in zin.infolist()
            if not info.is_dir()
        }
    return files, infos


def _write_package(files, infos, output_path):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        out,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as zout:
        for name, data in files.items():
            original = infos.get(name)
            if original is None:
                zout.writestr(name, data)
                continue

            zi = zipfile.ZipInfo(
                original.filename,
                date_time=original.date_time,
            )
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.comment = original.comment
            zi.extra = original.extra
            zi.create_system = original.create_system
            zi.create_version = original.create_version
            zi.extract_version = original.extract_version
            zi.flag_bits = original.flag_bits
            zout.writestr(zi, data)


def _discover_text_nodes(files):
    """
    Discover visible PowerPoint text from:
      ppt/slides/slide*.xml
      ppt/diagrams/*.xml

    Only source-language strings are sent to the model.
    """
    items = []

    for path, data in files.items():
        if path.startswith("ppt/slides/") and path.endswith(".xml"):
            pass
        elif path.startswith("ppt/diagrams/") and path.endswith(".xml"):
            pass
        else:
            continue

        matches = list(TEXT_PART_RE.finditer(data))
        for index, match in enumerate(matches):
            raw = match.group(1)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue

            # XML entity decoding for model input.
            import html
            text = html.unescape(text)

            if not text.strip() or not _contains_source(text):
                continue

            items.append((path, index, text))

    return items


def _patch_text_nodes(files, translations):
    """
    Replace only the contents of <a:t> tags in original XML bytes.

    We do NOT parse/serialize slide XML. This preserves:
      - shape geometry
      - groups
      - connectors
      - theme markup
      - extension lists
      - animation/timing metadata
      - vendor-specific PowerPoint XML
    """
    changed = 0

    grouped = {}
    for (path, index), translation in translations.items():
        grouped.setdefault(path, {})[index] = translation

    for path, replacements in grouped.items():
        data = files.get(path)
        if data is None:
            continue

        matches = list(TEXT_PART_RE.finditer(data))
        output = bytearray()
        last = 0

        for index, match in enumerate(matches):
            translation = replacements.get(index)
            if translation is None:
                continue

            import html
            escaped = html.escape(
                str(translation)
                .replace("\r", " ")
                .replace("\n", " ")
                .replace("\t", " "),
                quote=False,
            ).encode("utf-8")

            output.extend(data[last:match.start(1)])
            output.extend(escaped)
            last = match.end(1)
            changed += 1

        if changed:
            output.extend(data[last:])
            files[path] = bytes(output)

    return changed


def _image_data_url(data: bytes, name: str) -> str:
    mime = mimetypes.guess_type(name)[0] or "image/png"
    # Gemini/image APIs support common raster formats. Unsupported formats
    # are skipped instead of damaging the PPTX.
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _image_model_capabilities(api_key: str, model: str) -> dict:
    """
    Read OpenRouter's image-model capability record.

    Do not reject Gemini 3.1 Flash Lite Image merely because its model
    capability record does not expose input_references. PPTX image editing
    is performed through the multimodal chat endpoint below.
    """
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/images/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        for item in response.json().get("data", []):
            if item.get("id") == model:
                return item.get("supported_parameters") or {}
    except Exception:
        pass
    return {}

def _nearest_supported_resolution(capabilities: dict, preferred="2K"):
    data = capabilities.get("resolution") or {}
    values = data.get("values") or []
    if preferred in values:
        return preferred
    for fallback in ("2K", "1K", "512"):
        if fallback in values:
            return fallback
    return None


def _aspect_ratio(width: int, height: int):
    ratio = width / max(height, 1)
    choices = {
        "1:1": 1.0,
        "4:5": 4 / 5,
        "3:4": 3 / 4,
        "2:3": 2 / 3,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
        "3:2": 3 / 2,
        "4:3": 4 / 3,
    }
    return min(choices, key=lambda k: abs(choices[k] - ratio))


def _extract_b64_image(data: dict) -> bytes:
    items = data.get("data") or []
    if not items:
        raise RuntimeError("OpenRouter image API returned no data.")

    b64 = items[0].get("b64_json")
    if not b64:
        raise RuntimeError("OpenRouter image API returned no b64_json output.")

    return base64.b64decode(b64)


def _extract_generated_image_from_chat(response) -> bytes:
    """
    Extract an image generated by an OpenRouter multimodal chat response.

    Gemini image models can return message.images[].image_url.url as a
    data URL. Some providers may return a plain URL; download that URL.
    """
    if not response.choices:
        raise RuntimeError("OpenRouter returned no choices for image editing.")

    message = response.choices[0].message
    images = getattr(message, "images", None) or []

    # SDK object/list form.
    for item in images:
        image_url = None
        if isinstance(item, dict):
            image_url = (item.get("image_url") or {}).get("url")
        else:
            image_obj = getattr(item, "image_url", None)
            image_url = getattr(image_obj, "url", None) if image_obj else None

        if not image_url:
            continue

        if image_url.startswith("data:"):
            try:
                encoded = image_url.split(",", 1)[1]
                return base64.b64decode(encoded)
            except Exception as exc:
                raise RuntimeError(
                    f"OpenRouter returned an invalid image data URL: {exc}"
                ) from exc

        r = requests.get(image_url, timeout=120)
        r.raise_for_status()
        return r.content

    # Some SDK/provider variants expose the image as content parts.
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                image_url = (part.get("image_url") or {}).get("url")
                if image_url:
                    if image_url.startswith("data:"):
                        return base64.b64decode(image_url.split(",", 1)[1])
                    r = requests.get(image_url, timeout=120)
                    r.raise_for_status()
                    return r.content

    raise RuntimeError(
        "Gemini image model returned no generated image. "
        "The response contained no message.images image output."
    )


def _generate_translated_image(
    api_key: str,
    model: str,
    source_bytes: bytes,
    filename: str,
    source_language: str,
    target_language: str,
) -> bytes:
    """
    Translate text embedded in a PPTX image using OpenRouter's multimodal
    chat endpoint.

    IMPORTANT:
    This intentionally does NOT require input_references from
    /api/v1/images/models. Gemini 3.1 Flash Lite Image is invoked directly
    as a multimodal image-output model using /chat/completions.

    The source image is provided as an image_url data URL. The model returns
    the edited image in message.images.
    """
    from PIL import Image

    # Validate source image before sending.
    try:
        source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
        width, height = source.size
    except Exception as exc:
        raise RuntimeError(
            f"Unsupported/corrupt embedded image {filename}: {exc}"
        ) from exc

    image_url = _image_data_url(source_bytes, filename)

    prompt = (
        f"Edit this PowerPoint image and translate ALL visible "
        f"{source_language} text into {target_language}.\n\n"
        "This is an existing presentation image. Return a complete edited "
        "version of the SAME image.\n\n"
        "STRICT RULES:\n"
        "1. Preserve the exact original composition and visual geometry.\n"
        "2. Preserve every diagram, shape, line, arrow, icon, logo and artwork.\n"
        "3. Do not invent, remove or rearrange objects.\n"
        "4. Do not crop or reframe the image.\n"
        "5. Replace/transliterate text only.\n"
        "6. Translate small labels and annotations as well as large text.\n"
        "7. Keep translated text in the same visual location.\n"
        "8. Keep approximate font size, weight, alignment and hierarchy.\n"
        "9. Preserve numbers, units, model numbers and technical identifiers "
        "unless they are ordinary-language text.\n"
        "10. Do not create a new table or redesign the slide.\n"
        "11. Output the complete translated image."
    )

    client = _client(api_key)

    # OpenRouter's image-generation documentation supports image-output
    # models through chat completions with modalities=["image","text"].
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        modalities=["image", "text"],
    )

    result = _extract_generated_image_from_chat(response)

    # Validate generated bytes before replacing PPTX media.
    try:
        Image.open(io.BytesIO(result)).verify()
    except Exception as exc:
        raise RuntimeError(
            f"Gemini returned invalid image bytes for {filename}: {exc}"
        ) from exc

    return result

def _image_media_names(files):
    raster_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
    )
    return [
        name
        for name in files
        if name.startswith("ppt/media/")
        and name.lower().endswith(raster_extensions)
    ]


def translate_pptx(
    input_path: str,
    output_path: str,
    api_key: str,
    text_model: str,
    source_language: str,
    target_language: str,
    image_model: str | None = None,
    translate_native_text: bool = True,
    translate_embedded_images: bool = True,
    translate_diagrams: bool = True,
):
    """
    PPTX translator that preserves the original OOXML package.

    Native PPTX text:
      - text boxes
      - titles
      - paragraphs
      - tables
      - grouped shapes
      - shape text
      - SmartArt/diagram text in ppt/diagrams/*.xml

    Embedded/scanned images:
      - each supported raster in ppt/media/* can be sent to the selected
        OpenRouter image-generation model
      - the generated image replaces only the media bytes
      - the existing relationship and placement/size remain unchanged

    PPTX is never round-tripped through python-pptx, because doing so can
    discard unsupported PowerPoint objects and formatting.
    """
    files, infos = _read_package(input_path)
    client: OpenAI | None = _client(api_key) if translate_native_text else None

    report = PptxTranslationReport()

    # ---------------------------------------------------------
    # 1. Native shape/table/text/diagram text
    # ---------------------------------------------------------
    if translate_native_text:
        if client is None:
            raise RuntimeError("OpenRouter client was not initialized.")
        items = _discover_text_nodes(files)
        if not translate_diagrams:
            items = [
                item for item in items
                if not item[0].startswith("ppt/diagrams/")
            ]
        report.text_nodes_found = len(items)

        rows = [
            (
                f"TEXT::{path}::{index}",
                text,
            )
            for path, index, text in items
        ]

        translations = {}

        for start in range(0, len(rows), 80):
            batch = _translate_batch(
                client,
                text_model,
                source_language,
                target_language,
                rows[start:start + 80],
            )

            for key, value in batch.items():
                parts = key.split("::", 2)
                if len(parts) != 3:
                    continue

                translations[
                    (parts[1], int(parts[2]))
                ] = value

        report.text_nodes_translated = _patch_text_nodes(
            files,
            translations,
        )

    # ---------------------------------------------------------
    # 2. Embedded raster/scanned images
    # ---------------------------------------------------------
    media_files = _image_media_names(files)
    report.images_found = len(media_files)

    if translate_embedded_images:
        if not image_model:
            raise RuntimeError(
                "Embedded image translation is enabled, but no image model was selected."
            )

        for media_name in media_files:
            # Read source media bytes before replacing it.
            source_bytes = files[media_name]

            try:
                translated = _generate_translated_image(
                    api_key=api_key,
                    model=image_model,
                    source_bytes=source_bytes,
                    filename=media_name,
                    source_language=source_language,
                    target_language=target_language,
                )

                files[media_name] = translated
                report.images_translated += 1

            except Exception as exc:
                raise RuntimeError(
                    f"Failed translating embedded image {media_name}: {exc}"
                ) from exc

    # ---------------------------------------------------------
    # 3. Final package
    # ---------------------------------------------------------
    _write_package(
        files,
        infos,
        output_path,
    )

    # Validate ZIP integrity before returning.
    with zipfile.ZipFile(output_path, "r") as test_zip:
        bad = test_zip.testzip()
        if bad:
            raise RuntimeError(
                f"Generated PPTX ZIP is corrupt at: {bad}"
            )

    return {
        "output": str(output_path),
        "text_nodes_found": report.text_nodes_found,
        "text_nodes_translated": report.text_nodes_translated,
        "images_found": report.images_found,
        "images_translated": report.images_translated,
    }

from __future__ import annotations

import base64
import html
import io
import mimetypes
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from openai import OpenAI, RateLimitError, APIStatusError

WT_RE = re.compile(rb"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)
SOURCE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]")


@dataclass
class WordReport:
    text_nodes_found: int = 0
    text_nodes_translated: int = 0
    image_files_found: int = 0
    image_files_translated: int = 0
    unsupported_images: int = 0


def _contains_source(text: str) -> bool:
    return bool(SOURCE_RE.search(text or ""))


def _client(api_key: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing.")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://translator.local",
            "X-Title": "Document Translator Studio",
        },
        max_retries=0,
    )


def _message_content(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                out.append(str(text))
        return "".join(out)
    return str(content or "")


def _translate_batch(client, model, source, target, rows):
    if not rows:
        return {}
    expected = {k for k, _ in rows}
    payload = []
    for key, value in rows:
        safe = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
        payload.append(f"{key}\t{safe}")
    prompt = (
        f"Translate this Microsoft Word text from {source} to {target}.\n"
        "The text may come from paragraphs, tables, headers, footers, "
        "footnotes, endnotes, comments, text boxes, shapes or diagrams.\n"
        "Preserve technical identifiers, numbers, units, codes, filenames and URLs.\n"
        "Return ONLY KEY<TAB>TRANSLATION. No JSON, Markdown or explanations.\n\n"
        + "\n".join(payload)
    )
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise technical Word translator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=16000,
            )
            text = _message_content(r.choices[0].message)
            out = {}
            for line in text.splitlines():
                if "\t" not in line:
                    continue
                key, value = line.split("\t", 1)
                key, value = key.strip(), value.strip()
                if key in expected and value:
                    out[key] = value
            return out
        except (RateLimitError, APIStatusError) as exc:
            status = getattr(exc, "status_code", None)
            retryable = isinstance(exc, RateLimitError) or status in (429, 500, 502, 503, 504)
            if retryable and attempt < 2:
                time.sleep(min(8, 2 ** attempt))
                continue
            raise
        except Exception as exc:
            if attempt < 2:
                time.sleep(min(6, 2 ** attempt))
                continue
            raise RuntimeError(f"Word text translation failed: {exc}") from exc
    return {}


def _read_package(path):
    with zipfile.ZipFile(path, "r") as zin:
        files = {i.filename: zin.read(i.filename) for i in zin.infolist() if not i.is_dir()}
        infos = {i.filename: i for i in zin.infolist() if not i.is_dir()}
    return files, infos


def _write_package(files, infos, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zout:
        for name, data in files.items():
            old = infos.get(name)
            if old is None:
                zout.writestr(name, data)
                continue
            info = zipfile.ZipInfo(old.filename, date_time=old.date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.comment = old.comment
            info.extra = old.extra
            info.create_system = old.create_system
            info.create_version = old.create_version
            info.extract_version = old.extract_version
            info.flag_bits = old.flag_bits
            zout.writestr(info, data)


def _word_text_parts(files):
    parts = []
    for path in files:
        if not (path.startswith("word/") and path.endswith(".xml")):
            continue
        name = Path(path).name.lower()
        if (
            path == "word/document.xml"
            or name.startswith("header")
            or name.startswith("footer")
            or name in {"footnotes.xml", "endnotes.xml", "comments.xml"}
            or path == "word/glossary/document.xml"
        ):
            parts.append(path)
    return parts


def _collect_text_nodes(files):
    items = []
    for path in _word_text_parts(files):
        matches = list(WT_RE.finditer(files[path]))
        for index, match in enumerate(matches):
            try:
                text = html.unescape(match.group(1).decode("utf-8"))
            except UnicodeDecodeError:
                continue
            if text.strip() and _contains_source(text):
                items.append((path, index, text))
    return items


def _patch_text_nodes(files, translations):
    grouped = {}
    for (path, index), value in translations.items():
        grouped.setdefault(path, {})[index] = value
    changed = 0
    for path, replacements in grouped.items():
        data = files.get(path)
        if data is None:
            continue
        matches = list(WT_RE.finditer(data))
        out = bytearray()
        last = 0
        path_changed = False
        for index, match in enumerate(matches):
            value = replacements.get(index)
            if value is None:
                continue
            escaped = html.escape(
                str(value).replace("\r", " ").replace("\n", " ").replace("\t", " "),
                quote=False,
            ).encode("utf-8")
            out.extend(data[last:match.start(1)])
            out.extend(escaped)
            last = match.end(1)
            changed += 1
            path_changed = True
        if path_changed:
            out.extend(data[last:])
            files[path] = bytes(out)
    return changed


def _image_data_url(data, filename):
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _extract_generated_image(response):
    message = response.choices[0].message
    for item in (getattr(message, "images", None) or []):
        if isinstance(item, dict):
            url = (item.get("image_url") or {}).get("url")
        else:
            obj = getattr(item, "image_url", None)
            url = getattr(obj, "url", None) if obj else None
        if not url:
            continue
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return r.content
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                url = (part.get("image_url") or {}).get("url")
                if url:
                    if url.startswith("data:"):
                        return base64.b64decode(url.split(",", 1)[1])
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    return r.content
    raise RuntimeError("Gemini image model returned no generated image.")


def _generate_translated_image(api_key, model, source_bytes, filename, source, target):
    from PIL import Image
    try:
        Image.open(io.BytesIO(source_bytes)).verify()
    except Exception as exc:
        raise RuntimeError(f"Unsupported or invalid image {filename}: {exc}") from exc

    prompt = (
        f"Edit this Microsoft Word image and translate ALL visible {source} text "
        f"into {target}.\n\n"
        "Preserve the exact composition, geometry, diagrams, tables, borders, "
        "arrows, logos, icons and artwork. Do not crop, rotate, redesign or add "
        "objects. Replace text only. Translate small labels and annotations too. "
        "Keep translated text in the same visual locations and preserve approximate "
        "font size, weight, alignment and hierarchy. Return the complete edited image."
    )

    client = _client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_data_url(source_bytes, filename)}},
            ],
        }],
        modalities=["image", "text"],
    )
    result = _extract_generated_image(response)
    try:
        Image.open(io.BytesIO(result)).verify()
    except Exception as exc:
        raise RuntimeError(f"Gemini returned invalid image for {filename}: {exc}") from exc
    return result


def _media_names(files):
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
    return [
        name for name in files
        if name.startswith("word/media/") and name.lower().endswith(extensions)
    ]


def _validate_docx(path):
    with zipfile.ZipFile(path, "r") as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Generated DOCX ZIP is corrupt: {bad}")
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                from xml.etree import ElementTree as ET
                ET.fromstring(z.read(name))


def translate_word(
    input_path: str,
    output_path: str,
    api_key: str,
    text_model: str,
    source_language: str,
    target_language: str,
    image_model: str | None = None,
    translate_native_text: bool = True,
    translate_images: bool = True,
):
    """
    XML-preserving .docx translator.

    Native:
      paragraphs, tables, headers, footers, footnotes, endnotes, comments,
      text boxes and other WordprocessingML text.

    Images:
      all common raster media under word/media are sent to the selected
      Gemini image-output model. Only the media bytes are replaced, so
      Word relationship, anchor, position, displayed size and wrapping stay.
    """
    if not str(input_path).lower().endswith(".docx"):
        raise RuntimeError("This Word pipeline requires .docx input. Convert legacy .doc first.")

    files, infos = _read_package(input_path)
    client = _client(api_key)
    report = WordReport()

    if translate_native_text:
        items = _collect_text_nodes(files)
        report.text_nodes_found = len(items)
        rows = [(f"TEXT::{path}::{index}", text) for path, index, text in items]
        result = {}
        for start in range(0, len(rows), 80):
            result.update(
                _translate_batch(
                    client, text_model, source_language, target_language,
                    rows[start:start + 80],
                )
            )

        translations = {}
        for key, value in result.items():
            parts = key.split("::", 2)
            if len(parts) == 3:
                try:
                    translations[(parts[1], int(parts[2]))] = value
                except ValueError:
                    pass

        report.text_nodes_translated = _patch_text_nodes(files, translations)

    media = _media_names(files)
    report.image_files_found = len(media)

    if translate_images:
        if not image_model:
            raise RuntimeError("Word image translation is enabled but no image model was selected.")
        for name in media:
            try:
                files[name] = _generate_translated_image(
                    api_key, image_model, files[name], name, source_language, target_language
                )
                report.image_files_translated += 1
            except Exception as exc:
                raise RuntimeError(f"Failed translating Word embedded image {name}: {exc}") from exc

    _write_package(files, infos, output_path)
    _validate_docx(output_path)

    return {
        "output": str(output_path),
        "text_nodes_found": report.text_nodes_found,
        "text_nodes_translated": report.text_nodes_translated,
        "image_files_found": report.image_files_found,
        "image_files_translated": report.image_files_translated,
        "mode": "OOXML-preserving-docx",
    }

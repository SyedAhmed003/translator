
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import xml.etree.ElementTree as ET

from openai import OpenAI, RateLimitError, APIStatusError


NS = {
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

ET.register_namespace("xdr", NS["xdr"])
ET.register_namespace("a", NS["a"])


@dataclass
class DrawingText:
    drawing_path: str
    text_index: int
    source: str


def _message_content(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                result.append(str(text))
        return "".join(result)
    return str(content or "")


def _is_probably_translatable(text: str) -> bool:
    text = text.strip()
    if not text:
        return False

    # Avoid translating cells/shape labels that are only identifiers/numbers.
    if re.fullmatch(r"[\d\s.,:+\-*/=()_%#<>A-Za-z_./\\-]+", text):
        return False

    # Japanese/Hiragana/Katakana/Kanji.
    if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text):
        return True

    # Other common source scripts can pass as natural language.
    if re.search(r"[\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]", text):
        return True

    return False


def _safe_key(path: str, index: int) -> str:
    return f"DRAWING::{path}::{index}"


class ExcelDrawingTranslator:
    """
    Translates text inside Excel DrawingML while preserving the ORIGINAL XML.

    This is intentionally XML based instead of using openpyxl to save the
    workbook. That prevents openpyxl from dropping unsupported drawing shapes,
    text boxes, connectors and other DrawingML objects.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        source_language: str,
        target_language: str,
        http_referer: str = "",
        app_name: str = "Document Translator Studio",
        batch_size: int = 100,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing.")
        if not model:
            raise RuntimeError("An OpenRouter text model is required.")

        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.batch_size = max(1, batch_size)

        headers = {}
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_name:
            headers["X-Title"] = app_name

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
            max_retries=0,
        )

    def _collect(self, files: Dict[str, bytes]) -> List[DrawingText]:
        found = []

        for path, raw in files.items():
            if not path.startswith("xl/drawings/") or not path.endswith(".xml"):
                continue

            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue

            index = 0
            for elem in root.iter():
                if elem.tag == f"{{{NS['a']}}}t":
                    value = elem.text or ""
                    if _is_probably_translatable(value):
                        found.append(
                            DrawingText(
                                drawing_path=path,
                                text_index=index,
                                source=value,
                            )
                        )
                    index += 1

        return found

    def _request_batch(self, batch: List[DrawingText]) -> Dict[str, str]:
        input_lines = []

        for item in batch:
            key = _safe_key(item.drawing_path, item.text_index)
            safe = item.source.replace("\t", " ").replace("\r", " ").replace("\n", " ")
            input_lines.append(f"{key}\t{safe}")

        block = "\n".join(input_lines)

        prompt = (
            f"Translate the following Excel drawing/shape text from "
            f"{self.source_language} to {self.target_language}.\n\n"
            "These are labels inside an engineering spreadsheet diagram.\n\n"
            "Rules:\n"
            "1. Translate natural-language text faithfully.\n"
            "2. Preserve technical identifiers, component numbers, reference "
            "designators, model numbers and numeric values.\n"
            "3. Do not invent information.\n"
            "4. Keep the translation concise enough to fit the original shape.\n"
            "5. Return exactly one line for every supplied key.\n"
            "6. Keep the key EXACTLY unchanged.\n"
            "7. Return only KEY<TAB>TRANSLATION.\n"
            "8. No JSON, Markdown, explanations or numbering.\n\n"
            "INPUT:\n"
            + block
        )

        last_error = None

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise technical-document translator. "
                                "Never alter identifiers."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=12000,
                )

                content = _message_content(response.choices[0].message)
                result = {}

                expected = {
                    _safe_key(x.drawing_path, x.text_index)
                    for x in batch
                }

                for line in content.splitlines():
                    line = line.strip()
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
                raise RuntimeError(
                    f"Drawing translation request failed: {exc}"
                ) from exc

        raise RuntimeError(f"Drawing translation failed: {last_error}")

    def translate_zip(self, input_path: str, output_path: str) -> dict:
        """
        Reads the XLSX as ZIP/XML, patches only DrawingML <a:t> text nodes,
        and writes every other ZIP entry byte-for-byte unchanged.
        """
        with zipfile.ZipFile(input_path, "r") as zin:
            files = {
                name: zin.read(name)
                for name in zin.namelist()
                if not name.endswith("/")
            }

        items = self._collect(files)

        if not items:
            self._write_zip(files, output_path)
            return {
                "drawings_found": 0,
                "shape_texts_found": 0,
                "shape_texts_translated": 0,
            }

        translations = {}

        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            translations.update(self._request_batch(batch))

        changed = 0

        # Rebuild only drawing XML files containing translations.
        by_path: Dict[str, List[DrawingText]] = {}
        for item in items:
            by_path.setdefault(item.drawing_path, []).append(item)

        for path, path_items in by_path.items():
            root = ET.fromstring(files[path])
            text_nodes = [
                elem
                for elem in root.iter()
                if elem.tag == f"{{{NS['a']}}}t"
            ]

            for item in path_items:
                key = _safe_key(item.drawing_path, item.text_index)
                translated = translations.get(key)

                if not translated:
                    continue

                if item.text_index >= len(text_nodes):
                    continue

                node = text_nodes[item.text_index]

                # Do not allow translation to contain line/tab control chars
                # that could corrupt DrawingML text.
                translated = (
                    translated
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .replace("\t", " ")
                )

                node.text = translated
                changed += 1

            buffer = io.BytesIO()
            ET.ElementTree(root).write(
                buffer,
                encoding="utf-8",
                xml_declaration=True,
            )
            files[path] = buffer.getvalue()

        self._write_zip(files, output_path)

        return {
            "drawings_found": len(by_path),
            "shape_texts_found": len(items),
            "shape_texts_translated": changed,
        }

    @staticmethod
    def _write_zip(files: Dict[str, bytes], output_path: str):
        with zipfile.ZipFile(
            output_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zout:
            for name, data in files.items():
                zout.writestr(name, data)

from __future__ import annotations

import html
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from openai import OpenAI, RateLimitError, APIStatusError


# IMPORTANT:
# This version NEVER parses-and-reserializes the XML that is written back.
# XML is read only for discovery. Actual changes are byte/text replacements
# inside the original XML bytes. This avoids losing OOXML namespace declarations,
# extension namespaces, prefixes, mc:Ignorable, drawing extensions, etc.


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _message_content(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts)

    return str(content or "")


def _client(api_key: str):
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=0,
    )


def _contains_source_script(text: str) -> bool:
    return bool(
        re.search(
            r"[\u3040-\u30ff"
            r"\u3400-\u4dbf"
            r"\u4e00-\u9fff"
            r"\uac00-\ud7af"
            r"\u0400-\u04ff"
            r"\u0600-\u06ff]",
            text or "",
        )
    )


def _translate_batch(
    client,
    model,
    source_language,
    target_language,
    rows,
):
    if not rows:
        return {}

    expected = {key for key, _ in rows}

    prompt_lines = []
    for key, value in rows:
        safe = (
            str(value)
            .replace("\t", " ")
            .replace("\r", " ")
            .replace("\n", " ")
        )
        prompt_lines.append(f"{key}\t{safe}")

    prompt = (
        f"Translate the following Excel text from {source_language} "
        f"to {target_language}.\n\n"
        "Return ONLY one line for every input item:\n"
        "KEY<TAB>TRANSLATION\n\n"
        "Rules:\n"
        "- Translate natural language accurately.\n"
        "- Preserve engineering identifiers, reference designators, "
        "model numbers, codes, numbers and units.\n"
        "- Do not add explanations.\n"
        "- Do not return JSON or Markdown.\n"
        "- Do not modify KEY.\n"
        "- Keep the translation concise enough for its original area.\n\n"
        "INPUT:\n" + "\n".join(prompt_lines)
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise technical spreadsheet translator."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=16000,
            )

            if not response.choices:
                raise RuntimeError("OpenRouter returned no choices.")

            content = _message_content(response.choices[0].message)
            result = {}

            for line in content.splitlines():
                if "\t" not in line:
                    continue

                key, value = line.split("\t", 1)
                key = key.strip()
                value = value.strip()

                if key in expected and value:
                    result[key] = value

            return result

        except (RateLimitError, APIStatusError) as exc:
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
            if attempt < 2:
                time.sleep(min(6, 2 ** attempt))
                continue

            raise RuntimeError(
                f"OpenRouter translation failed: {exc}"
            ) from exc

    return {}


def _read_xlsx(path):
    with zipfile.ZipFile(path, "r") as zin:
        names = zin.namelist()
        files = {
            name: zin.read(name)
            for name in names
            if not name.endswith("/")
        }
        infos = {
            info.filename: info
            for info in zin.infolist()
            if not info.is_dir()
        }
    return files, infos


def _write_xlsx(files, infos, output_path):
    """
    Write a valid XLSX while preserving the original package order and
    metadata where possible. No XML is reserialized here.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as zout:
        for name, data in files.items():
            original = infos.get(name)

            if original is None:
                zout.writestr(name, data)
                continue

            info = zipfile.ZipInfo(
                original.filename,
                date_time=original.date_time,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.comment = original.comment
            info.extra = original.extra
            info.create_system = original.create_system
            info.create_version = original.create_version
            info.extract_version = original.extract_version
            info.flag_bits = original.flag_bits

            zout.writestr(info, data)


def _workbook_sheets(files):
    """
    Read workbook.xml/workbook rels for discovery only.
    Nothing returned by this function is serialized back.
    """
    workbook_root = ET.fromstring(files["xl/workbook.xml"])
    rels_root = ET.fromstring(files["xl/_rels/workbook.xml.rels"])

    targets = {}

    for rel in rels_root:
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")

        if target.startswith("/"):
            target = target[1:]

        if not target.startswith("xl/"):
            target = "xl/" + target

        targets[rid] = target

    sheets = workbook_root.find(
        f"{{{NS_MAIN}}}sheets"
    )

    result = []

    if sheets is None:
        return result

    for index, sheet in enumerate(sheets):
        name = sheet.attrib.get("name")
        rid = sheet.attrib.get(
            f"{{{NS_REL}}}id"
        )

        result.append(
            {
                "index": index,
                "name": name,
                "path": targets.get(rid),
            }
        )

    return result


def _shared_strings(files):
    raw = files.get("xl/sharedStrings.xml")

    if not raw:
        return []

    root = ET.fromstring(raw)
    values = []

    for si in root.findall(
        f"{{{NS_MAIN}}}si"
    ):
        values.append(
            "".join(
                node.text or ""
                for node in si.iter(
                    f"{{{NS_MAIN}}}t"
                )
            )
        )

    return values


def _native_cells(files):
    """
    Discover native source-language cells from the ORIGINAL XML.
    """
    shared = _shared_strings(files)
    result = []

    for sheet in _workbook_sheets(files):
        path = sheet["path"]

        if not path or path not in files:
            continue

        root = ET.fromstring(files[path])

        for cell in root.iter(
            f"{{{NS_MAIN}}}c"
        ):
            coordinate = cell.attrib.get("r")

            if not coordinate:
                continue

            cell_type = cell.attrib.get("t")

            if cell_type == "inlineStr":
                value = "".join(
                    node.text or ""
                    for node in cell.iter(
                        f"{{{NS_MAIN}}}t"
                    )
                )
            else:
                v = cell.find(
                    f"{{{NS_MAIN}}}v"
                )

                if v is None:
                    continue

                value = v.text or ""

                if cell_type == "s":
                    try:
                        index = int(value)
                        value = (
                            shared[index]
                            if 0 <= index < len(shared)
                            else value
                        )
                    except ValueError:
                        pass

            value = str(value).strip()

            if not value or value.startswith("="):
                continue

            if _contains_source_script(value):
                result.append(
                    (
                        sheet["name"],
                        coordinate,
                        value,
                    )
                )

    return result


# ------------------------------------------------------------
# Direct XML replacement helpers
# ------------------------------------------------------------

def _xml_escape_text(value: str) -> str:
    """
    XML text-node escaping.
    """
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_escape_attribute(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _replace_nth_cell(
    xml: bytes,
    coordinate: str,
    translation: str,
) -> tuple[bytes, bool]:
    """
    Replace a single <c r="COORD"...>...</c> cell while preserving the
    original worksheet XML byte-for-byte everywhere else.

    We change the cell to inlineStr. This avoids modifying sharedStrings.xml.
    """
    coord_pattern = re.escape(coordinate)

    pattern = re.compile(
        rb"<c\b(?=[^>]*\br\s*=\s*['\"]"
        + coord_pattern.encode("ascii")
        + rb"['\"])[^>]*>.*?</c>",
        re.DOTALL,
    )

    match = pattern.search(xml)

    if not match:
        # Some empty cells can be self-closing.
        self_pattern = re.compile(
            rb"<c\b(?=[^>]*\br\s*=\s*['\"]"
            + coord_pattern.encode("ascii")
            + rb"['\"])[^>]*/>",
            re.DOTALL,
        )
        match = self_pattern.search(xml)

    if not match:
        return xml, False

    original = match.group(0)

    start_tag_match = re.match(
        rb"(<c\b[^>]*?)(/?>)",
        original,
        re.DOTALL,
    )

    if not start_tag_match:
        return xml, False

    start_tag = start_tag_match.group(1).decode(
        "utf-8",
        errors="strict",
    )

    # Preserve every attribute except t.
    if re.search(
        r"\bt\s*=",
        start_tag,
    ):
        start_tag = re.sub(
            r"\bt\s*=\s*(['\"])[^'\"]*\1",
            't="inlineStr"',
            start_tag,
            count=1,
        )
    else:
        start_tag += ' t="inlineStr"'

    text = _xml_escape_text(translation)

    replacement = (
        start_tag.encode("utf-8")
        + b">"
        + b"<is><t>"
        + text.encode("utf-8")
        + b"</t></is>"
        + b"</c>"
    )

    return (
        xml[:match.start()]
        + replacement
        + xml[match.end():],
        True,
    )


def _patch_native_cells(files, translations):
    changed = 0

    sheet_paths = {
        sheet["name"]: sheet["path"]
        for sheet in _workbook_sheets(files)
        if sheet["name"] and sheet["path"]
    }

    grouped = {}

    for (sheet_name, coordinate), translation in translations.items():
        path = sheet_paths.get(sheet_name)

        if path:
            grouped.setdefault(path, []).append(
                (coordinate, translation)
            )

    for path, changes in grouped.items():
        xml = files[path]

        for coordinate, translation in changes:
            xml, did_change = _replace_nth_cell(
                xml,
                coordinate,
                translation,
            )

            if did_change:
                changed += 1

        files[path] = xml

    return changed


def _patch_sheet_names(
    files,
    translations,
):
    """
    Direct byte-level replacement of workbook.xml sheet name attributes.

    No XML parser writes workbook.xml, so namespace declarations and
    extension markup remain byte-for-byte unchanged.
    """
    xml = files["xl/workbook.xml"]

    # Match each <sheet ... name="..."> independently.
    pattern = re.compile(
        rb"(<sheet\b[^>]*\bname\s*=\s*)(['\"])(.*?)(\2)",
        re.DOTALL,
    )

    matches = list(pattern.finditer(xml))

    if not matches:
        return {}

    changes = {}
    output = bytearray()
    last = 0

    for index, match in enumerate(matches):
        old = html.unescape(
            match.group(3).decode(
                "utf-8",
                errors="strict",
            )
        )

        new = translations.get(
            index,
            old,
        )

        if new == old:
            continue

        new = re.sub(
            r"[\[\]:*?/\\]",
            " ",
            new,
        )
        new = re.sub(
            r"\s+",
            " ",
            new,
        ).strip()

        if not new:
            new = old

        new = new[:31]

        replacement = (
            match.group(1)
            + match.group(2)
            + _xml_escape_attribute(new).encode("utf-8")
            + match.group(4)
        )

        output.extend(
            xml[last:match.start()]
        )
        output.extend(replacement)
        last = match.end()

        changes[old] = new

    if not output:
        return {}

    output.extend(xml[last:])
    files["xl/workbook.xml"] = bytes(output)

    return changes


def _patch_drawing_text(
    files,
    translations,
):
    """
    Direct byte-level replacement of <a:t>...</a:t>.

    This is the critical corruption fix:
    we do NOT parse and serialize drawing1.xml/drawing2.xml.
    Therefore namespace declarations, mc:Ignorable, extensions, grouped
    shapes, connectors and vendor-specific DrawingML survive untouched.
    """
    changed = 0

    for path, path_translations in translations.items():
        xml = files.get(path)

        if xml is None:
            continue

        pattern = re.compile(
            rb"(<a:t(?:\s[^>]*)?>)(.*?)(</a:t>)",
            re.DOTALL,
        )

        matches = list(pattern.finditer(xml))

        if not matches:
            continue

        output = bytearray()
        last = 0

        for index, match in enumerate(matches):
            translation = path_translations.get(index)

            if not translation:
                continue

            replacement = (
                match.group(1)
                + _xml_escape_text(
                    translation
                ).encode("utf-8")
                + match.group(3)
            )

            output.extend(
                xml[last:match.start()]
            )
            output.extend(replacement)
            last = match.end()

            changed += 1

        if changed:
            output.extend(xml[last:])
            files[path] = bytes(output)

    return changed


def _collect_drawing_rows(files):
    """
    Discover source-language <a:t> nodes without modifying the XML.
    """
    rows = []
    path_indexes = {}

    pattern = re.compile(
        rb"<a:t(?:\s[^>]*)?>(.*?)</a:t>",
        re.DOTALL,
    )

    for path, xml in files.items():
        if not (
            path.startswith("xl/drawings/")
            and path.endswith(".xml")
        ):
            continue

        matches = list(pattern.finditer(xml))
        path_indexes[path] = matches

        for index, match in enumerate(matches):
            raw_text = match.group(1)

            try:
                text = html.unescape(
                    raw_text.decode(
                        "utf-8",
                        errors="strict",
                    )
                )
            except UnicodeDecodeError:
                continue

            text = text.strip()

            if text and _contains_source_script(text):
                rows.append(
                    (
                        path,
                        index,
                        text,
                    )
                )

    return rows


def _safe_sheet_name(name):
    name = re.sub(
        r"[\[\]:*?/\\]",
        " ",
        name or "",
    )
    name = re.sub(
        r"\s+",
        " ",
        name,
    ).strip()

    return (name or "Sheet")[:31]


def translate_excel_workbook(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str,
    image_model: str | None,
    source_language: str,
    target_language: str,
    translate_native_cells: bool = True,
    translate_images: bool = True,
    translate_shapes: bool = True,
):
    """
    CORRUPTION-SAFE XLSX XML pipeline.

    The key rule is:
        NEVER parse and reserialize an OOXML part that we modify.

    We read XML only to discover text. We patch the original XML bytes
    directly. This preserves Excel-specific namespace declarations and
    extension markup.

    The original package remains the base, including:
      - all worksheets
      - all drawings
      - all drawing relationships
      - all media
      - charts
      - grouped shapes
      - connectors
      - workbook relationships
      - content types
    """
    if Path(input_path).suffix.lower() != ".xlsx":
        raise RuntimeError("This XML-preserving pipeline requires .xlsx.")

    client = _client(api_key)

    files, infos = _read_xlsx(input_path)

    report = {
        "output": str(output_path),
        "native_cells_detected": 0,
        "native_cells_translated": 0,
        "sheet_names_translated": {},
        "drawing": {
            "drawing_files": 0,
            "drawing_texts_found": 0,
            "drawing_texts_translated": 0,
        },
        "images": {
            "preserved": True,
            "translated": 0,
        },
    }

    # ------------------------------------------------------------
    # 1. Native cells
    # ------------------------------------------------------------
    if translate_native_cells:
        native = _native_cells(files)
        report["native_cells_detected"] = len(native)

        rows = [
            (
                f"CELL::{sheet}::{coordinate}",
                text,
            )
            for sheet, coordinate, text in native
        ]

        result = {}

        for start in range(0, len(rows), 80):
            result.update(
                _translate_batch(
                    client,
                    model,
                    source_language,
                    target_language,
                    rows[start:start + 80],
                )
            )

        translations = {}

        for key, value in result.items():
            parts = key.split("::", 2)

            if len(parts) == 3:
                translations[
                    (parts[1], parts[2])
                ] = value

        report["native_cells_translated"] = _patch_native_cells(
            files,
            translations,
        )

    # ------------------------------------------------------------
    # 2. Sheet names
    # ------------------------------------------------------------
    if True:
        # Read original names.
        workbook_root = ET.fromstring(
            files["xl/workbook.xml"]
        )
        sheets = workbook_root.find(
            f"{{{NS_MAIN}}}sheets"
        )

        name_rows = []

        if sheets is not None:
            for index, sheet in enumerate(sheets):
                name = sheet.attrib.get("name")

                if name and _contains_source_script(name):
                    name_rows.append(
                        (
                            f"SHEET::{index}",
                            name,
                        )
                    )

        result = {}

        for start in range(0, len(name_rows), 80):
            result.update(
                _translate_batch(
                    client,
                    model,
                    source_language,
                    target_language,
                    name_rows[start:start + 80],
                )
            )

        by_index = {}

        for key, value in result.items():
            try:
                index = int(
                    key.split("::", 1)[1]
                )
                by_index[index] = _safe_sheet_name(value)
            except Exception:
                pass

        report["sheet_names_translated"] = _patch_sheet_names(
            files,
            by_index,
        )

    # ------------------------------------------------------------
    # 3. DrawingML text
    # ------------------------------------------------------------
    if translate_shapes:
        drawing_items = _collect_drawing_rows(files)

        report["drawing"]["drawing_files"] = len(
            {
                path
                for path, _, _ in drawing_items
            }
        )
        report["drawing"]["drawing_texts_found"] = len(
            drawing_items
        )

        rows = [
            (
                f"DRAW::{path}::{index}",
                text,
            )
            for path, index, text in drawing_items
        ]

        result = {}

        for start in range(0, len(rows), 80):
            result.update(
                _translate_batch(
                    client,
                    model,
                    source_language,
                    target_language,
                    rows[start:start + 80],
                )
            )

        by_path = {}

        for key, value in result.items():
            parts = key.split("::", 2)

            if len(parts) != 3:
                continue

            path = parts[1]

            try:
                index = int(parts[2])
            except ValueError:
                continue

            by_path.setdefault(
                path,
                {},
            )[index] = value

        report["drawing"]["drawing_texts_translated"] = (
            _patch_drawing_text(
                files,
                by_path,
            )
        )

    # ------------------------------------------------------------
    # 4. Embedded images
    # ------------------------------------------------------------
    # Preserve them here. The Gemini image branch should replace only the
    # bytes under xl/media/* in a separate direct-media operation.
    # It must never rebuild drawings through openpyxl.
    report["images"] = {
        "preserved": True,
        "translated": 0,
    }

    # ------------------------------------------------------------
    # 5. Final package
    # ------------------------------------------------------------
    _write_xlsx(
        files,
        infos,
        output_path,
    )

    # ------------------------------------------------------------
    # 6. Self-validation: make sure the output is a readable ZIP and every
    # XML entry is parseable. This catches the exact class of corruption
    # that was happening before returning the file to Streamlit.
    # ------------------------------------------------------------
    with zipfile.ZipFile(output_path, "r") as test_zip:
        bad_member = test_zip.testzip()

        if bad_member:
            raise RuntimeError(
                f"Generated XLSX ZIP is corrupt at member: {bad_member}"
            )

        for name in test_zip.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(
                    test_zip.read(name)
                )

    return report
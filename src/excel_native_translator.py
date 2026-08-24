from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import time

from openai import OpenAI, RateLimitError, APIStatusError


@dataclass
class CellTranslation:
    sheet: str
    coordinate: str
    source: str
    translation: str


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


def _parse_lines(content: str, expected_ids: set[str]) -> dict[str, str]:
    """
    Expected model response:

        SHEET!A1<TAB>translated text
        SHEET!B7<TAB>translated text

    We deliberately use a simple line protocol instead of JSON.
    """
    result: dict[str, str] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if not line or "\t" not in line:
            continue

        key, value = line.split("\t", 1)

        key = key.strip()
        value = value.strip()

        if key in expected_ids and value:
            result[key] = value

    return result


def _chunks(items: list[tuple[str, str]], size: int = 80):
    for start in range(0, len(items), size):
        yield items[start:start + size]


class ExcelNativeTranslator:
    """
    Translate actual Excel cell values while preserving workbook structure.

    This class does NOT use OCR for native Excel cells.

    It translates:
        actual cell value -> translated cell value

    It leaves:
        formulas
        formatting
        merged cells
        row heights
        column widths
        sheet structure

    untouched.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        source_language: str,
        target_language: str,
        http_referer: str = "",
        app_name: str = "Document Translator Studio",
        max_retries: int = 3,
        batch_size: int = 80,
    ):
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. "
                "Add it to the project's .env file."
            )

        if not model:
            raise RuntimeError(
                "No OpenRouter text model was selected."
            )

        if source_language == target_language:
            raise RuntimeError(
                "Source and target languages must be different."
            )

        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.max_retries = max(1, int(max_retries))
        self.batch_size = max(1, int(batch_size))

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

    def _build_prompt(
        self,
        sheet: str,
        cells: list[tuple[str, str]],
    ) -> tuple[str, set[str]]:
        expected_ids = {
            f"{sheet}!{coordinate}"
            for coordinate, _ in cells
        }

        input_lines = []

        for coordinate, text in cells:
            # Remove tabs/newlines from cell content because the response
            # protocol uses TAB as its field separator and one cell per line.
            safe_text = str(text)
            safe_text = safe_text.replace("\t", " ")
            safe_text = safe_text.replace("\r", " ")
            safe_text = safe_text.replace("\n", " ")

            input_lines.append(
                f"{sheet}!{coordinate}\t{safe_text}"
            )

        numbered_input = "\n".join(input_lines)

        prompt = (
            f"Translate the Excel cell contents from "
            f"{self.source_language} to {self.target_language}.\n\n"
            "You are translating a real Excel workbook. "
            "Do not redesign or reinterpret the spreadsheet.\n\n"
            "Rules:\n"
            "1. Translate only the supplied cell text.\n"
            "2. Keep names, codes, numbers, units and technical terminology accurate.\n"
            "3. Do not translate formulas; formulas are never supplied to you.\n"
            "4. Keep the translation concise enough for the original cell.\n"
            "5. Do not add explanations or commentary.\n"
            "6. Do not add or remove rows or columns.\n"
            "7. Do not merge cells.\n"
            "8. Preserve dates, identifiers, abbreviations and measurement units.\n"
            "9. Return exactly ONE output line for each input cell that you can translate.\n"
            "10. Use the exact CELL_ID from the input.\n\n"
            "OUTPUT FORMAT:\n"
            "CELL_ID<TAB>TRANSLATION\n\n"
            "Do not return JSON.\n"
            "Do not use Markdown.\n"
            "Do not add numbering.\n"
            "Do not repeat the source text separately.\n\n"
            "INPUT CELLS:\n"
            + numbered_input
        )

        return prompt, expected_ids

    def _request(
        self,
        prompt: str,
    ) -> str:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise spreadsheet translation engine. "
                                "Follow the requested line-based output protocol exactly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0,
                    max_tokens=12000,
                )

                if not response.choices:
                    raise RuntimeError(
                        "OpenRouter returned no choices for the Excel translation request."
                    )

                content = _message_content(response.choices[0].message)

                if not content.strip():
                    raise RuntimeError(
                        "OpenRouter returned an empty Excel translation response. "
                        f"finish_reason={response.choices[0].finish_reason!r}"
                    )

                return content

            except (RateLimitError, APIStatusError) as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)

                retryable = (
                    isinstance(exc, RateLimitError)
                    or status in (429, 500, 502, 503, 504)
                )

                if retryable and attempt + 1 < self.max_retries:
                    time.sleep(min(10, 2 ** attempt))
                    continue

                raise

            except Exception as exc:
                last_error = exc

                if attempt + 1 < self.max_retries:
                    time.sleep(min(6, 2 ** attempt))
                    continue

                raise RuntimeError(
                    f"Excel translation request failed for model "
                    f"{self.model!r}: {exc}"
                ) from exc

        raise RuntimeError(
            f"Excel translation request failed: {last_error}"
        )

    def _translate_chunk(
        self,
        sheet: str,
        cells: list[tuple[str, str]],
    ) -> dict[str, str]:
        prompt, expected_ids = self._build_prompt(sheet, cells)

        content = self._request(prompt)
        translated = _parse_lines(content, expected_ids)

        # Retry only missing cells, rather than resending the whole batch.
        missing = expected_ids - set(translated)

        if missing:
            missing_cells = [
                (coordinate, text)
                for coordinate, text in cells
                if f"{sheet}!{coordinate}" in missing
            ]

            if missing_cells:
                retry_input = []

                for coordinate, text in missing_cells:
                    safe_text = str(text)
                    safe_text = safe_text.replace("\t", " ")
                    safe_text = safe_text.replace("\r", " ")
                    safe_text = safe_text.replace("\n", " ")

                    retry_input.append(
                        f"{sheet}!{coordinate}\t{safe_text}"
                    )

                retry_block = "\n".join(retry_input)

                retry_prompt = (
                    f"Translate ONLY these missing Excel cells from "
                    f"{self.source_language} to {self.target_language}.\n\n"
                    "Return exactly:\n"
                    "CELL_ID<TAB>TRANSLATION\n\n"
                    "No JSON. No Markdown. No explanations.\n\n"
                    "MISSING CELLS:\n"
                    + retry_block
                )

                retry_content = self._request(retry_prompt)
                retry_translated = _parse_lines(
                    retry_content,
                    missing,
                )

                translated.update(retry_translated)

        return translated

    def translate_cells(
        self,
        cells: Iterable[tuple[str, str, str]],
    ) -> dict[tuple[str, str], str]:
        """
        Input:
            (sheet_name, coordinate, source_text)

        Output:
            {(sheet_name, coordinate): translation}
        """
        grouped: dict[str, list[tuple[str, str]]] = {}

        for sheet, coordinate, text in cells:
            grouped.setdefault(sheet, []).append(
                (coordinate, text)
            )

        output: dict[tuple[str, str], str] = {}

        for sheet, items in grouped.items():
            for chunk in _chunks(items, self.batch_size):
                translated = self._translate_chunk(
                    sheet,
                    chunk,
                )

                for coordinate, _ in chunk:
                    key = f"{sheet}!{coordinate}"
                    value = translated.get(key)

                    if value:
                        output[(sheet, coordinate)] = value

        return output
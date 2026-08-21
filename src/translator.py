import json
import time
from openai import OpenAI, RateLimitError, APIStatusError

from .cache import TranslationCache
from .models import TextUnit


class OpenRouterTranslator:
    """Text translator and multimodal OCR/translator for OpenRouter."""

    def __init__(
        self,
        api_key,
        model,
        source_language,
        target_language,
        cache=None,
        max_retries=5,
        batch_size=4,
        http_referer="",
        app_name="PDF Translator Studio",
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Put it in your .env file.")
        self.headers = {}
        if http_referer:
            self.headers["HTTP-Referer"] = http_referer
        if app_name:
            self.headers["X-Title"] = app_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=self.headers,
            max_retries=0,
        )
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.cache = cache
        self.max_retries = max_retries
        self.batch_size = batch_size

    def _prompt(self, units: list[TextUnit]) -> str:
        payload = [{"id": u.unit_id, "kind": u.kind, "text": u.text} for u in units]
        expected_ids = [u.unit_id for u in units]
        return f"""
Translate the following document segments from {self.source_language} to {self.target_language}.

The translated text will be placed back into the EXACT visual regions of the source PDF.

Rules:
- Translate faithfully. Never summarize, omit, invent, or explain.
- Preserve names, company names, identifiers, stock codes, dates, phone numbers, quantities,
  currencies, percentages, URLs and numbers exactly unless only surrounding words change.
- Keep headings and labels concise.
- Preserve table semantics: labels remain labels, values remain values, and separate rows stay separate.
- Do not move information between table cells.
- Do not add translator notes.
- Return ONLY valid JSON in this shape:
  {{"translations": [{{"id": "segment_id", "translation": "translated text"}}]}}
- Return EXACTLY one translation object for every input segment.
- Never skip a segment, even if it is short, numeric, a label, or punctuation-heavy.
- Copy every input id character-for-character into the output id.
- Required output ids for this request: {json.dumps(expected_ids, ensure_ascii=False)}

Segments:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    def _call_text(self, units):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional document translator. Return the requested JSON only.",
                        },
                        {"role": "user", "content": self._prompt(units)},
                    ],
                    temperature=0,
                )
            except RateLimitError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(30, 2 ** attempt))
            except APIStatusError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(30, 2 ** attempt))
                    continue
                raise
        raise RuntimeError(
            f"OpenRouter model '{self.model}' could not complete the request after "
            f"{self.max_retries + 1} attempts. Try another model or retry later."
        ) from last_error

    @staticmethod
    def _message_content(message):
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

    @staticmethod
    def _parse_json_content(content):
        content = (content or "").strip()
        if content.startswith("```"):
            parts = content.split("\n", 1)
            content = parts[1] if len(parts) == 2 else content
            if content.endswith("```"):
                content = content[:-3].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start:end + 1])
            raise

    @staticmethod
    def _normalize_id(value):
        return str(value or "").strip().strip('`').strip()

    def _parse_translations(self, response, pending):
        message = response.choices[0].message
        content = self._message_content(message)
        if not content:
            raise RuntimeError("OpenRouter returned an empty translation response.")
        data = self._parse_json_content(content)
        translations = data.get("translations", []) if isinstance(data, dict) else []
        pending_map = {u.unit_id: u for u in pending}
        result = {}
        for item in translations if isinstance(translations, list) else []:
            if not isinstance(item, dict):
                continue
            unit_id = self._normalize_id(item.get("id") or item.get("unit_id") or item.get("segment_id"))
            if unit_id not in pending_map:
                continue
            translation = str(item.get("translation") or item.get("translated_text") or "").strip()
            if translation:
                result[unit_id] = translation
        return result

    def _store(self, unit, translation, result, use_cache):
        result[unit.unit_id] = translation
        if self.cache and use_cache:
            self.cache.put(
                unit.text,
                self.source_language,
                self.target_language,
                self.model,
                translation,
            )

    def _translate_pending_batch(self, pending, result, use_cache, depth=0):
        response = self._call_text(pending)
        parsed = self._parse_translations(response, pending)
        for unit_id, translation in parsed.items():
            self._store({u.unit_id: u for u in pending}[unit_id], translation, result, use_cache)

        missing = [u for u in pending if u.unit_id not in parsed]
        if not missing:
            return

        # Models occasionally drop a short label/number when asked for a batch.
        # Recover by splitting only the missing work, rather than failing the whole PDF.
        if len(missing) > 1 and depth < 3:
            mid = max(1, len(missing) // 2)
            self._translate_pending_batch(missing[:mid], result, use_cache, depth + 1)
            self._translate_pending_batch(missing[mid:], result, use_cache, depth + 1)
            return

        # Final singleton recovery with a stricter prompt.
        unit = missing[0]
        if depth < 4:
            response = self._call_text([unit])
            parsed = self._parse_translations(response, [unit])
            translation = parsed.get(unit.unit_id)
            if translation:
                self._store(unit, translation, result, use_cache)
                return

        raise RuntimeError(
            "OpenRouter did not return a translation for segment "
            f"{unit.unit_id!r}. The request was retried with smaller batches. "
            "Try a more capable model or disable any model-side response formatting that modifies IDs."
        )

    def translate_batch(self, units: list[TextUnit], use_cache=True) -> dict[str, str]:
        result = {}
        pending = []
        for unit in units:
            cached = None
            if use_cache and self.cache:
                cached = self.cache.get(
                    unit.text,
                    self.source_language,
                    self.target_language,
                    self.model,
                )
            if cached:
                result[unit.unit_id] = cached
            else:
                pending.append(unit)

        for i in range(0, len(pending), self.batch_size):
            self._translate_pending_batch(pending[i:i + self.batch_size], result, use_cache)
        return result


import json

from src.image_translator import ImageTextRegion
from src.translator import OpenRouterTranslator


def test_image_region_maps_pixels_to_pdf_coordinates():
    region = ImageTextRegion(
        region_id="r1",
        page_index=0,
        xref=10,
        occurrence_index=0,
        image_rect=(100, 200, 300, 400),
        image_width=1000,
        image_height=1000,
        bbox_px=(100, 200, 600, 700),
        source_text="元画像",
        translation="Translated",
    )
    assert region.page_bbox == (120.0, 240.0, 220.0, 340.0)


def test_json_parser_accepts_code_fence():
    payload = '{"translations":[{"id":"p1_t001","translation":"English"}]}'
    parsed = OpenRouterTranslator._parse_json_content("```json\n" + payload + "\n```")
    assert parsed["translations"][0]["translation"] == "English"

class _FakeResponse:
    def __init__(self, content):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})()]


def test_missing_batch_ids_are_recovered_by_smaller_requests():
    units = [
        type('U', (), {'unit_id': 'p1_u001', 'kind': 'paragraph', 'text': 'こんにちは'})(),
        type('U', (), {'unit_id': 'p1_u002', 'kind': 'paragraph', 'text': '世界'})(),
        type('U', (), {'unit_id': 'p1_u003', 'kind': 'label', 'text': '東京'})(),
    ]
    translator = OpenRouterTranslator.__new__(OpenRouterTranslator)
    translator.source_language = 'Japanese'
    translator.target_language = 'English'
    translator.model = 'test/model'
    translator.cache = None
    translator.max_retries = 0
    translator.batch_size = 4
    calls = []

    def fake_call(batch):
        calls.append([u.unit_id for u in batch])
        if len(batch) == 3:
            return _FakeResponse(json.dumps({
                'translations': [
                    {'id': 'p1_u001', 'translation': 'Hello'},
                ]
            }))
        return _FakeResponse(json.dumps({
            'translations': [
                {'id': batch[0].unit_id, 'translation': 'World' if batch[0].unit_id == 'p1_u002' else 'Tokyo'}
            ]
        }))

    translator._call_text = fake_call
    result = translator.translate_batch(units, use_cache=False)
    assert result == {'p1_u001': 'Hello', 'p1_u002': 'World', 'p1_u003': 'Tokyo'}
    assert calls[0] == ['p1_u001', 'p1_u002', 'p1_u003']
    assert len(calls) >= 2


def test_message_content_list_is_supported():
    message = type('Message', (), {
        'content': [
            {'type': 'text', 'text': '{"translations": []}'},
        ]
    })()
    assert OpenRouterTranslator._message_content(message) == '{"translations": []}'


def test_vision_message_content_list_helper():
    from src.image_translator import _message_content
    message = type('Message', (), {'content': [{'type': 'text', 'text': '{"regions": []}'}]})()
    assert _message_content(message) == '{"regions": []}'

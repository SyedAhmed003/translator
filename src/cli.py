import argparse
import json
import os
from pathlib import Path

from .pipeline import inspect_document, translate_document_with_options


def build_parser():
    parser = argparse.ArgumentParser(description="Layout-preserving PDF translator using OpenRouter")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect", help="Inspect PDF structure without calling OpenRouter.")
    inspect_cmd.add_argument("input", help="Input PDF")
    inspect_cmd.add_argument("--report", default=None, help="Optional JSON report path")
    inspect_cmd.add_argument("--source", default="Japanese", help="Source language")

    translate_cmd = sub.add_parser("translate", help="Translate PDF with OpenRouter.")
    translate_cmd.add_argument("input", help="Input PDF")
    translate_cmd.add_argument("--output", default=None, help="Output PDF path")
    translate_cmd.add_argument("--model", required=True, help="OpenRouter model ID")
    translate_cmd.add_argument("--source", default="Japanese", help="Source language")
    translate_cmd.add_argument("--target", default="English", help="Target language")
    translate_cmd.add_argument("--no-cache", action="store_true", help="Ignore existing text translation cache")
    translate_cmd.add_argument("--no-image-vision", action="store_true", help="Disable OpenRouter image OCR/translation")
    translate_cmd.add_argument("--max-image-edge", type=int, default=2400, help="Maximum image edge sent to the vision model")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        report = args.report or str(Path("output") / f"{Path(args.input).stem}.inspection.json")
        result = inspect_document(args.input, report, source_language=args.source)
        print(json.dumps(result["analysis"], ensure_ascii=False, indent=2))
        print(f"\nInspection report: {report}")
        return

    output = args.output or str(Path("output") / f"{Path(args.input).stem}_en.pdf")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    result = translate_document_with_options(
        args.input,
        output,
        api_key=api_key,
        model=args.model,
        source_language=args.source,
        target_language=args.target,
        use_cache=not args.no_cache,
        use_vision_images=not args.no_image_vision,
        max_image_edge=args.max_image_edge,
    )
    print(f"Created: {result['output']}")
    print(f"Report:  {result['report']}")
    print(f"Text units: {result['unit_count']}")
    print(f"Image text regions: {result['image_text_region_count']}")
    print(f"Warnings: {result['warning_count']}")


if __name__ == "__main__":
    main()

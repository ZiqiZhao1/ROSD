#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


"""
python scripts/judge_spurious_context_attribution.py \
  --input-jsonl checkpoints/qwen3-4b/biology/validation_data-skip-qwen3-4b-biology/210.jsonl \
  --limit 800

python scripts/judge_spurious_context_attribution.py \
  --input-jsonl checkpoints/qwen3-4b/biology/validation_data-qwen3-4b-biology/210.jsonl \
  --limit 800
"""

# Fill these in once if you want the script to run without environment variables.
DEFAULT_API_KEY = "sk-rJoHNTrvVve6KIF6tktBXtlrU8R5V8oLOmqPo4OTIAu4pTdg"
DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"
DEFAULT_JUDGE_MODEL = "gpt-5.4-nano-2026-03-17"


PROMPT_TEMPLATE = """You are detecting spurious context attribution.

The user prompt did not contain any passage, article, document, example, or external context.

Given the model answer below, decide whether the answer falsely implies that it is based on a provided text, passage, document, example, or context.

Return JSON only:
{{
  "label": 0 or 1,
  "confidence": 0.0 to 1.0
}}

Label 1 if the answer says or implies things like:
- based on the text
- according to the passage
- the provided information suggests
- from the example
- 根据材料
- 根据上文
- 由所给内容可知

Label 0 if the answer simply answers the question without such attribution.

Model answer:
{answer}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge spurious context attribution in validation JSONL outputs."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input JSONL file containing model outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Judge only the first N lines. Use -1 for all lines.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path to save per-sample judgments. Defaults next to input.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to save aggregate statistics. Defaults next to input.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="output",
        help="Field in each JSONL row to judge. Defaults to 'output'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model name. Defaults to the in-file DEFAULT_JUDGE_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Optional OpenAI-compatible base URL. Defaults to the in-file DEFAULT_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key. Defaults to the in-file DEFAULT_API_KEY.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for each sample.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between retries.",
    )
    return parser.parse_args()


def default_output_paths(input_jsonl: Path, limit: int) -> Tuple[Path, Path]:
    stem = input_jsonl.stem
    suffix = f"first-{limit}" if limit >= 0 else "all"
    out_jsonl = input_jsonl.with_name(f"{stem}.spurious-context.{suffix}.jsonl")
    summary_json = input_jsonl.with_name(f"{stem}.spurious-context.{suffix}.summary.json")
    return out_jsonl, summary_json


def build_request_config(api_key: Optional[str], base_url: Optional[str]) -> Dict[str, str]:
    if not api_key:
        raise ValueError("Missing API key. Fill DEFAULT_API_KEY in the script or pass --api-key.")
    if not base_url:
        raise ValueError("Missing base URL. Fill DEFAULT_BASE_URL in the script or pass --base-url.")
    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
    }


def extract_json_dict(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in response: {text[:300]}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object, got: {type(value)}")
    return value


def normalize_judgment(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "label" not in raw or "confidence" not in raw:
        raise ValueError(f"Judge response missing keys: {raw}")

    label = int(raw["label"])
    if label not in (0, 1):
        raise ValueError(f"Invalid label: {label}")

    confidence = float(raw["confidence"])
    confidence = max(0.0, min(1.0, confidence))

    return {"label": label, "confidence": confidence}


def read_rows(input_jsonl: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if limit >= 0 and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_line"] = line_idx
            rows.append(row)
    return rows


def judge_answer(
    request_config: Dict[str, str],
    model: str,
    answer: str,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(answer=answer)
    last_error: Optional[Exception] = None
    url = f"{request_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {request_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"] or ""
            return normalize_judgment(extract_json_dict(content))
        except Exception as exc:
            response_text = ""
            if "response" in locals():
                try:
                    response_text = response.text[:1000]
                except Exception:
                    response_text = ""
            last_error = RuntimeError(f"{type(exc).__name__}: {exc}. response_text={response_text}")
            if attempt == max_retries:
                break
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Judge request failed after {max_retries} attempts: {last_error}")


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    successful = [r for r in results if r.get("judge_ok")]
    failed = [r for r in results if not r.get("judge_ok")]
    label_1 = [r for r in successful if r["judge"]["label"] == 1]
    label_0 = [r for r in successful if r["judge"]["label"] == 0]

    def mean_conf(items: List[Dict[str, Any]]) -> Optional[float]:
        if not items:
            return None
        return sum(item["judge"]["confidence"] for item in items) / len(items)

    return {
        "total_rows_read": total,
        "successful_judgments": len(successful),
        "failed_judgments": len(failed),
        "label_1_count": len(label_1),
        "label_0_count": len(label_0),
        "label_1_rate_among_successful": (len(label_1) / len(successful)) if successful else None,
        "mean_confidence_among_successful": mean_conf(successful),
        "mean_confidence_label_1": mean_conf(label_1),
        "mean_confidence_label_0": mean_conf(label_0),
        "failed_source_lines": [r.get("_source_line") for r in failed],
    }


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    output_jsonl, summary_json = default_output_paths(args.input_jsonl, args.limit)
    if args.output_jsonl is not None:
        output_jsonl = args.output_jsonl
    if args.summary_json is not None:
        summary_json = args.summary_json

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    request_config = build_request_config(api_key=args.api_key, base_url=args.base_url)
    rows = read_rows(args.input_jsonl, args.limit)

    results: List[Dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as fout:
        progress = tqdm(rows, desc="Judging rows", unit="row")
        for row in progress:
            answer = row.get(args.field, "")
            result = dict(row)
            result["judge_model"] = args.model
            result["judge_field"] = args.field

            if not isinstance(answer, str):
                result["judge_ok"] = False
                result["judge_error"] = f"Field {args.field!r} is not a string."
            elif not answer.strip():
                result["judge_ok"] = False
                result["judge_error"] = f"Field {args.field!r} is empty."
            else:
                try:
                    judgment = judge_answer(
                        request_config=request_config,
                        model=args.model,
                        answer=answer,
                        max_retries=args.max_retries,
                        sleep_seconds=args.sleep_seconds,
                    )
                    result["judge_ok"] = True
                    result["judge"] = judgment
                except Exception as exc:
                    result["judge_ok"] = False
                    result["judge_error"] = str(exc)

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
            success_count = sum(1 for item in results if item.get("judge_ok"))
            label_1_count = sum(
                1 for item in results if item.get("judge_ok") and item["judge"]["label"] == 1
            )
            progress.set_postfix(
                success=success_count,
                label1=label_1_count,
                failed=len(results) - success_count,
            )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "model": args.model,
        "field": args.field,
        "limit": args.limit,
        **summarize(results),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

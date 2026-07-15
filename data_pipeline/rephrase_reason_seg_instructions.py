"""Generate semantically equivalent ReasonSeg instruction variants with OpenAI.

This is a text-only data operation. It does not load LISA, SAM, CLIP, or any
model weights. By default it reads ReasonSegRelabel/train JSON annotations and
writes reviewable proposals to a JSONL manifest without changing annotations.

Typical workflow:

  python data_pipeline/rephrase_reason_seg_instructions.py --dry-run
  python data_pipeline/rephrase_reason_seg_instructions.py --limit 10
  python data_pipeline/rephrase_reason_seg_instructions.py --apply

The first networked run requires OPENAI_API_KEY. The second command above only
generates/reuses the manifest; --apply reuses accepted cached proposals and
updates each annotation's ``text`` list. Never commit an API key.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = "dataset/reason_seg/ReasonSegRelabel/train"
DEFAULT_MANIFEST = "dataset/reason_seg/ReasonSegRelabel/instruction_rewrites.jsonl"
DEFAULT_SUMMARY = "dataset/reason_seg/ReasonSegRelabel/instruction_rewrites_summary.json"
DEFAULT_MODEL = "gpt-3.5-turbo"

CATEGORY_GUIDANCE = {
    "equipment_proximity": "设备与人员距离过近、存在碰撞或接触风险的目标区域",
    "guardrail_missing": "缺少防护栏杆且存在坠落风险的临边区域",
    "harness_missing": "高处作业中未按要求佩戴安全带的人员",
    "helmet_missing": "未佩戴安全帽的人员",
    "no helmet": "未佩戴安全帽的人员",
    "no_helmet": "未佩戴安全帽的人员",
    "no jacket": "未穿反光衣或安全背心的人员",
    "no_jacket": "未穿反光衣或安全背心的人员",
    "opening_unprotected": "未采取防护措施的洞口或开口区域",
    "poor_housekeeping": "材料堆放混乱或影响安全通行的区域",
    "safe": "原指令所指的安全状态目标；不得添加原指令没有说明的具体安全属性",
    "unsafe": "原指令所指的不安全状态目标；不得擅自推断或添加具体隐患类型",
}

SYSTEM_INSTRUCTIONS = """你是施工安全语义分割数据集的标注专家。你的任务是改写训练指令，而不是扩写、解释或改变标签。

所有改写必须满足：
1. 与原指令严格语义等价，答案必须对应完全相同的一组像素和实例。
2. 不增加或删除人员、设备、位置、风险类型、属性、数量或空间关系。
3. 保持原指令的分割粒度；不得把人员改成区域、把目标改成整幅场景。
4. 不使用含义不确定的代词，不输出类别代码，不解释改写过程。
5. 使用自然、简洁、适合中文施工安全场景的表达，并适当变化句式和动词。
6. 只返回要求的 JSON 对象。"""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalized_key(value: str) -> str:
    value = normalize_text(value).lower()
    return re.sub(r"[\s，。！？、；：,.!?;:]", "", value)


def load_annotation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = data.get("text")
    if not isinstance(texts, list) or not texts or not all(isinstance(x, str) for x in texts):
        raise ValueError(f"Annotation must contain a non-empty string list in text: {path}")
    if not normalize_text(texts[0]):
        raise ValueError(f"Canonical text is empty: {path}")
    return data


def annotation_category(data: dict[str, Any]) -> str:
    source = data.get("source")
    if isinstance(source, dict):
        value = source.get("source_category") or source.get("sample_key")
        if isinstance(value, str):
            return value
    return "unknown"


def validate_rewrites(canonical: str, rewrites: Any, count: int) -> list[str]:
    if not isinstance(rewrites, list):
        raise ValueError("Response field rewrites must be a list")
    if len(rewrites) != count:
        raise ValueError(f"Expected {count} rewrites, got {len(rewrites)}")

    canonical_key = normalized_key(canonical)
    seen = {canonical_key}
    cleaned = []
    for index, value in enumerate(rewrites, start=1):
        if not isinstance(value, str):
            raise ValueError(f"Rewrite {index} is not a string")
        text = normalize_text(value)
        if len(text) < 6 or len(text) > 100:
            raise ValueError(f"Rewrite {index} has unreasonable length: {len(text)}")
        key = normalized_key(text)
        if not key or key in seen:
            raise ValueError(f"Rewrite {index} duplicates the canonical text or another rewrite")
        seen.add(key)
        cleaned.append(text)
    return cleaned


def user_prompt(canonical: str, category: str, count: int) -> str:
    guidance = CATEGORY_GUIDANCE.get(category, "严格保持原指令所指的目标、实例范围和分割粒度")
    return (
        f"请为下面的训练指令生成 {count} 条严格等价的中文改写。\n"
        f"数据集类别：{category}\n"
        f"该类别的语义约束：{guidance}\n"
        f"原始标准指令：{canonical}\n\n"
        f'返回 JSON：{{"rewrites": ["改写1", "改写2"]}}。'
        f"rewrites 数组必须恰好包含 {count} 条。"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response does not contain a JSON object")
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    return parsed


def response_output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("Responses API result has no output_text")


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        request_id = exc.headers.get("x-request-id", "unknown")
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"OpenAI API HTTP {exc.code}, request_id={request_id}: {body}"
        ) from exc


def request_gpt35_rewrites(
    base_url: str,
    api_key: str,
    model: str,
    canonical: str,
    category: str,
    count: int,
    timeout: float,
) -> list[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt(canonical, category, count)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.8,
    }
    response = post_json(f"{base_url}/chat/completions", payload, api_key, timeout)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Chat Completions result has no assistant content") from exc
    parsed = extract_json_object(content)
    return validate_rewrites(canonical, parsed.get("rewrites"), count)


def request_responses_rewrites(
    base_url: str,
    api_key: str,
    model: str,
    canonical: str,
    category: str,
    count: int,
    timeout: float,
) -> list[str]:
    payload = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": user_prompt(canonical, category, count),
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "instruction_rewrites",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "rewrites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": count,
                            "maxItems": count,
                        }
                    },
                    "required": ["rewrites"],
                    "additionalProperties": False,
                },
            }
        },
    }
    response = post_json(f"{base_url}/responses", payload, api_key, timeout)
    parsed = extract_json_object(response_output_text(response))
    return validate_rewrites(canonical, parsed.get("rewrites"), count)


def request_rewrites(
    base_url: str,
    api_key: str,
    model: str,
    canonical: str,
    category: str,
    count: int,
    timeout: float,
    max_retries: int,
) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if model.startswith("gpt-3.5"):
                return request_gpt35_rewrites(
                    base_url, api_key, model, canonical, category, count, timeout
                )
            return request_responses_rewrites(
                base_url, api_key, model, canonical, category, count, timeout
            )
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = min(2**attempt, 8)
            print(f"[retry] {attempt + 1}/{max_retries}: {exc}; wait {delay}s", file=sys.stderr)
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            json_path = item.get("json_path")
            if isinstance(json_path, str):
                latest[json_path] = item
    return latest


def append_manifest(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def cached_rewrites(
    item: dict[str, Any] | None, canonical: str, count: int
) -> list[str] | None:
    if not item or item.get("status") != "ok" or item.get("canonical") != canonical:
        return None
    try:
        return validate_rewrites(canonical, item.get("rewrites"), count)
    except ValueError:
        return None


def write_annotation(path: Path, data: dict[str, Any], texts: list[str]) -> None:
    data["text"] = texts
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def balanced_limit(
    plans: list[tuple[Path, str, dict[str, Any], str, str]], limit: int, seed: int
) -> list[tuple[Path, str, dict[str, Any], str, str]]:
    """Select a deterministic, category-round-robin pilot batch."""
    rng = random.Random(seed)
    by_category: dict[str, list[tuple[Path, str, dict[str, Any], str, str]]] = {}
    for plan in plans:
        by_category.setdefault(plan[4], []).append(plan)
    for items in by_category.values():
        rng.shuffle(items)

    selected = []
    categories = sorted(by_category)
    while len(selected) < limit:
        added = False
        for category in categories:
            items = by_category[category]
            if items:
                selected.append(items.pop())
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate equivalent ReasonSeg instruction variants with OpenAI."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, type=Path)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_REPHRASE_MODEL", DEFAULT_MODEL),
        help="OpenAI model. gpt-3.5-turbo uses Chat Completions; newer models use Responses.",
    )
    parser.add_argument(
        "--variants", default=4, type=int, help="New variants per canonical instruction."
    )
    parser.add_argument("--limit", type=int, help="Process at most this many JSON files.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly allow API generation for every JSON file in the input directory.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed for reproducible category-balanced --limit."
    )
    parser.add_argument("--timeout", default=60.0, type=float)
    parser.add_argument("--max-retries", default=2, type=int)
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update annotations from valid cached proposals only; never call the API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore valid cached proposals and call the API again.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs and print the plan; no API or writes."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variants < 1 or args.variants > 5:
        raise ValueError("--variants must be between 1 and 5")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.limit is not None and args.all:
        raise ValueError("Use either --limit or --all, not both")
    if args.apply and args.force:
        raise ValueError("--apply cannot be combined with --force")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")

    root = repo_root()
    input_dir = resolve_path(args.input_dir, root)
    manifest_path = resolve_path(args.manifest, root)
    summary_path = resolve_path(args.summary, root)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_paths = sorted(input_dir.glob("*.json"))
    cache = load_manifest(manifest_path)

    plans = []
    for path in json_paths:
        data = load_annotation(path)
        canonical = normalize_text(data["text"][0])
        category = annotation_category(data)
        relative_path = path.relative_to(root).as_posix()
        plans.append((path, relative_path, data, canonical, category))
    if args.limit is not None:
        plans = balanced_limit(plans, args.limit, args.seed)

    category_counts = Counter()
    for _, _, _, _, category in plans:
        category_counts[category] += 1

    print(f"[plan] input: {input_dir}")
    print(f"[plan] annotations: {len(plans)}")
    print(f"[plan] variants per annotation: {args.variants} (+ canonical = {args.variants + 1})")
    print(f"[plan] model: {args.model}")
    print(f"[plan] categories: {dict(sorted(category_counts.items()))}")
    if args.model.startswith("gpt-3.5"):
        print(
            "[warning] GPT-3.5 Turbo is deprecated in the current OpenAI model catalog; "
            "use --model to select a supported newer model when appropriate.",
            file=sys.stderr,
        )
    if args.dry_run:
        for _, relative_path, _, canonical, category in plans[:5]:
            print(f"[sample] {relative_path} | {category} | {canonical}")
        print("[dry-run] no API calls or files were changed")
        return

    if not args.apply and args.limit is None and not args.all:
        raise RuntimeError("Refusing an implicit full API run; pass --limit N or --all")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.apply and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for instruction generation")

    stats = Counter()
    for index, (path, relative_path, data, canonical, category) in enumerate(plans, start=1):
        rewrites = None if args.force else cached_rewrites(cache.get(relative_path), canonical, args.variants)
        if rewrites is not None:
            stats["cached"] += 1
        elif args.apply:
            stats["missing_cached_proposal"] += 1
            print(f"[skip] {relative_path}: no valid cached proposal", file=sys.stderr)
            continue
        else:
            try:
                rewrites = request_rewrites(
                    args.base_url.rstrip("/"),
                    api_key,
                    args.model,
                    canonical,
                    category,
                    args.variants,
                    args.timeout,
                    args.max_retries,
                )
                item = {
                    "status": "ok",
                    "json_path": relative_path,
                    "category": category,
                    "model": args.model,
                    "canonical": canonical,
                    "original_text": data["text"],
                    "rewrites": rewrites,
                }
                append_manifest(manifest_path, item)
                cache[relative_path] = item
                stats["generated"] += 1
            except Exception as exc:  # keep batch progress and a reviewable failure record
                item = {
                    "status": "error",
                    "json_path": relative_path,
                    "category": category,
                    "model": args.model,
                    "canonical": canonical,
                    "error": str(exc),
                }
                append_manifest(manifest_path, item)
                cache[relative_path] = item
                stats["failed"] += 1
                print(f"[error] {relative_path}: {exc}", file=sys.stderr)
                continue

        if args.apply:
            write_annotation(path, data, [canonical, *rewrites])
            stats["applied"] += 1
        print(f"[{index}/{len(plans)}] {relative_path}: ok")

    summary = {
        "input_dir": input_dir.relative_to(root).as_posix(),
        "manifest": manifest_path.relative_to(root).as_posix(),
        "model": args.model,
        "variants": args.variants,
        "apply": args.apply,
        "selected_annotations": len(plans),
        "category_counts": dict(sorted(category_counts.items())),
        "stats": dict(stats),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] stats: {dict(stats)}")
    if not args.apply:
        print("[next] review the manifest, then rerun the same selection with --apply")


if __name__ == "__main__":
    main()

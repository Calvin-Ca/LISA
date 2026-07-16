from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = {"config.json", "tokenizer_config.json"}
WEIGHT_SUFFIXES = {".bin", ".safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_source(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"model source directory does not exist: {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    names = {path.name for path in files}
    missing = sorted(REQUIRED_FILES - names)
    if missing:
        raise ValueError(f"model source is missing required files: {missing}")
    if not any(path.suffix in WEIGHT_SUFFIXES for path in files):
        raise ValueError("model source contains no .bin or .safetensors weights")
    return files


def build_manifest(
    *,
    repo_root: Path,
    source: Path,
    version: str,
    files: list[Path],
) -> dict:
    entries = []
    for path in files:
        entries.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "source": source.as_posix(),
        "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }


def write_model_card(path: Path, manifest: dict) -> None:
    content = f"""# {manifest['model_version']}

## Purpose

LISA construction-safety reasoning segmentation production artifact.

## Provenance

- Git commit: `{manifest['git_commit']}`
- Source: `{manifest['source']}`
- Created at: `{manifest['created_at']}`
- Files: `{manifest['file_count']}`
- Total bytes: `{manifest['total_size_bytes']}`

## Runtime

- Repository inference entry: `production.app:app`
- Default precision: bf16
- Required external vision tower: `clip-vit-large-patch14`

## Validation

Before promotion, run the production-precision benchmark, independent golden
test, smoke test, load test, and rollback drill described in
`production/todo.md`.
"""
    path.write_text(content, encoding="utf-8")


def freeze(
    *,
    repo_root: Path,
    source: Path,
    output_root: Path,
    version: str,
    copy_model: bool,
) -> Path:
    files = validate_source(source)
    destination = output_root / version
    if destination.exists():
        raise ValueError(f"artifact destination already exists: {destination}")
    destination.mkdir(parents=True)

    if copy_model:
        shutil.copytree(source, destination / "merged_hf")

    manifest = build_manifest(
        repo_root=repo_root,
        source=source,
        version=version,
        files=files,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (destination / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for item in manifest["files"]:
            handle.write(f"{item['sha256']}  merged_hf/{item['path']}\n")
    write_model_card(destination / "MODEL_CARD.md", manifest)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a merged LISA model as a versioned artifact."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/lisa-safety-seg"),
    )
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Do not copy model files; generate metadata and checksums only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    destination = freeze(
        repo_root=repo_root,
        source=args.source.resolve(),
        output_root=args.output_root.resolve(),
        version=args.version,
        copy_model=not args.manifest_only,
    )
    print(f"Frozen artifact: {destination}")


if __name__ == "__main__":
    main()


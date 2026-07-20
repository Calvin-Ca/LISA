from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_VERSION_LABEL = "io.lisa.model.version"
SOURCE_COMMIT_LABEL = "org.opencontainers.image.revision"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def parse_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        parts = raw_line.split(None, 1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
            raise ValueError(
                f"invalid SHA256SUMS line {line_number}: {raw_line!r}"
            )
        relative = safe_relative_path(parts[1].strip()).as_posix()
        if relative in checksums:
            raise ValueError(f"duplicate SHA256SUMS path: {relative}")
        checksums[relative] = parts[0]
    if not checksums:
        raise ValueError("SHA256SUMS is empty")
    return checksums


def verify_model_artifact(artifact: Path) -> dict[str, Any]:
    artifact = artifact.resolve()
    manifest_path = artifact / "manifest.json"
    sums_path = artifact / "SHA256SUMS"
    model_card_path = artifact / "MODEL_CARD.md"
    model_dir = artifact / "merged_hf"
    for path, description in (
        (manifest_path, "manifest.json"),
        (sums_path, "SHA256SUMS"),
        (model_card_path, "MODEL_CARD.md"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {description}: {path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"missing merged_hf directory: {model_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("model manifest must be a JSON object")
    model_version = manifest.get("model_version")
    entries = manifest.get("files")
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("model manifest has no model_version")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model manifest has no files")

    expected: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("model manifest file entry is not an object")
        relative = safe_relative_path(str(item.get("path", ""))).as_posix()
        checksum = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if relative in expected:
            raise ValueError(f"duplicate model manifest path: {relative}")
        if not isinstance(checksum, str) or not SHA256_RE.fullmatch(checksum):
            raise ValueError(f"invalid manifest SHA-256 for: {relative}")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"invalid manifest size for: {relative}")
        expected[relative] = {
            "sha256": checksum,
            "size_bytes": size_bytes,
        }

    checksums = parse_sha256sums(sums_path)
    expected_sum_paths = {
        f"merged_hf/{relative}" for relative in expected
    }
    if set(checksums) != expected_sum_paths:
        missing = sorted(expected_sum_paths - set(checksums))
        extra = sorted(set(checksums) - expected_sum_paths)
        raise ValueError(
            f"SHA256SUMS paths differ from manifest: "
            f"missing={missing}, extra={extra}"
        )

    actual_paths = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise ValueError(
            f"merged_hf files differ from manifest: "
            f"missing={missing}, extra={extra}"
        )

    total_size_bytes = 0
    for relative, metadata in expected.items():
        model_path = model_dir / relative
        actual_size = model_path.stat().st_size
        if actual_size != metadata["size_bytes"]:
            raise ValueError(
                f"model size mismatch for {relative}: "
                f"{actual_size} != {metadata['size_bytes']}"
            )
        actual_checksum = sha256(model_path)
        if actual_checksum != metadata["sha256"]:
            raise ValueError(f"model SHA-256 mismatch for: {relative}")
        if checksums[f"merged_hf/{relative}"] != actual_checksum:
            raise ValueError(
                f"SHA256SUMS checksum mismatch for: {relative}"
            )
        total_size_bytes += actual_size

    if manifest.get("file_count") != len(expected):
        raise ValueError("model manifest file_count is inconsistent")
    if manifest.get("total_size_bytes") != total_size_bytes:
        raise ValueError("model manifest total_size_bytes is inconsistent")

    return {
        "model_version": model_version,
        "file_count": len(expected),
        "total_size_bytes": total_size_bytes,
        "manifest_sha256": sha256(manifest_path),
        "sha256sums_sha256": sha256(sums_path),
        "model_card_sha256": sha256(model_card_path),
    }


def inspect_docker_image(image_ref: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("docker image inspect returned an invalid payload")
    if not isinstance(payload[0], dict):
        raise ValueError("docker image inspect entry is not an object")
    return payload[0]


def portable_repo_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"validation evidence must be inside the repository: {path}"
        ) from exc


def build_release_manifest(
    *,
    repo_root: Path,
    artifact_metadata: dict[str, Any],
    image_ref: str,
    image_metadata: dict[str, Any],
    container_summary: dict[str, Any],
    validation_summaries: list[Path],
    require_image_digest: bool,
) -> dict[str, Any]:
    config = image_metadata.get("Config") or {}
    labels = config.get("Labels") or {}
    model_version = artifact_metadata["model_version"]
    acceptance = container_summary.get("acceptance") or {}
    if acceptance.get("passed") is not True:
        raise ValueError("container validation summary did not pass")
    if container_summary.get("model_version") != model_version:
        raise ValueError(
            "container validation model version differs from artifact"
        )
    expected_git_commit = container_summary.get("repo_git_commit")
    if not isinstance(expected_git_commit, str) or not expected_git_commit:
        raise ValueError("container validation has no Git commit")
    verified_image = container_summary.get("image") or {}
    verified_image_id = str(verified_image.get("id", "")).removeprefix(
        "sha256:"
    )
    actual_image_id = str(image_metadata.get("Id", "")).removeprefix(
        "sha256:"
    )
    if not verified_image_id or actual_image_id != verified_image_id:
        raise ValueError(
            f"image ID differs from validated image: "
            f"{actual_image_id!r} != {verified_image_id!r}"
        )

    model_label = labels.get(MODEL_VERSION_LABEL)
    commit_label = labels.get(SOURCE_COMMIT_LABEL)
    if model_label is not None and model_label != model_version:
        raise ValueError(
            f"image model label mismatch: "
            f"{model_label!r} != {model_version!r}"
        )
    if commit_label is not None and commit_label != expected_git_commit:
        raise ValueError(
            f"image source commit mismatch: "
            f"{commit_label!r} != {expected_git_commit!r}"
        )

    repo_digests = sorted(image_metadata.get("RepoDigests") or [])
    if require_image_digest and not repo_digests:
        raise ValueError(
            "image has no registry digest; push it before release"
        )

    evidence = []
    for summary_path in validation_summaries:
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"missing validation summary: {summary_path}"
            )
        evidence.append(
            {
                "path": portable_repo_path(summary_path, repo_root),
                "sha256": sha256(summary_path),
            }
        )
    if not evidence:
        raise ValueError("at least one validation summary is required")

    return {
        "schema_version": 1,
        "release_version": (
            f"{model_version}-{expected_git_commit[:12]}"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": expected_git_commit,
        "model": artifact_metadata,
        "container": {
            "reference": image_ref,
            "image_id": image_metadata.get("Id"),
            "repo_digests": repo_digests,
            "created": image_metadata.get("Created"),
            "architecture": image_metadata.get("Architecture"),
            "os": image_metadata.get("Os"),
            "size_bytes": image_metadata.get("Size"),
            "labels": {
                MODEL_VERSION_LABEL: model_label,
                SOURCE_COMMIT_LABEL: commit_label,
            },
            "validated_image_id": verified_image_id,
        },
        "validation_evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and bind a frozen model to a pushed image."
    )
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument(
        "--container-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-summary",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-image-digest",
        action="store_true",
        help="Allow a local-only image that has not been pushed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_metadata = verify_model_artifact(args.model_artifact)
    image_metadata = inspect_docker_image(args.image_ref)
    container_summary = json.loads(
        args.container_summary.read_text(encoding="utf-8")
    )
    release_manifest = build_release_manifest(
        repo_root=repo_root,
        artifact_metadata=artifact_metadata,
        image_ref=args.image_ref,
        image_metadata=image_metadata,
        container_summary=container_summary,
        validation_summaries=args.validation_summary,
        require_image_digest=not args.allow_missing_image_digest,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Release manifest: {output}")
    print(
        f"Release version: {release_manifest['release_version']}"
    )


if __name__ == "__main__":
    main()

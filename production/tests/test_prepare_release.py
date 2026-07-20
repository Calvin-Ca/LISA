import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from production.freeze_model_artifact import freeze
from production.prepare_release import (
    build_release_manifest,
    parse_sha256sums,
    verify_model_artifact,
)


class PrepareReleaseTest(unittest.TestCase):
    def make_artifact(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "config.json").write_text("{}", encoding="utf-8")
        (source / "tokenizer_config.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (source / "model.bin").write_bytes(b"weights")
        with patch(
            "production.freeze_model_artifact.git_commit",
            return_value="a" * 40,
        ):
            return freeze(
                repo_root=root,
                source=source,
                output_root=root / "artifacts",
                version="test-v1",
                copy_model=True,
            )

    def test_verify_model_artifact_and_build_release_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = self.make_artifact(root)
            metadata = verify_model_artifact(artifact)
            summary = root / "summary.json"
            summary.write_text('{"passed": true}', encoding="utf-8")
            image_id = "b" * 64
            container_summary = {
                "model_version": "test-v1",
                "repo_git_commit": "a" * 40,
                "image": {"id": image_id},
                "acceptance": {"passed": True},
            }
            release = build_release_manifest(
                repo_root=root,
                artifact_metadata=metadata,
                image_ref="registry.example/test:test-v1",
                image_metadata={
                    "Id": f"sha256:{image_id}",
                    "RepoDigests": [
                        f"registry.example/test@sha256:{'c' * 64}"
                    ],
                    "Config": {"Labels": {}},
                    "Architecture": "amd64",
                    "Os": "linux",
                    "Size": 123,
                },
                container_summary=container_summary,
                validation_summaries=[summary],
                require_image_digest=True,
            )
            self.assertEqual(
                release["release_version"],
                f"test-v1-{'a' * 12}",
            )
            self.assertEqual(
                release["container"]["validated_image_id"],
                image_id,
            )
            self.assertEqual(release["model"]["file_count"], 3)

    def test_verify_model_artifact_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = self.make_artifact(root)
            (artifact / "merged_hf" / "model.bin").write_bytes(
                b"corrupt"
            )
            with self.assertRaises(ValueError):
                verify_model_artifact(artifact)

    def test_release_rejects_unvalidated_image_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = self.make_artifact(root)
            summary = root / "summary.json"
            summary.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_release_manifest(
                    repo_root=root,
                    artifact_metadata=verify_model_artifact(artifact),
                    image_ref="registry.example/test:test-v1",
                    image_metadata={
                        "Id": f"sha256:{'b' * 64}",
                        "RepoDigests": [
                            f"registry.example/test@sha256:{'c' * 64}"
                        ],
                        "Config": {"Labels": {}},
                    },
                    container_summary={
                        "model_version": "test-v1",
                        "repo_git_commit": "a" * 40,
                        "image": {"id": "d" * 64},
                        "acceptance": {"passed": True},
                    },
                    validation_summaries=[summary],
                    require_image_digest=True,
                )

    def test_sha256sums_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SHA256SUMS"
            path.write_text(
                f"{'a' * 64}  ../secret\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                parse_sha256sums(path)

    def test_publish_script_retags_validated_image_without_rebuild(self):
        repo_root = Path(__file__).resolve().parents[2]
        script = (
            repo_root / "production" / "publish_release.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("docker tag", script)
        self.assertIn("docker push", script)
        self.assertIn("rclone check", script)
        self.assertNotIn("docker build", script)


if __name__ == "__main__":
    unittest.main()

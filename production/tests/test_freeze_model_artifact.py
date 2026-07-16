import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from production.freeze_model_artifact import freeze, validate_source


class FreezeArtifactTest(unittest.TestCase):
    def test_validate_requires_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "config.json").write_text("{}")
            (source / "tokenizer_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                validate_source(source)

    def test_manifest_only_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "tokenizer_config.json").write_text("{}")
            (source / "model.bin").write_bytes(b"weights")
            output = root / "artifacts"

            with patch(
                "production.freeze_model_artifact.git_commit",
                return_value="deadbeef",
            ):
                destination = freeze(
                    repo_root=root,
                    source=source,
                    output_root=output,
                    version="test-v1",
                    copy_model=False,
                )

            manifest = json.loads(
                (destination / "manifest.json").read_text()
            )
            self.assertEqual(manifest["model_version"], "test-v1")
            self.assertEqual(manifest["git_commit"], "deadbeef")
            self.assertFalse((destination / "merged_hf").exists())
            self.assertTrue((destination / "SHA256SUMS").is_file())
            self.assertTrue((destination / "MODEL_CARD.md").is_file())


if __name__ == "__main__":
    unittest.main()


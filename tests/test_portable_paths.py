"""Relocation changes resource locations, never numerical checkpoint state."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from da2_vibe.paths import resolve, remap_metadata, repo_root


class PortablePaths(unittest.TestCase):
    def test_resolution_and_longest_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "paths.json"
            p.write_text(json.dumps({"resources": {"weights": "${ASSET_ROOT}/weights"}, "path_prefixes": {"/old": "${ASSET_ROOT}", "/old/special": "${weights}"}}))
            with patch.dict("os.environ", DA2_VIBE_PATHS=str(p), DA2_VIBE_ASSET_ROOT="/new"):
                self.assertEqual(resolve("${weights}/model.pt"), "/new/weights/model.pt")
                self.assertEqual(remap_metadata({"path": "/old/special/a", "number": 42.5, "text": "/older/a"}), {"path": "/new/weights/a", "number": 42.5, "text": "/older/a"})

    def test_missing_resource_and_cycle_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "paths.json"
            p.write_text(json.dumps({"resources": {"a": "${b}", "b": "${a}"}}))
            with patch.dict("os.environ", DA2_VIBE_PATHS=str(p)):
                with self.assertRaises(ValueError): resolve("${a}")
                with self.assertRaises(ValueError): resolve("${missing_resource}")

    def test_repository_is_resolved_independent_of_cwd(self):
        self.assertTrue((repo_root() / "configs/tasks/stairs.json").is_file())


if __name__ == "__main__":
    unittest.main()

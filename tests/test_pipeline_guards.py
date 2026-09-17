"""Fast regression tests that do not require GPU packages or model downloads."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainers.lora_state import validate_adapter_state
from tools.build_gpt_prompt_pairs import read_manifest, group_by_speaker, build_pairs


class TensorStub:
    def __init__(self, *shape):
        self.shape = shape


class PipelineGuards(unittest.TestCase):
    def test_adapter_complete_and_shapes(self):
        expected = {"a": TensorStub(2, 3), "b": TensorStub(3, 2)}
        validate_adapter_state(expected, expected.copy())
        for invalid in ({}, {"a": TensorStub(2, 3)},
                        dict(expected, c=TensorStub(1)),
                        dict(expected, b=TensorStub(1, 2))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_adapter_state(expected, invalid)

    def test_pair_paths_and_distinct_speakers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "features.jsonl"
            rows = [{"id": name, "speaker": speaker, "text_len": 10, "code_len": 100,
                     "condition_path": f"condition/{name}.npy", "codes_path": f"codes/{name}.npy",
                     "text_ids_path": f"text_ids/{name}.npy"}
                    for name, speaker in (("a", "one"), ("b", "one"), ("c", "two"))]
            source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            pairs = build_pairs(group_by_speaker(read_manifest(source)), 2, 1, 1)
            self.assertEqual(len(pairs), 2)
            for pair in pairs:
                self.assertNotEqual(pair["prompt_id"], pair["target_id"])
                self.assertEqual(pair["speaker"], "one")
                self.assertTrue(Path(pair["target_codes_path"]).is_absolute())
                self.assertEqual(Path(pair["target_codes_path"]).parent, root / "codes")

    def test_training_entry_import_has_no_side_effects(self):
        spec = importlib.util.spec_from_file_location("entry", ROOT / "train.py")
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_project_python_syntax(self):
        paths = [ROOT / "train.py", ROOT / "webui.py"]
        for directory in ("tools", "trainers", "tests"):
            paths.extend((ROOT / directory).rglob("*.py"))
        for path in paths:
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_setup_python_version_expression(self):
        script = (ROOT / "scripts/windows/setup.ps1").read_text()
        expression = next(line.split('-c "', 1)[1].rsplit('"', 1)[0]
                          for line in script.splitlines() if 'sys.version_info' in line)
        compile(expression, "setup-version-check", "exec")


if __name__ == "__main__":
    unittest.main()

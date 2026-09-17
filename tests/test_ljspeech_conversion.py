"""Dependency-free regression tests: python -m unittest discover -s tests -p test_ljspeech_conversion.py"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_ljspeech.py"


class ConversionTests(unittest.TestCase):
    def test_columns_unicode_paths_and_failure_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "voice 中文"
            (root / "wavs").mkdir(parents=True)
            for name in ("one", "two"):
                (root / "wavs" / (name + ".wav")).touch()
            metadata = root / "metadata.csv"
            metadata.write_text("one|raw 1|normalized one\ntwo|raw 2|normalized two\n", encoding="utf-8-sig")
            output = root / "voice.lst"
            command = [sys.executable, str(SCRIPT), "--dataset-dir", str(root), "--output", str(output)]
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0],
                             str((root / "wavs/one.wav").resolve()) + "\tnormalized one")
            result = subprocess.run(command + ["--text-column", "raw"], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            previous = output.read_bytes()
            self.assertIn(b"raw 1", previous)
            (root / "wavs/two.wav").unlink()
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()

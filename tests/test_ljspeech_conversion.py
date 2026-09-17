import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_ljspeech.py"


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0" * 2400)


class ConversionTests(unittest.TestCase):
    def test_two_column_filename_with_suffix_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "voice 中文"
            (root / "wavs").mkdir(parents=True)
            for name in ("one", "two"):
                write_wav(root / "wavs" / f"{name}.wav")
            (root / "metadata.csv").write_text(
                "one.wav|text one\ntwo.wav|text two\n", encoding="utf-8-sig")
            output = root / "voice.jsonl"
            command = [sys.executable, str(SCRIPT), "--dataset-dir", str(root), "--output", str(output)]
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["id"], "one")
            self.assertEqual(first["text"], "text one")
            self.assertEqual(Path(first["audio"]).name, "one.wav")
            self.assertEqual(first["speaker"], "ljspeech")
            self.assertEqual(first["language"], "en")
            self.assertAlmostEqual(first["duration"], 0.1)
            self.assertTrue(Path(first["audio"]).is_absolute())

            previous = output.read_bytes()
            (root / "wavs" / "two.wav").unlink()
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), previous)

            write_wav(root / "wavs" / "two.wav")
            for bad_id in ("one.wav", "ONE.wav", "../outside.wav", "bad:name.wav"):
                (root / "metadata.csv").write_text(
                    f"one.wav|one\n{bad_id}|two\n", encoding="utf-8")
                result = subprocess.run(command, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_bytes(), previous)

    def test_standard_three_column_format_remains_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "wavs").mkdir()
            for name in ("one", "two"):
                write_wav(root / "wavs" / f"{name}.wav")
            (root / "metadata.csv").write_text(
                "one|raw one|normalized one\ntwo|raw two|normalized two\n",
                encoding="utf-8",
            )
            output = root / "voice.jsonl"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--dataset-dir", str(root),
                 "--output", str(output)], capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["id"], "one")
            self.assertEqual(first["text"], "normalized one")

    def test_mixed_column_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "wavs").mkdir()
            for name in ("one", "two"):
                write_wav(root / "wavs" / f"{name}.wav")
            (root / "metadata.csv").write_text(
                "one.wav|one\ntwo|raw two|normalized two\n", encoding="utf-8"
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--dataset-dir", str(root),
                 "--output", str(root / "voice.jsonl")], capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"mixed two- and three-column", result.stderr)


if __name__ == "__main__":
    unittest.main()

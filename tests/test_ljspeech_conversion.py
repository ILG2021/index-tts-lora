import json
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
from pathlib import Path

from tools.prepare_ljspeech import wav_duration

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_ljspeech.py"


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0" * 2400)


class ConversionTests(unittest.TestCase):
    def test_extensible_wav_duration_uses_soundfile_fallback(self):
        fake_soundfile = types.SimpleNamespace(
            info=lambda _path: types.SimpleNamespace(frames=48000, samplerate=24000)
        )
        with patch("tools.prepare_ljspeech.wave.open", side_effect=wave.Error("unknown format: 65534")):
            with patch.dict(sys.modules, {"soundfile": fake_soundfile}):
                self.assertEqual(wav_duration(Path("extensible.wav")), 2.0)

    def test_missing_audio_is_skipped_when_enough_samples_remain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "wavs").mkdir()
            for name in ("one", "three"):
                write_wav(root / "wavs" / f"{name}.wav")
            (root / "metadata.csv").write_text(
                "one.wav|one\nmissing.wav|missing\nthree.wav|three\n",
                encoding="utf-8",
            )
            output = root / "voice.jsonl"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--dataset-dir", str(root),
                 "--output", str(output)], capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"missing audio, skipped", result.stderr)
            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertNotIn("missing", output.read_text(encoding="utf-8"))

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
            self.assertEqual(first["language"], "zh")
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

    def test_nested_audio_path_and_traversal_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested = root / "wavs" / "speaker one"
            nested.mkdir(parents=True)
            write_wav(nested / "file0001.wav")
            write_wav(nested / "file0002.wav")
            metadata = root / "metadata.csv"
            metadata.write_text(
                "speaker one/file0001.wav|text one\n"
                "speaker one\\file0002.wav|text two\n",
                encoding="utf-8",
            )
            output = root / "voice.jsonl"
            command = [sys.executable, str(SCRIPT), "--dataset-dir", str(root),
                       "--output", str(output)]
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(Path(rows[0]["audio"]), nested / "file0001.wav")
            self.assertNotIn("/", rows[0]["id"])
            self.assertNotIn("\\", rows[0]["id"])
            self.assertNotEqual(rows[0]["id"], rows[1]["id"])

            previous = output.read_bytes()
            metadata.write_text(
                "speaker one/file0001.wav|one\n../outside.wav|two\n", encoding="utf-8"
            )
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), previous)

    def test_speaker_from_folder_derives_subfolder_as_speaker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested1 = root / "wavs" / "chapter_01"
            nested2 = root / "wavs" / "chapter_02"
            nested1.mkdir(parents=True)
            nested2.mkdir(parents=True)
            write_wav(nested1 / "001.wav")
            write_wav(nested1 / "002.wav")
            write_wav(nested2 / "001.wav")
            metadata = root / "metadata.csv"
            metadata.write_text(
                "chapter_01/001.wav|text one\n"
                "chapter_01/002.wav|text two\n"
                "chapter_02/001.wav|text three\n",
                encoding="utf-8",
            )
            output = root / "voice.jsonl"
            command_default = [
                sys.executable, str(SCRIPT), "--dataset-dir", str(root),
                "--output", str(output),
            ]
            result = subprocess.run(command_default, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            rows_default = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            for r in rows_default:
                self.assertEqual(r["speaker"], "ljspeech")

            command_folder = [
                sys.executable, str(SCRIPT), "--dataset-dir", str(root),
                "--output", str(output), "--speaker-from-folder",
            ]
            result = subprocess.run(command_folder, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            rows_folder = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows_folder[0]["speaker"], "chapter_01")
            self.assertEqual(rows_folder[1]["speaker"], "chapter_01")
            self.assertEqual(rows_folder[2]["speaker"], "chapter_02")


if __name__ == "__main__":
    unittest.main()

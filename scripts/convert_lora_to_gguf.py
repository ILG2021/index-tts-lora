#!/usr/bin/env python3
"""Build a standalone audio.cpp IndexTTS2 GGUF with one LoRA merged in.

audio.cpp v0.8.1 has no runtime LoRA adapter API for the ``index_tts2``
family, so this creates a complete per-voice GGUF.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGER = PROJECT_ROOT / "integrations/audiocpp/scripts/convert_index_tts2.py"


def build_command(args: argparse.Namespace, staging: Path) -> list[str]:
    return [sys.executable, str(STAGER), "--model-dir", str(Path(args.model_dir).resolve()),
            "--output-dir", str(staging), "--lora-checkpoint", str(Path(args.lora_checkpoint).resolve()),
            "--run-converter", str(Path(args.audiocpp_gguf).resolve()),
            "--output-gguf", str(Path(args.output).resolve()), "--type", args.dtype]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--audiocpp-gguf", default="integrations/audiocpp/bin/audiocpp_gguf.exe")
    parser.add_argument("--output", required=True, help="完整的、已合并 LoRA 的 GGUF 输出路径")
    parser.add_argument("--dtype", default="q8_0", choices=("orig", "f16", "bf16", "q8_0"))
    parser.add_argument("--staging-dir", type=Path)
    args = parser.parse_args()
    for label, path in (("LoRA checkpoint", args.lora_checkpoint), ("audiocpp_gguf", args.audiocpp_gguf)):
        if not Path(path).is_file():
            parser.error(f"{label} 不存在：{path}")
    Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.staging_dir:
        args.staging_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.run(build_command(args, args.staging_dir.resolve())).returncode
    with tempfile.TemporaryDirectory(prefix="indextts2-lora-gguf-") as tmp:
        return subprocess.run(build_command(args, Path(tmp))).returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert LJSpeech metadata.csv to the JSONL format used by IndexTTS2 preprocessing."""

import argparse
import csv
import json
import re
import wave
from pathlib import Path


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", default="datasets/ljspeech.jsonl")
    parser.add_argument("--text-column", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--speaker", default="ljspeech")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    root = Path(args.dataset_dir).expanduser().resolve()
    metadata = root / "metadata.csv"
    wav_dir = root / "wavs"
    if not metadata.is_file() or not wav_dir.is_dir():
        parser.error(f"Expected {metadata} and {wav_dir}")

    records = []
    seen_ids = set()
    text_index = 2 if args.text_column == "normalized" else 1
    with metadata.open("r", encoding="utf-8-sig", newline="") as source:
        for line_number, row in enumerate(csv.reader(source, delimiter="|"), 1):
            if len(row) < 3:
                raise ValueError(f"metadata.csv line {line_number}: expected id|raw|normalized")
            sample_id = row[0].strip()
            if not sample_id or sample_id in (".", "..") or re.search(r'[<>:"/\\|?*\x00-\x1f]', sample_id) or sample_id.endswith((".", " ")):
                raise ValueError(f"metadata.csv line {line_number}: invalid sample ID {sample_id!r}")
            if sample_id.casefold() in seen_ids:
                raise ValueError(f"metadata.csv line {line_number}: duplicate sample ID {sample_id!r}")
            seen_ids.add(sample_id.casefold())
            audio = (wav_dir / f"{sample_id}.wav").resolve()
            text = " ".join(row[text_index].split())
            if not audio.is_file():
                raise FileNotFoundError(f"metadata.csv line {line_number}: missing {audio}")
            if not text:
                continue
            records.append({"id": sample_id, "text": text, "audio": str(audio),
                            "speaker": args.speaker, "language": args.language,
                            "duration": round(wav_duration(audio), 6)})
    if len(records) < 2:
        raise RuntimeError("Fewer than two usable samples were found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(f"Wrote {len(records)} IndexTTS2 samples to {output.resolve()}")


if __name__ == "__main__":
    main()

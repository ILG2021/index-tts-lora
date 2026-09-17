#!/usr/bin/env python3
"""Convert an LJSpeech directory into extract_codec.py's UTF-8 audio list."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare LJSpeech metadata for IndexTTS LoRA fine-tuning.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing metadata.csv and wavs/")
    parser.add_argument("--output", default="finetune_data/ljspeech.lst")
    parser.add_argument("--text-column", choices=["normalized", "raw"], default="normalized")
    args = parser.parse_args()

    root = Path(args.dataset_dir).expanduser().resolve()
    metadata = root / "metadata.csv"
    wav_dir = root / "wavs"
    if not metadata.is_file() or not wav_dir.is_dir():
        parser.error(f"Expected {metadata} and {wav_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text_index = 2 if args.text_column == "normalized" else 1
    entries = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.reader(source, delimiter="|"), 1):
            if len(row) < 3:
                raise ValueError(f"metadata.csv line {row_number}: expected id|raw|normalized")
            wav_path = (wav_dir / f"{row[0].strip()}.wav").resolve()
            text = row[text_index].strip().replace("\t", " ").replace("\r", " ").replace("\n", " ")
            if not wav_path.is_file():
                raise FileNotFoundError(f"metadata.csv line {row_number}: missing {wav_path}")
            if not text:
                continue
            entries.append(f"{wav_path}\t{text}\n")
    if len(entries) < 2:
        raise RuntimeError("Fewer than two usable samples were found.")
    with output.open("w", encoding="utf-8", newline="\n") as target:
        target.writelines(entries)
    print(f"Wrote {len(entries)} samples to {output.resolve()}")


if __name__ == "__main__":
    main()

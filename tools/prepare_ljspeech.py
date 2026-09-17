#!/usr/bin/env python3
"""Convert two- or three-column LJSpeech metadata to IndexTTS2 JSONL."""

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
    parser.add_argument(
        "--text-column", choices=("normalized", "raw"), default="normalized",
        help="For three-column metadata only; two-column metadata always uses column 2.",
    )
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
    column_count = None
    with metadata.open("r", encoding="utf-8-sig", newline="") as source:
        for line_number, row in enumerate(csv.reader(source, delimiter="|"), 1):
            if len(row) not in (2, 3):
                raise ValueError(
                    f"metadata.csv line {line_number}: expected filename.wav|text "
                    "or id|raw|normalized"
                )
            if column_count is None:
                column_count = len(row)
            elif len(row) != column_count:
                raise ValueError(
                    f"metadata.csv line {line_number}: mixed two- and three-column rows"
                )
            filename_field = row[0].strip()
            if (not filename_field or filename_field in (".", "..")
                    or re.search(r'[<>:"/\\|?*\x00-\x1f]', filename_field)
                    or filename_field.endswith((".", " "))):
                raise ValueError(
                    f"metadata.csv line {line_number}: invalid filename/ID {filename_field!r}"
                )
            supplied_suffix = Path(filename_field).suffix
            if supplied_suffix and supplied_suffix.lower() != ".wav":
                raise ValueError(
                    f"metadata.csv line {line_number}: audio filename must end in .wav"
                )
            audio_filename = filename_field if supplied_suffix else f"{filename_field}.wav"
            sample_id = Path(audio_filename).stem
            if sample_id.casefold() in seen_ids:
                raise ValueError(f"metadata.csv line {line_number}: duplicate sample ID {sample_id!r}")
            seen_ids.add(sample_id.casefold())
            audio = (wav_dir / audio_filename).resolve()
            text_index = 1 if len(row) == 2 or args.text_column == "raw" else 2
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

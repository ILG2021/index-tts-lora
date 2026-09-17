#!/usr/bin/env python3
"""Convert two- or three-column LJSpeech metadata to IndexTTS2 JSONL."""

import argparse
import csv
import hashlib
import json
import re
import sys
import wave
from pathlib import Path, PurePosixPath


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())
    except (wave.Error, EOFError):
        # Python 3.10's wave module cannot read WAVE_FORMAT_EXTENSIBLE (65534),
        # which is common in 24-bit/multichannel files exported by audio tools.
        try:
            import soundfile

            info = soundfile.info(str(path))
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                f"Cannot read WAV metadata for {path}. Install soundfile or convert "
                "the file to PCM WAV (mono, 16/24-bit)."
            ) from error
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Audio has invalid frame count/sample rate: {path}")
        return info.frames / float(info.samplerate)


def parse_audio_reference(value: str, line_number: int) -> tuple[PurePosixPath, str]:
    """Return a safe path relative to wavs/ and a filesystem-safe sample ID."""
    normalized = value.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (not normalized or relative.is_absolute() or normalized.startswith("//")
            or any(part in ("", ".", "..") for part in relative.parts)):
        raise ValueError(
            f"metadata.csv line {line_number}: audio path must stay below wavs/: {value!r}"
        )
    for part in relative.parts:
        if (re.search(r'[<>:"|?*\x00-\x1f]', part)
                or part.endswith((".", " "))):
            raise ValueError(
                f"metadata.csv line {line_number}: invalid audio path component {part!r}"
            )
    if relative.suffix.lower() != ".wav":
        raise ValueError(
            f"metadata.csv line {line_number}: audio filename must end in .wav"
        )
    stem_path = relative.with_suffix("")
    if len(stem_path.parts) == 1:
        sample_id = stem_path.name
    else:
        digest = hashlib.sha1(relative.as_posix().casefold().encode("utf-8")).hexdigest()[:10]
        sample_id = "__".join(stem_path.parts) + f"__{digest}"
    return relative, sample_id


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
    missing_count = 0
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
            if len(row) == 2 or Path(filename_field).suffix:
                relative_audio, sample_id = parse_audio_reference(filename_field, line_number)
            else:
                # Official three-column LJSpeech stores a bare ID in column one.
                relative_audio, sample_id = parse_audio_reference(
                    f"{filename_field}.wav", line_number
                )
            if sample_id.casefold() in seen_ids:
                raise ValueError(f"metadata.csv line {line_number}: duplicate sample ID {sample_id!r}")
            seen_ids.add(sample_id.casefold())
            audio = wav_dir.joinpath(*relative_audio.parts).resolve()
            try:
                audio.relative_to(wav_dir.resolve())
            except ValueError as error:
                raise ValueError(
                    f"metadata.csv line {line_number}: audio path escaped wavs/"
                ) from error
            text_index = 1 if len(row) == 2 or args.text_column == "raw" else 2
            text = " ".join(row[text_index].split())
            if not audio.is_file():
                missing_count += 1
                print(
                    f"[Warn] metadata.csv line {line_number}: missing audio, skipped: {audio}",
                    file=sys.stderr,
                )
                continue
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
    print(
        f"Wrote {len(records)} IndexTTS2 samples to {output.resolve()} "
        f"(missing skipped: {missing_count})"
    )


if __name__ == "__main__":
    main()

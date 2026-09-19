#!/usr/bin/env python3
"""Convert this project's PEFT checkpoint to an audio.cpp runtime adapter.

The output contains only GPT-2 LoRA A/B tensors.  ``alpha / rank`` is folded
into B once so the C++ runtime can apply an additional per-request scale
without materialising or modifying the base weights.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


TARGET_SUFFIXES = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


def _metadata(checkpoint: dict) -> tuple[int, float, tuple[str, ...]]:
    meta = checkpoint.get("lora")
    if meta is None and isinstance(checkpoint.get("extra"), dict):
        meta = checkpoint["extra"].get("lora")
    if not isinstance(meta, dict) or not all(key in meta for key in ("r", "alpha", "target_modules")):
        raise ValueError("LoRA checkpoint 缺少 r/alpha/target_modules 元数据")
    rank, alpha = int(meta["r"]), float(meta["alpha"])
    if rank <= 0 or not math.isfinite(alpha):
        raise ValueError("LoRA rank 必须为正数，alpha 必须为有限数")
    raw_targets = meta["target_modules"]
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    if not isinstance(raw_targets, (list, tuple)) or not raw_targets:
        raise ValueError("LoRA target_modules 必须是非空列表")
    targets = tuple(str(value) for value in raw_targets)
    unsupported = set(targets) - {"c_attn", "c_proj", "c_fc"}
    if unsupported:
        raise ValueError(f"不支持的 LoRA target_modules：{sorted(unsupported)}")
    return rank, alpha, targets


def _validate_coverage(pairs: dict[str, dict[str, object]], targets: tuple[str, ...]) -> None:
    expected_suffixes = set()
    if "c_attn" in targets:
        expected_suffixes.add("attn.c_attn")
    if "c_fc" in targets:
        expected_suffixes.add("mlp.c_fc")
    if "c_proj" in targets:
        expected_suffixes.update(("attn.c_proj", "mlp.c_proj"))
    expected = {f"gpt.h.{layer}.{suffix}" for layer in range(24) for suffix in expected_suffixes}
    actual = set(pairs)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing[:8]}" + ("..." if len(missing) > 8 else ""))
        if extra:
            details.append(f"extra={extra[:8]}" + ("..." if len(extra) > 8 else ""))
        raise ValueError("LoRA 投影覆盖与 checkpoint 元数据不一致：" + ", ".join(details))


def _split_key(key: str) -> tuple[str, str]:
    for kind in ("A", "B"):
        for marker in (f".lora_{kind}.weight", f".lora_{kind}.default.weight"):
            if key.endswith(marker):
                stem = key.removeprefix("base_model.model.")[: -len(marker)]
                stem = stem.removeprefix("transformer.")
                if not stem.startswith("gpt."):
                    stem = f"gpt.{stem}"
                if not stem.startswith("gpt.h.") or not stem.endswith(TARGET_SUFFIXES):
                    raise ValueError(f"不支持的 IndexTTS2 LoRA 目标：{key}")
                return stem, kind
    raise ValueError(f"不支持的 adapter 张量：{key}")


def convert(checkpoint_path: Path, output_path: Path) -> dict[str, object]:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("需要安装 torch 和 safetensors") from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = checkpoint.get("adapter") if isinstance(checkpoint, dict) else None
    if not isinstance(adapter, dict) or not adapter:
        raise ValueError("不是本项目生成的 LoRA checkpoint（缺少 adapter）")
    rank, alpha, targets = _metadata(checkpoint)
    pairs: dict[str, dict[str, object]] = {}
    for key, tensor in adapter.items():
        stem, kind = _split_key(key)
        pair = pairs.setdefault(stem, {})
        if kind in pair:
            raise ValueError(f"LoRA 归一化后出现重复张量：{stem}.lora_{kind}")
        pair[kind] = tensor.detach().float().contiguous()

    _validate_coverage(pairs, targets)

    tensors = {}
    for stem, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"LoRA A/B 不完整：{stem}")
        a, b = pair["A"], pair["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"LoRA rank/形状不匹配：{stem}")
        tensors[f"{stem}.lora_A"] = a
        tensors[f"{stem}.lora_B"] = (b * (alpha / rank)).contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata={
        "format": "audiocpp.index_tts2.lora.v1",
        "rank": str(rank),
        "alpha": str(alpha),
        "alpha_folded_into_b": "true",
    })
    return {"projections": len(pairs), "rank": rank, "alpha": alpha, "output": output_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint 不存在：{args.checkpoint}")
    result = convert(args.checkpoint.resolve(), args.output.resolve())
    print(
        f"已生成运行时 LoRA：{result['output']} "
        f"({result['projections']} projections, rank={result['rank']}, alpha={result['alpha']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

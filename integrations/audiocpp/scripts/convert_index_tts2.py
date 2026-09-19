#!/usr/bin/env python3
"""专门用于 IndexTTS2（v2）官方权重的 Safetensors staging 与 audio.cpp GGUF 转换工具。

本工具专为 IndexTTS2 设计，将 checkpoints 目录中的 PyTorch 权重（.pth/.pt）
转换为 audio.cpp index_tts2 规范所需的 Safetensors 布局，并组装 root/ sidecar
文件（config.yaml, bpe.model 等），然后调用 audiocpp_gguf 生成 GGUF 格式模型。

用法：
    python integrations/audiocpp/scripts/convert_index_tts2.py \\
        --model-dir checkpoints \\
        --output-dir staging \\
        --run-converter integrations/audiocpp/bin/audiocpp_gguf.exe \\
        --type f16
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

# ── 必需检查的文件 ────────────────────────────────────────────────────────────
REQUIRED_INDEXTTS2_FILES = [
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
]

# GGUF 张量命名空间映射（需匹配 audio.cpp index_tts2 模型规范）
TENSOR_OUTPUTS = [
    ("gpt", "gpt.safetensors"),
    ("s2mel", "s2mel.safetensors"),
    ("speaker_matrix", "speaker_matrix.safetensors"),
    ("emotion_matrix", "emotion_matrix.safetensors"),
    ("wav2vec2bert_stats", "wav2vec2bert_stats.safetensors"),
    ("wav2vec2bert", "wav2vec2bert.safetensors"),
    ("semantic_codec", "semantic_codec.safetensors"),
    ("campplus", "campplus.safetensors"),
    ("bigvgan", "bigvgan.safetensors"),
    ("qwen_emotion", "qwen_emotion.safetensors"),
]

QWEN_SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"缺少必要文件 [{label}]: {path}")
    return path


def _load_checkpoint(path: Path):
    try:
        import torch
    except ImportError:
        raise ImportError("模型转换需要 PyTorch。请在项目虚拟环境 (.venv) 中运行。")
    return torch.load(path, map_location="cpu", weights_only=False)


def _flatten(obj, prefix: str = "", out: Dict[str, any] | None = None) -> Dict[str, any]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(value, f"{prefix}{key}.", out)
    elif hasattr(obj, "shape"):
        out[prefix.rstrip(".")] = obj.contiguous()
    else:
        raise TypeError(f"发现意外的非张量叶子节点 {prefix!r}: {type(obj)}")
    return out


def _save_safetensors(tensors: Dict[str, any], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("模型转换需要 safetensors。请运行 pip install safetensors。")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    print(f"已写入: {path} ({len(tensors)} 个张量)")


def _extract_lora_config(checkpoint: dict) -> tuple[int, float]:
    metadata = checkpoint.get("lora")
    if metadata is None and isinstance(checkpoint.get("extra"), dict):
        metadata = checkpoint["extra"].get("lora")
    if not isinstance(metadata, dict) or "r" not in metadata or "alpha" not in metadata:
        raise ValueError("LoRA checkpoint 缺少 r/alpha 元数据")
    return int(metadata["r"]), float(metadata["alpha"])


def _merge_lora(base: Dict[str, any], checkpoint_path: Path) -> Dict[str, any]:
    """Merge this project's PEFT adapter into the official GPT state dict.

    IndexTTS2 uses transformers Conv1D modules whose stored weight is [in,out],
    while PEFT's B@A delta is [out,in].  Shape matching below handles that
    transpose explicitly and rejects every ambiguous or incomplete adapter.
    """
    checkpoint = _load_checkpoint(_require_file(checkpoint_path, "LoRA checkpoint"))
    adapter = checkpoint.get("adapter") if isinstance(checkpoint, dict) else None
    if not isinstance(adapter, dict) or not adapter:
        raise ValueError("不是本项目生成的 LoRA checkpoint（缺少 adapter）")
    rank, alpha = _extract_lora_config(checkpoint)
    scale = alpha / rank
    pairs: dict[str, dict[str, any]] = {}
    for key, value in adapter.items():
        marker = None
        for candidate in (".lora_A.weight", ".lora_B.weight"):
            if key.endswith(candidate):
                marker = candidate
                break
        if marker is None:
            raise ValueError(f"不支持的 adapter 张量：{key}")
        stem = key.removeprefix("base_model.model.")[:-len(marker)]
        pairs.setdefault(stem, {})[marker[6]] = value.float()
    merged = dict(base)
    for stem, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"LoRA A/B 不完整：{stem}")
        normalized = stem.removeprefix("transformer.")
        candidates = (f"{stem}.weight", f"{normalized}.weight", f"gpt.{normalized}.weight")
        matches = [name for name in candidates if name in merged]
        if len(set(matches)) != 1:
            raise KeyError(f"基础 GPT 中无法唯一定位 LoRA 目标 {stem}；候选={candidates}，命中={matches}")
        target = matches[0]
        delta = pair["B"] @ pair["A"]
        weight = merged[target]
        if delta.shape == weight.shape:
            pass
        elif delta.T.shape == weight.shape:
            delta = delta.T
        else:
            raise ValueError(f"LoRA 形状不匹配 {target}: delta={tuple(delta.shape)}, base={tuple(weight.shape)}")
        merged[target] = (weight.float() + delta * scale).to(weight.dtype).contiguous()
    print(f"已将 LoRA 合并到 GPT：{checkpoint_path}（{len(pairs)} 个模块，scale={scale:g}）")
    return merged


def _convert_gpt(model_dir: Path, output_dir: Path, lora_checkpoint: Path | None = None) -> None:
    obj = _load_checkpoint(_require_file(model_dir / "gpt.pth", "gpt.pth"))
    flat = _flatten(obj)
    if lora_checkpoint is not None:
        flat = _merge_lora(flat, lora_checkpoint)
    _save_safetensors(flat, output_dir / "gpt.safetensors")


def _convert_s2mel(model_dir: Path, output_dir: Path) -> None:
    obj = _load_checkpoint(_require_file(model_dir / "s2mel.pth", "s2mel.pth"))
    if isinstance(obj, dict) and isinstance(obj.get("net"), dict):
        obj = obj["net"]
    _save_safetensors(_flatten(obj), output_dir / "s2mel.safetensors")


def _convert_emotion_matrices(model_dir: Path, output_dir: Path) -> None:
    """feat1.pt (说话人矩阵) 与 feat2.pt (情感矩阵)。"""
    speaker = _load_checkpoint(_require_file(model_dir / "feat1.pt", "feat1.pt"))
    emotion = _load_checkpoint(_require_file(model_dir / "feat2.pt", "feat2.pt"))
    _save_safetensors({"tensor": speaker.float().contiguous()}, output_dir / "speaker_matrix.safetensors")
    _save_safetensors({"tensor": emotion.float().contiguous()}, output_dir / "emotion_matrix.safetensors")


def _convert_wav2vec2bert_stats(model_dir: Path, output_dir: Path) -> None:
    obj = _load_checkpoint(_require_file(model_dir / "wav2vec2bert_stats.pt", "wav2vec2bert_stats.pt"))
    flat = {key: value.float().contiguous() for key, value in _flatten(obj).items()}
    _save_safetensors(flat, output_dir / "wav2vec2bert_stats.safetensors")


def _convert_w2v_bert(w2v_bert_dir: Path, output_dir: Path) -> None:
    """提取/复制 w2v-bert-2.0 权重至 staging 目录。"""
    w2v_safetensors = w2v_bert_dir / "model.safetensors"
    if w2v_safetensors.is_file():
        shutil.copyfile(w2v_safetensors, output_dir / "wav2vec2bert.safetensors")
        print(f"已复制 w2v-bert 权重: {w2v_safetensors} -> {output_dir / 'wav2vec2bert.safetensors'}")
    else:
        w2v_bin = w2v_bert_dir / "pytorch_model.bin"
        if w2v_bin.is_file():
            obj = _load_checkpoint(w2v_bin)
            _save_safetensors(_flatten(obj), output_dir / "wav2vec2bert.safetensors")
        else:
            raise FileNotFoundError(
                f"在 {w2v_bert_dir} 中未找到 model.safetensors 或 pytorch_model.bin。\n"
                "请运行 scripts/windows/download_models.ps1 下载完整模型。"
            )


def _stage_auxiliary_models(model_dir: Path, output_dir: Path, root_dir: Path) -> None:
    hf_cache = model_dir / "hf_cache"
    _copy_file(hf_cache / "semantic_codec_model.safetensors",
               output_dir / "semantic_codec.safetensors", "semantic codec")

    camp = _flatten(_load_checkpoint(_require_file(
        hf_cache / "campplus_cn_common.bin", "CAMPPlus")))
    camp = {k if k.startswith("speaker_encoder.") else f"speaker_encoder.{k}": v for k, v in camp.items()}
    _save_safetensors(camp, output_dir / "campplus.safetensors")

    bigvgan_dir = hf_cache / "bigvgan"
    bigvgan = _flatten(_load_checkpoint(_require_file(
        bigvgan_dir / "bigvgan_generator.pt", "BigVGAN")))
    bigvgan = {k.removeprefix("generator."): v for k, v in bigvgan.items()}
    _save_safetensors(bigvgan, output_dir / "bigvgan.safetensors")
    _copy_file(bigvgan_dir / "config.json", root_dir / "bigvgan/config.json", "BigVGAN config")

    qwen_dir = model_dir / "qwen0.6bemo4-merge"
    _copy_file(qwen_dir / "model.safetensors", output_dir / "qwen_emotion.safetensors", "Qwen emotion")
    for name in QWEN_SIDECARS:
        _copy_file(qwen_dir / name, root_dir / "qwen0.6bemo4-merge" / name, f"Qwen {name}")


def _copy_file(src: Path, dst: Path, label: str) -> None:
    _require_file(src, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"已复制 [{label}]: {src.name} -> {dst}")


def stage_indextts2(
    model_dir: Path,
    output_dir: Path,
    w2v_bert_dir: Path | None = None,
    lora_checkpoint: Path | None = None,
) -> Path:
    """完成 IndexTTS2 的 Safetensors staging 与 root/ sidecar 目录组装。"""
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = output_dir / "root"
    root_dir.mkdir(parents=True, exist_ok=True)

    if w2v_bert_dir is None:
        candidates = [
            model_dir / "hf_cache" / "w2v-bert-2.0",
            model_dir / "w2v-bert-2.0",
        ]
        for c in candidates:
            if c.is_dir():
                w2v_bert_dir = c
                break
        if w2v_bert_dir is None:
            w2v_bert_dir = candidates[0]

    print("=== 开始 IndexTTS2 Safetensors 转换 ===")
    _convert_gpt(model_dir, output_dir, lora_checkpoint)
    _convert_s2mel(model_dir, output_dir)
    _convert_emotion_matrices(model_dir, output_dir)
    _convert_wav2vec2bert_stats(model_dir, output_dir)
    _convert_w2v_bert(w2v_bert_dir, output_dir)
    _stage_auxiliary_models(model_dir, output_dir, root_dir)

    print("\n=== 组装 sidecar 配置 ===")
    _copy_file(model_dir / "config.yaml", root_dir / "config.yaml", "config.yaml")
    _copy_file(model_dir / "bpe.model", root_dir / "bpe.model", "bpe.model")

    # w2v-bert sidecar 配置
    if (w2v_bert_dir / "config.json").is_file():
        _copy_file(w2v_bert_dir / "config.json", root_dir / "w2v-bert-2.0" / "config.json", "w2v config.json")
    if (w2v_bert_dir / "preprocessor_config.json").is_file():
        _copy_file(
            w2v_bert_dir / "preprocessor_config.json",
            root_dir / "w2v-bert-2.0" / "preprocessor_config.json",
            "w2v preprocessor_config.json",
        )

    print(f"Staging 准备完成: {output_dir}")
    return output_dir


def build_converter_command(
    output_dir: Path,
    converter_exe: str,
    quant_type: str = "f16",
    output_gguf: Path | None = None,
) -> list[str]:
    """构造调用 audiocpp_gguf.exe 转换 GGUF 的命令行。"""
    if output_gguf is None:
        output_gguf = output_dir / f"index_tts2-{quant_type}.gguf"

    cmd = [str(converter_exe)]
    for namespace, filename in TENSOR_OUTPUTS:
        tensor_file = output_dir / filename
        cmd.extend(["--input", f"{namespace}={tensor_file}"])

    cmd.extend([
        "--root", str(output_dir / "root"),
        "--family", "index_tts2",
        "--type", quant_type,
        "--output", str(output_gguf),
    ])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IndexTTS2 (v2) PyTorch 权重 -> audio.cpp GGUF 转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"),
                        help="IndexTTS2 checkpoints 目录（含 gpt.pth, s2mel.pth 等），默认 checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="用于存放中间 Safetensors 与 sidecar 的 staging 目录")
    parser.add_argument("--w2v-bert-dir", type=Path, default=None,
                        help="w2v-bert-2.0 目录（默认自动检测 <model-dir>/hf_cache/w2v-bert-2.0）")
    parser.add_argument("--lora-checkpoint", type=Path, default=None,
                        help="先把本项目 LoRA checkpoint 合并进 GPT，再打包独立 GGUF")
    parser.add_argument("--run-converter", type=str, default=None,
                        help="audiocpp_gguf 可执行文件路径；若指定则立即执行 GGUF 生成")
    parser.add_argument("--output-gguf", type=Path, default=None,
                        help="最终生成的 GGUF 文件路径")
    parser.add_argument("--type", dest="quant_type", default="f16",
                        choices=["f16", "bf16", "q8_0", "orig"],
                        help="GGUF 权重类型（默认 f16）")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()

    missing = [f for f in REQUIRED_INDEXTTS2_FILES if not (model_dir / f).is_file()]
    if missing:
        print(f"错误: checkpoints 目录缺少以下 IndexTTS2 文件: {missing}", file=sys.stderr)
        return 1

    stage_indextts2(model_dir, output_dir, args.w2v_bert_dir, args.lora_checkpoint)

    if args.run_converter:
        cmd = build_converter_command(
            output_dir=output_dir,
            converter_exe=args.run_converter,
            quant_type=args.quant_type,
            output_gguf=args.output_gguf,
        )
        print("\n执行 GGUF 转换:")
        print(" ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"错误: 转换失败 (返回码 {res.returncode})", file=sys.stderr)
            return res.returncode
        print("\n✅ IndexTTS2 GGUF 转换成功！")

    return 0


if __name__ == "__main__":
    sys.exit(main())

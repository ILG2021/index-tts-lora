#!/usr/bin/env python3
"""将 IndexTTS2 / IndexTTS2.5 的 PyTorch checkpoint 转换为 audio.cpp 兼容的 GGUF 格式。

此脚本是对 audio.cpp 的 tools/community_models/convert_index_tts2_5.py 和
audiocpp_gguf 转换器的封装，提供与本项目 checkpoints 目录约定一致的接口。

用法（推荐先确保已下载 audio.cpp 预编译二进制，见 integrations/audiocpp/scripts/）：

    python scripts/convert_to_gguf.py \\
        --model-dir checkpoints \\
        --audiocpp-dir integrations/audiocpp/bin \\
        --output integrations/audiocpp/weights/base.gguf

可选精度（--dtype）：f16 | bf16 | q8_0（默认 f16）。

    python scripts/convert_to_gguf.py \\
        --model-dir checkpoints \\
        --audiocpp-dir integrations/audiocpp/bin \\
        --output integrations/audiocpp/weights/base.gguf \\
        --dtype q8_0

参考架构：
  MOSS-TTS openmoss  →  integrations/openmoss/scripts/convert_hf_to_gguf.py
  audio.cpp          →  tools/community_models/convert_index_tts2_5.py
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 辅助：检查所需文件
# ---------------------------------------------------------------------------

# IndexTTS2 checkpoints 目录中必须存在的文件
# IndexTTS2 官方 checkpoints 必需文件
REQUIRED_CHECKPOINT_FILES = [
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
]


def check_model_dir(model_dir: Path) -> str:
    """校验 IndexTTS2 checkpoints 目录完整性。"""
    missing = [f for f in REQUIRED_CHECKPOINT_FILES if not (model_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"IndexTTS2 checkpoints 目录 {model_dir} 缺少以下文件：{missing}\n"
            "请先运行 scripts/windows/download_models.ps1 下载完整模型。"
        )
    log.info("IndexTTS2 模型文件校验通过: %s", model_dir)
    return "v2"


def find_audiocpp_gguf(audiocpp_dir: Path) -> Path:
    """在 audiocpp_dir 中寻找 audiocpp_gguf 可执行文件（或 .exe）。"""
    for name in ("audiocpp_gguf", "audiocpp_gguf.exe"):
        candidate = audiocpp_dir / name
        if candidate.is_file():
            return candidate
    # 备选：在 PATH 中寻找
    which = shutil.which("audiocpp_gguf")
    if which:
        return Path(which)
    raise FileNotFoundError(
        f"在 {audiocpp_dir} 及 PATH 中均未找到 audiocpp_gguf。\n"
        "请先运行 integrations/audiocpp/scripts/download_audiocpp.ps1 下载二进制文件。"
    )


def find_audiocpp_staging_script(audiocpp_dir: Path) -> Path:
    """寻找 IndexTTS2 专用的 staging 转换脚本。"""
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        # 1. 本项目自带的专为 IndexTTS2 优化的转换脚本（最高优先级）
        project_root / "integrations" / "audiocpp" / "scripts" / "convert_index_tts2.py",
        # 2. audio.cpp 源码仓库中的 community models 脚本
        audiocpp_dir.parent / "tools" / "community_models" / "convert_index_tts2.py",
        audiocpp_dir.parent / "tools" / "community_models" / "convert_index_tts2_5.py",
        audiocpp_dir / "convert_index_tts2.py",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "未找到 convert_index_tts2.py 脚本。\n"
        "请确保 integrations/audiocpp/scripts/convert_index_tts2.py 存在。"
    )


# ---------------------------------------------------------------------------
# 主转换流程
# ---------------------------------------------------------------------------

def run_staging(
    model_dir: Path,
    staging_dir: Path,
    staging_script: Path,
    audiocpp_gguf_exe: Path,
    output_gguf: Path,
    dtype: str,
    lora_checkpoint: Path | None = None,
) -> None:
    """运行 staging 脚本，将 .pth 文件整理为 Safetensors 并调用 audiocpp_gguf。"""
    cmd = [
        sys.executable,
        str(staging_script),
        "--model-dir", str(model_dir),
        "--output-dir", str(staging_dir),
        "--run-converter", str(audiocpp_gguf_exe),
        "--output-gguf", str(output_gguf),
        "--type", dtype,
    ]
    if lora_checkpoint is not None:
        cmd.extend(["--lora-checkpoint", str(lora_checkpoint)])
    log.info("运行 staging 脚本：%s", " ".join(cmd))
    subprocess.check_call(cmd)


def convert(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).resolve()
    audiocpp_dir = Path(args.audiocpp_dir).resolve()
    output = Path(args.output).resolve()

    log.info("=== IndexTTS2 → audio.cpp GGUF 转换 ===")
    log.info("模型目录：%s", model_dir)
    log.info("audio.cpp 目录：%s", audiocpp_dir)
    log.info("目标输出：%s", output)
    log.info("目标精度：%s", args.dtype)

    # 1. 校验输入
    check_model_dir(model_dir)
    gguf_exe = find_audiocpp_gguf(audiocpp_dir)
    staging_script = find_audiocpp_staging_script(audiocpp_dir)
    log.info("使用转换脚本：%s", staging_script)
    log.info("使用 audiocpp_gguf：%s", gguf_exe)

    output.parent.mkdir(parents=True, exist_ok=True)

    # 2. 创建/复用 staging 目录
    cleanup = False
    if args.staging_dir:
        staging_dir = Path(args.staging_dir).resolve()
        staging_dir.mkdir(parents=True, exist_ok=True)
    else:
        staging_dir = Path(tempfile.mkdtemp(prefix="indextts2-staging-"))
        cleanup = not args.keep_staging

    try:
        run_staging(model_dir, staging_dir, staging_script, gguf_exe, output, args.dtype,
                    Path(args.lora_checkpoint).resolve() if args.lora_checkpoint else None)
        # 若 staging 脚本写入了 staging_dir/index_tts2-xxx.gguf 且未直接写入 output
        if not output.is_file():
            candidates = list(staging_dir.glob("*.gguf"))
            if candidates:
                shutil.move(str(candidates[0]), str(output))
                log.info("已移动 GGUF: %s -> %s", candidates[0].name, output)

        # 3. 可选：自动执行 Q8_0 量化
        if args.quantize_q8:
            q8_output = output.with_name(f"{output.stem}-q8_0.gguf")
            if args.dtype == "q8_0":
                log.info("输出已经是 Q8_0，跳过重复转换")
            else:
                run_staging(model_dir, staging_dir, staging_script, gguf_exe,
                            q8_output, "q8_0",
                            Path(args.lora_checkpoint).resolve() if args.lora_checkpoint else None)

    finally:
        if cleanup:
            shutil.rmtree(staging_dir, ignore_errors=True)
            log.info("已清理临时 staging 目录")
        else:
            log.info("staging 目录保留于：%s", staging_dir)

    if output.is_file():
        size_gb = output.stat().st_size / 1024 ** 3
        log.info("转换完成 → %s（%.2f GB）", output, size_gb)
        log.info("")
        log.info("后续步骤：")
        log.info("  1. 命令行测试推理：")
        log.info("     python scripts/infer_audiocpp.py --text '你好，IndexTTS2' --voice-ref assets/reference.wav")
        log.info("  2. 启动 WebUI：")
        log.info("     python webui.py --backend audiocpp --model %s", output)
    else:
        log.error("转换结束但未找到输出文件：%s", output)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IndexTTS2 checkpoints → audio.cpp GGUF 转换器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model-dir", default="checkpoints",
                   help="IndexTTS2 checkpoints 目录（含 config.yaml, gpt.pth 等），默认：checkpoints")
    p.add_argument("--audiocpp-dir", default="integrations/audiocpp/bin",
                   help="audio.cpp 二进制目录（含 audiocpp_gguf.exe）")
    p.add_argument("--output", default="integrations/audiocpp/weights/base.gguf",
                   help="输出 GGUF 文件路径，默认：integrations/audiocpp/weights/base.gguf")
    p.add_argument("--dtype", default="f16", choices=["f16", "bf16", "q8_0", "orig"],
                   help="权重存储精度，默认 f16")
    p.add_argument("--lora-checkpoint", default=None,
                   help="可选：先合并本项目 LoRA checkpoint，再生成独立 GGUF")
    p.add_argument("--quantize-q8", action="store_true",
                   help="转换后额外自动生成 Q8_0 量化文件（<output>-q8_0.gguf）")
    p.add_argument("--staging-dir", default=None,
                   help="Staging 目录（用于存放中间 Safetensors 文件）。"
                        "默认使用系统临时目录，转换后自动清理")
    p.add_argument("--keep-staging", action="store_true",
                   help="转换后保留 staging 目录，方便调试")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    convert(args)

#!/usr/bin/env python3
"""IndexTTS2 audio.cpp 独立命令行推理工具。

无需启动 Gradio WebUI 或浏览器，直接通过命令行执行高品质本地推理。
支持使用 HTTP Server 模式或直接调用 audiocpp_cli.exe 模式。

用法示例：
    # 1. 基础单句合成
    python scripts/infer_audiocpp.py \\
        --text "你好，这是使用 audio.cpp 高性能推理的 IndexTTS2 语音。" \\
        --voice-ref assets/reference.wav \\
        --out outputs/test.wav

    # 2. 情感描述文本控制
    python scripts/infer_audiocpp.py \\
        --text "真不敢相信，我们竟然成功做到了！" \\
        --voice-ref assets/reference.wav \\
        --emotion-text "极度兴奋与惊喜" \\
        --emotion-alpha 0.8 \\
        --out outputs/excited.wav

    # 3. 8维情感向量控制 (喜悦, 愤怒, 悲伤, 恐惧, 厌恶, 沮丧, 惊奇, 平静)
    python scripts/infer_audiocpp.py \\
        --text "今天的天气真好，心情非常舒畅。" \\
        --voice-ref assets/reference.wav \\
        --emotion-vector "0.9,0.0,0.0,0.0,0.0,0.0,0.1,0.2" \\
        --out outputs/joy.wav

    # 4. 语速调节（duration_factor < 1 加快，> 1 减慢）
    python scripts/infer_audiocpp.py \\
        --text "这是一段快速播报的提示音。" \\
        --voice-ref assets/reference.wav \\
        --duration-factor 0.85 \\
        --out outputs/fast.wav
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# 确保能找到本项目的 scripts 目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audiocpp_bridge import AudioCppBridge

log = logging.getLogger("infer_audiocpp")


def parse_vector(vec_str: str | None) -> list[float] | None:
    if not vec_str:
        return None
    try:
        parts = [float(x.strip()) for x in vec_str.split(",")]
        if len(parts) != 8:
            raise ValueError(f"情感向量必须包含 8 个数值，当前提供了 {len(parts)} 个")
        return parts
    except Exception as e:
        raise ValueError(f"解析情感向量失败 ('{vec_str}'): {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IndexTTS2 audio.cpp 独立命令行推理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 输入与输出
    parser.add_argument("--text", type=str, default=None, help="待合成文本")
    parser.add_argument("--text-file", type=Path, default=None, help="待合成文本文件（UTF-8）")
    parser.add_argument("--voice-ref", type=Path, default=Path("assets/reference.wav"),
                        help="说话人参考音频路径（WAV），默认 assets/reference.wav")
    parser.add_argument("--out", type=Path, default=Path("outputs/audiocpp_output.wav"),
                        help="输出音频路径，默认 outputs/audiocpp_output.wav")
    parser.add_argument("--language", default="zh", choices=["zh", "en", "ja", "es", "ar"],
                        help="语言代码，默认 zh")

    # 模型与后端
    parser.add_argument("--model", type=Path, default=None,
                        help="GGUF 模型权重文件路径（默认自动寻找 base-q8_0.gguf 或 base.gguf）")
    parser.add_argument("--server-exe", type=Path,
                        default=Path("integrations/audiocpp/bin/audiocpp_server.exe"),
                        help="audiocpp_server.exe 路径")
    parser.add_argument("--cli-exe", type=Path,
                        default=Path("integrations/audiocpp/bin/audiocpp_cli.exe"),
                        help="audiocpp_cli.exe 路径")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu", "vulkan", "metal", "hip"],
                        help="计算后端，默认 cuda")
    parser.add_argument("--mode", default="server", choices=["server", "cli"],
                        help="执行模式：server（常驻 HTTP 懒加载，推荐批量推理）或 cli（直接调用命令行二进制）")
    parser.add_argument("--port", type=int, default=8080, help="Server 端口")
    parser.add_argument("--lora", default=None, metavar="NAME=PATH",
                        help="可选的运行时 LoRA adapter，格式 name=path.safetensors")

    # IndexTTS2 情感与语速控制
    parser.add_argument("--emotion-text", type=str, default=None,
                        help="情感描述文本（如：'特别开心'，'愤怒的质问'）")
    parser.add_argument("--emotion-audio", type=Path, default=None,
                        help="情感参考音频路径（WAV）")
    parser.add_argument("--emotion-vector", type=str, default=None,
                        help="8 维情感向量，逗号分隔（喜悦,愤怒,悲伤,恐惧,厌恶,沮丧,惊奇,平静），如 '1,0,0,0,0,0,0,0'")
    parser.add_argument("--emotion-alpha", type=float, default=1.0,
                        help="情感控制强度 [0.0 - 1.0]，默认 1.0")
    parser.add_argument("--random-emotion", action="store_true",
                        help="启用随机情感采样")
    parser.add_argument("--duration-factor", type=float, default=1.0,
                        help="语速调节倍率（>1 变慢，<1 变快），默认 1.0")
    parser.add_argument("--interval-silence-ms", type=int, default=200,
                        help="分段标点间的静音间隔（毫秒），默认 200")

    # 生成采样超参数
    parser.add_argument("--temperature", type=float, default=0.8, help="GPT 采样温度")
    parser.add_argument("--top-p", type=float, default=0.8, help="GPT top-p 采样阈值")
    parser.add_argument("--top-k", type=int, default=30, help="GPT top-k 采样")
    parser.add_argument("--num-beams", type=int, default=3, help="GPT beam count")
    parser.add_argument("--repetition-penalty", type=float, default=10.0, help="GPT 重复惩罚")
    parser.add_argument("--length-penalty", type=float, default=0.0, help="GPT 长度惩罚")
    parser.add_argument("--max-mel-tokens", type=int, default=1500, help="最大 mel token 数")

    args = parser.parse_args()

    # 获取待合成文本
    text = args.text
    if text is None and args.text_file:
        if not args.text_file.is_file():
            print(f"错误: 文本文件不存在: {args.text_file}", file=sys.stderr)
            return 1
        text = args.text_file.read_text(encoding="utf-8").strip()

    if not text or not text.strip():
        print("错误: 请通过 --text 或 --text-file 提供待合成文本。", file=sys.stderr)
        return 1

    # 确定模型路径
    model_path = args.model
    if model_path is None:
        candidates = [
            PROJECT_ROOT / "integrations" / "audiocpp" / "weights" / "base-q8_0.gguf",
            PROJECT_ROOT / "integrations" / "audiocpp" / "weights" / "base.gguf",
        ]
        for c in candidates:
            if c.is_file():
                model_path = c
                break
        if model_path is None:
            model_path = candidates[0]

    # 解析 LoRA
    lora_map = {}
    active_lora = None
    if args.lora:
        if "=" in args.lora:
            name, p = args.lora.split("=", 1)
            lora_map[name.strip()] = Path(p.strip()).resolve()
            active_lora = name.strip()
        else:
            print("错误: --lora 格式需为 name=path", file=sys.stderr)
            return 1

    # 解析情感向量
    try:
        emo_vec = parse_vector(args.emotion_vector)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    log.info("初始化 IndexTTS2 audio.cpp 桥接层...")
    try:
        bridge = AudioCppBridge(
            server_exe=args.server_exe,
            model_path=model_path,
            family="index_tts2",
            lora_map=lora_map,
            port=args.port,
            backend=args.backend,
            cli_exe=args.cli_exe,
        )
    except Exception as e:
        print(f"初始化失败: {e}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"开始合成文本: '{text[:50]}{'...' if len(text) > 50 else ''}'")

    try:
        if args.mode == "cli":
            wav_bytes = bridge.generate_via_cli(
                text=text,
                voice_ref=args.voice_ref,
                output_wav=args.out,
                language=args.language,
                lora=active_lora,
                emotion=args.emotion_text,
                emotion_audio=args.emotion_audio,
                emotion_alpha=args.emotion_alpha,
                emotion_vector=emo_vec,
                duration_factor=args.duration_factor,
                interval_silence_ms=args.interval_silence_ms,
                use_random_emotion=args.random_emotion,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                max_mel_tokens=args.max_mel_tokens,
            )
        else:
            wav_bytes = bridge.generate(
                text=text,
                voice_ref=args.voice_ref,
                language=args.language,
                lora=active_lora,
                emotion=args.emotion_text,
                emotion_audio=args.emotion_audio,
                emotion_alpha=args.emotion_alpha,
                emotion_vector=emo_vec,
                duration_factor=args.duration_factor,
                interval_silence_ms=args.interval_silence_ms,
                use_random_emotion=args.random_emotion,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                max_mel_tokens=args.max_mel_tokens,
            )
            args.out.write_bytes(wav_bytes)

        elapsed = time.perf_counter() - t0
        size_kb = args.out.stat().st_size / 1024
        print(f"\n✅ 合成成功！耗时: {elapsed:.2f}s | 输出: {args.out} ({size_kb:.1f} KB)")
        return 0

    except Exception as e:
        print(f"\n❌ 推理失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())

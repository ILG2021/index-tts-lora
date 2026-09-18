"""Gradio inference UI for the base IndexTTS2 model or one LoRA checkpoint."""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = PROJECT_ROOT / "vendor" / "index-tts"
if not UPSTREAM_ROOT.is_dir():
    raise RuntimeError("Missing vendor/index-tts. Run scripts/windows/setup.ps1 first.")
sys.path.insert(0, str(UPSTREAM_ROOT))

import gradio as gr
import torch
from indextts.infer_v2 import IndexTTS2
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from trainers.lora_state import validate_adapter_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS2 LoRA WebUI")
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--config", default="checkpoints/config.yaml")
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--root-path",
        default="",
        help="URL subpath used when serving Gradio behind a reverse proxy.",
    )
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def apply_lora(engine: IndexTTS2, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("LoRA checkpoint must contain a state dictionary.")
    if not isinstance(checkpoint.get("adapter"), Mapping) or not checkpoint["adapter"]:
        raise ValueError("Checkpoint does not contain IndexTTS2 LoRA adapter weights.")
    metadata = checkpoint.get("lora")
    if metadata is None:
        extra = checkpoint.get("extra", {})
        if not isinstance(extra, dict):
            raise ValueError("Checkpoint contains invalid extra metadata.")
        metadata = extra.get("lora", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint contains invalid LoRA configuration metadata.")
    if not all(key in metadata for key in ("r", "alpha", "target_modules")):
        raise ValueError("Checkpoint lacks required LoRA configuration metadata.")
    targets = metadata["target_modules"]
    if (
        not isinstance(targets, (list, tuple, set))
        or not targets
        or not all(isinstance(target, str) and target.strip() for target in targets)
    ):
        raise ValueError("Checkpoint contains invalid LoRA target modules.")
    rank = int(metadata["r"])
    alpha = int(metadata["alpha"])
    if rank < 1 or alpha < 1:
        raise ValueError("Checkpoint contains an invalid LoRA rank or alpha.")
    adapted = get_peft_model(
        engine.gpt.gpt,
        LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                   r=rank, lora_alpha=alpha, lora_dropout=0.0,
                   target_modules=list(targets), bias="none"),
    )
    validate_adapter_state(get_peft_model_state_dict(adapted), checkpoint["adapter"])
    result = set_peft_model_state_dict(adapted, checkpoint["adapter"])
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"Unexpected LoRA keys: {result.unexpected_keys}")
    merged = adapted.merge_and_unload().eval()
    engine.gpt.gpt = merged
    engine.gpt.inference_model.transformer = merged
    if engine.gpt.inference_model.transformer is not engine.gpt.gpt:
        raise RuntimeError("LoRA merge did not update the IndexTTS2 inference transformer.")


engine: IndexTTS2 | None = None

EMOTION_MODES = [
    "与音色参考音频相同",
    "使用情感参考音频",
    "使用情感向量控制",
    "使用情感描述文本控制",
]


def emotion_mode_visibility(mode: str):
    return (
        gr.update(visible=mode == EMOTION_MODES[1]),
        gr.update(visible=mode == EMOTION_MODES[2]),
        gr.update(visible=mode == EMOTION_MODES[3]),
        gr.update(visible=mode in (EMOTION_MODES[1], EMOTION_MODES[2])),
    )


def synthesize(
    speaker_audio,
    text,
    emotion_mode,
    emotion_audio,
    emotion_alpha,
    emotion_text,
    emotion_random,
    joy,
    anger,
    sadness,
    fear,
    disgust,
    depression,
    surprise,
    calm,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    max_text_tokens,
    interval_silence,
    progress=gr.Progress(),
):
    if engine is None:
        raise gr.Error("IndexTTS2 尚未初始化。")
    if not speaker_audio or not Path(speaker_audio).is_file():
        raise gr.Error("请上传有效的说话人参考音频。")
    if not text or not text.strip():
        raise gr.Error("请输入待合成文本。")
    if emotion_mode not in EMOTION_MODES:
        raise gr.Error("请选择有效的情感控制方式。")
    if emotion_mode == EMOTION_MODES[1] and (
        not emotion_audio or not Path(emotion_audio).is_file()
    ):
        raise gr.Error("使用情感参考音频时，请上传有效的情感音频。")

    emotion_vector = None
    use_emotion_text = emotion_mode == EMOTION_MODES[3]
    if emotion_mode == EMOTION_MODES[0]:
        emotion_audio = None
    elif emotion_mode == EMOTION_MODES[2]:
        emotion_audio = None
        emotion_vector = engine.normalize_emo_vec(
            [joy, anger, sadness, fear, disgust, depression, surprise, calm],
            apply_bias=True,
        )
    elif use_emotion_text:
        emotion_audio = None

    output = PROJECT_ROOT / "outputs" / f"indextts2_{time.time_ns()}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    engine.gr_progress = progress
    generated = engine.infer(
        spk_audio_prompt=speaker_audio,
        text=text.strip(),
        output_path=str(output),
        emo_audio_prompt=emotion_audio or None,
        emo_alpha=float(emotion_alpha),
        emo_vector=emotion_vector,
        use_emo_text=use_emotion_text,
        emo_text=(emotion_text or "").strip() or None,
        use_random=bool(emotion_random),
        max_text_tokens_per_segment=int(max_text_tokens),
        top_p=float(top_p),
        top_k=int(top_k) if int(top_k) > 0 else None,
        temperature=float(temperature),
        length_penalty=float(length_penalty),
        num_beams=int(num_beams),
        repetition_penalty=float(repetition_penalty),
        max_mel_tokens=int(max_mel_tokens),
        interval_silence=int(interval_silence),
        verbose=True,
    )
    if not generated or not output.is_file():
        raise gr.Error("IndexTTS2 未生成有效音频，请查看终端错误信息。")
    return str(output)


def build_demo() -> gr.Blocks:
    if engine is None:
        raise RuntimeError("IndexTTS2 must be initialized before building the WebUI.")
    title = "IndexTTS2 v2 · LoRA / Base"
    max_text_limit = int(engine.cfg.gpt.max_text_tokens)
    max_mel_limit = int(engine.cfg.gpt.max_mel_tokens)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"# {title}\n"
            "使用 IndexTTS2 v2 原生情感控制与生成参数；启动时可加载基础模型或 LoRA。"
        )
        with gr.Row():
            speaker = gr.Audio(
                label="音色参考音频",
                sources=["upload", "microphone"],
                type="filepath",
            )
            result = gr.Audio(label="生成结果", type="filepath")
        text = gr.TextArea(label="目标文本", lines=5, placeholder="请输入需要合成的文本")

        with gr.Accordion("IndexTTS2 情感控制", open=True):
            emotion_mode = gr.Radio(
                EMOTION_MODES,
                value=EMOTION_MODES[0],
                label="情感控制方式",
            )
            with gr.Group(visible=False) as emotion_audio_group:
                emotion_audio = gr.Audio(label="情感参考音频", type="filepath")
            with gr.Group(visible=False) as emotion_vector_group:
                with gr.Row():
                    joy = gr.Slider(0, 1, 0, step=0.05, label="喜")
                    anger = gr.Slider(0, 1, 0, step=0.05, label="怒")
                    sadness = gr.Slider(0, 1, 0, step=0.05, label="哀")
                    fear = gr.Slider(0, 1, 0, step=0.05, label="惧")
                with gr.Row():
                    disgust = gr.Slider(0, 1, 0, step=0.05, label="厌恶")
                    depression = gr.Slider(0, 1, 0, step=0.05, label="低落")
                    surprise = gr.Slider(0, 1, 0, step=0.05, label="惊喜")
                    calm = gr.Slider(0, 1, 0, step=0.05, label="平静")
                emotion_random = gr.Checkbox(False, label="情感随机采样")
            with gr.Group(visible=False) as emotion_text_group:
                emotion_text = gr.Textbox(
                    label="情感描述文本",
                    placeholder="留空时使用目标文本分析情感",
                )
            with gr.Group(visible=False) as emotion_alpha_group:
                emotion_alpha = gr.Slider(0, 1, 0.65, step=0.01, label="情感权重")

            emotion_mode.change(
                emotion_mode_visibility,
                inputs=emotion_mode,
                outputs=[
                    emotion_audio_group,
                    emotion_vector_group,
                    emotion_text_group,
                    emotion_alpha_group,
                ],
            )

        with gr.Accordion("高级生成参数", open=False):
            with gr.Row():
                temperature = gr.Slider(0.1, 2, 0.8, step=0.1, label="temperature")
                top_p = gr.Slider(0, 1, 0.8, step=0.01, label="top_p")
                top_k = gr.Slider(0, 100, 30, step=1, label="top_k")
            with gr.Row():
                num_beams = gr.Slider(1, 10, 3, step=1, label="num_beams")
                repetition_penalty = gr.Number(10.0, label="repetition_penalty")
                length_penalty = gr.Number(0.0, label="length_penalty")
            with gr.Row():
                max_mel_tokens = gr.Slider(
                    50, max_mel_limit, min(1500, max_mel_limit), step=10,
                    label="最大语音 Token",
                )
                max_text_tokens = gr.Slider(
                    20, max_text_limit, min(120, max_text_limit), step=2,
                    label="每段最大文本 Token",
                )
                interval_silence = gr.Slider(
                    0, 2000, 200, step=10, label="分段静音（毫秒）"
                )

        generate = gr.Button("生成语音", variant="primary")
        generate.click(
            synthesize,
            inputs=[
                speaker, text, emotion_mode, emotion_audio, emotion_alpha,
                emotion_text, emotion_random, joy, anger, sadness, fear,
                disgust, depression, surprise, calm, top_p, top_k,
                temperature, length_penalty, num_beams, repetition_penalty,
                max_mel_tokens, max_text_tokens, interval_silence,
            ],
            outputs=result,
        )
    return demo

if __name__ == "__main__":
    args = parse_args()
    for required in (Path(args.config), Path(args.model_dir) / "gpt.pth"):
        if not required.is_file():
            raise FileNotFoundError(f"Required IndexTTS2 file not found: {required}")
    if args.lora_checkpoint and not Path(args.lora_checkpoint).is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {args.lora_checkpoint}")
    engine = IndexTTS2(cfg_path=args.config, model_dir=args.model_dir,
                      device=args.device,
                      use_fp16=torch.cuda.is_available() and (args.device is None or args.device.startswith("cuda")),
                      use_cuda_kernel=False)
    if args.lora_checkpoint:
        apply_lora(engine, args.lora_checkpoint)
    print(f"Active weights: {args.lora_checkpoint or 'official base model'}")
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host,
                                                   server_port=args.port,
                                                   root_path=args.root_path or None,
                                                   share=args.share)

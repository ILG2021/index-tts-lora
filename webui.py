"""Gradio inference UI for the base IndexTTS2 model or one LoRA checkpoint."""
from __future__ import annotations

import argparse
import sys
import time
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
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def apply_lora(engine: IndexTTS2, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "adapter" not in checkpoint:
        raise ValueError("Checkpoint does not contain IndexTTS2 LoRA adapter weights.")
    metadata = checkpoint.get("lora", checkpoint.get("extra", {}).get("lora", {}))
    if not all(key in metadata for key in ("r", "alpha", "target_modules")):
        raise ValueError("Checkpoint lacks required LoRA configuration metadata.")
    targets = metadata.get("target_modules", ["c_attn", "c_proj", "c_fc"])
    adapted = get_peft_model(
        engine.gpt.gpt,
        LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                   r=int(metadata.get("r", 16)),
                   lora_alpha=int(metadata.get("alpha", 32)),
                   lora_dropout=0.0, target_modules=targets, bias="none"),
    )
    validate_adapter_state(get_peft_model_state_dict(adapted), checkpoint["adapter"])
    result = set_peft_model_state_dict(adapted, checkpoint["adapter"])
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"Unexpected LoRA keys: {result.unexpected_keys}")
    merged = adapted.merge_and_unload().eval()
    engine.gpt.gpt = merged
    engine.gpt.inference_model.transformer = merged


engine = None


def synthesize(speaker_audio, text, emotion_audio, emotion_alpha, use_emotion_text,
               emotion_text, max_text_tokens, top_p, top_k, temperature, num_beams,
               interval_silence):
    if not speaker_audio or not Path(speaker_audio).is_file():
        raise gr.Error("请上传有效的说话人参考音频。")
    if not text or not text.strip():
        raise gr.Error("请输入待合成文本。")
    output = PROJECT_ROOT / "outputs" / f"indextts2_{time.time_ns()}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    engine.infer(
        spk_audio_prompt=speaker_audio,
        text=text.strip(),
        output_path=str(output),
        emo_audio_prompt=emotion_audio or None,
        emo_alpha=float(emotion_alpha),
        use_emo_text=bool(use_emotion_text),
        emo_text=(emotion_text or "").strip() or None,
        max_text_tokens_per_segment=int(max_text_tokens),
        top_p=float(top_p), top_k=int(top_k), temperature=float(temperature),
        num_beams=int(num_beams), interval_silence=int(interval_silence),
        verbose=True,
    )
    return str(output)


title = "IndexTTS2 LoRA / Base"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}\n启动时选择基础模型或 LoRA；当前权重路径显示在终端。")
    with gr.Row():
        speaker = gr.Audio(label="说话人参考音频", sources=["upload", "microphone"], type="filepath")
        emotion = gr.Audio(label="情绪参考音频（可选）", sources=["upload"], type="filepath")
    text = gr.TextArea(label="合成文本", lines=5)
    with gr.Accordion("情绪与生成参数", open=False):
        emotion_alpha = gr.Slider(0, 1, value=1, step=0.05, label="情绪强度")
        use_emotion_text = gr.Checkbox(False, label="从情绪文本提取情绪")
        emotion_text = gr.Textbox(label="情绪文本（可选）")
        max_text_tokens = gr.Slider(20, 600, value=120, step=1, label="每段最大文本 Token")
        top_p = gr.Slider(0.1, 1, value=0.8, step=0.01, label="Top P")
        top_k = gr.Slider(1, 100, value=30, step=1, label="Top K")
        temperature = gr.Slider(0.1, 2, value=1, step=0.05, label="Temperature")
        num_beams = gr.Slider(1, 10, value=3, step=1, label="Beams")
        interval_silence = gr.Slider(0, 2000, value=200, step=10, label="分段静音（毫秒）")
    generate = gr.Button("生成", variant="primary")
    result = gr.Audio(label="生成结果", type="filepath")
    generate.click(synthesize,
                   inputs=[speaker, text, emotion, emotion_alpha, use_emotion_text,
                           emotion_text, max_text_tokens, top_p, top_k, temperature,
                           num_beams, interval_silence], outputs=result)

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
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host,
                                                   server_port=args.port,
                                                   share=args.share)

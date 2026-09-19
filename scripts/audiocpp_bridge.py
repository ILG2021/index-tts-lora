"""audio.cpp server 桥接层 — Python ↔ C++ 懒加载推理代理。

架构参考 MOSS-TTS openmoss 的 clis/moss_tts_openmoss_app.py 模式：
  - 首次推理时启动 audiocpp_server 子进程（懒加载）
  - 后续请求直接 HTTP 调用，不重新加载模型
  - 若目标端口已有兼容 server，直接复用（不重复启动）
  - 正常退出时终止由本进程启动的子 server；外部启动的不受影响

进程关系：
    浏览器 → Gradio(Python, 仅 UI/HTTP) → /tts → audiocpp_server(C++/ggml)

使用示例：

    from scripts.audiocpp_bridge import AudioCppBridge

    bridge = AudioCppBridge(
        server_exe="integrations/audiocpp/bin/audiocpp_server.exe",
        model_path="integrations/audiocpp/weights/base-q8_0.gguf",
        family="index_tts2",
        lora_map={"speaker-a": "integrations/audiocpp/weights/voices/speaker-a.safetensors"},
        host="127.0.0.1",
        port=8080,
    )
    wav_bytes = bridge.generate(
        text="你好世界",
        voice_ref="reference.wav",
        language="zh",
    )
    with open("output.wav", "wb") as f:
        f.write(wav_bytes)
"""
from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# audiocpp_server 健康检查端点（参考 audio.cpp docs/server.md）
_HEALTH_ENDPOINT = "/health"
# TTS 推理端点（OpenAI 兼容接口）
_TTS_ENDPOINT = "/v1/audio/speech"
# 模型信息端点
_INFO_ENDPOINT = "/v1/models"

# 默认等待 server 就绪的超时（秒）
_DEFAULT_STARTUP_TIMEOUT = 60
# 健康检查轮询间隔（秒）
_HEALTH_POLL_INTERVAL = 0.5
# audio.cpp /v1/audio/speech 对内联 voice_ref 的上限。
_MAX_INLINE_VOICE_REF_BYTES = 5 * 1024 * 1024


class AudioCppBridgeError(RuntimeError):
    """audio.cpp 桥接层异常基类。"""


class ServerStartupError(AudioCppBridgeError):
    """server 启动失败或超时。"""


class InferenceError(AudioCppBridgeError):
    """推理请求失败。"""


def _validate_wav_audio(data: bytes, source: str) -> None:
    """确认响应是至少包含一个音频帧的 WAV，而不只是空 RIFF 头。"""
    if not data:
        raise InferenceError(f"{source} 返回了空响应")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise InferenceError(f"{source} 返回的 WAV 不包含音频帧")
            if wav_file.getframerate() <= 0 or wav_file.getnchannels() <= 0:
                raise InferenceError(f"{source} 返回的 WAV 参数无效")
    except (EOFError, wave.Error) as exc:
        raise InferenceError(f"{source} 返回的内容不是有效 WAV：{exc}") from exc


def _stage_audio_ascii(source: Path) -> Path:
    """复制到 ASCII 临时路径，规避 Windows C++ 窄字符路径的代码页错误。"""
    suffix = source.suffix if source.suffix.isascii() else ".wav"
    handle = tempfile.NamedTemporaryFile(
        prefix="audiocpp-audio-",
        suffix=suffix,
        delete=False,
    )
    staged = Path(handle.name)
    handle.close()
    try:
        shutil.copyfile(source, staged)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _server_voice_ref(source: Path) -> tuple[dict[str, str] | str, Path | None]:
    """优先内联参考 WAV；过大时退回 ASCII 临时路径。"""
    audio_bytes = source.read_bytes()
    if len(audio_bytes) <= _MAX_INLINE_VOICE_REF_BYTES:
        return {
            "type": "base64",
            "data": base64.b64encode(audio_bytes).decode("ascii"),
        }, None
    staged = _stage_audio_ascii(source)
    return str(staged), staged


class AudioCppBridge:
    """IndexTTS2 audio.cpp server 的 Python 桥接层。

    参数：
        server_exe:   audiocpp_server 可执行文件路径
        model_path:   GGUF 模型权重路径
        family:       audio.cpp 模型族（默认 'index_tts2'）
        lora_map:     未融合的运行时 adapter 映射 {name: path}，路径应指向
                      convert_lora_adapter.py 生成的 .safetensors
        host:         server 监听地址
        port:         server 监听端口
        backend:      ggml 计算后端（cuda | cpu | vulkan | metal）
        extra_args:   传给 audiocpp_server 的额外命令行参数列表
        startup_timeout: server 启动超时（秒）
        preload:      True = 初始化时立即启动 server；False = 首次推理时懒加载
    """

    def __init__(
        self,
        server_exe: str | Path,
        model_path: str | Path,
        family: str = "index_tts2",
        lora_map: dict[str, str | Path] | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        backend: str = "cuda",
        extra_args: list[str] | None = None,
        startup_timeout: int = _DEFAULT_STARTUP_TIMEOUT,
        preload: bool = False,
        cli_exe: str | Path | None = None,
    ) -> None:
        self.server_exe = Path(server_exe).resolve()
        self.model_path = Path(model_path).resolve()
        self.family = family
        self.lora_map: dict[str, Path] = {
            name: Path(path).resolve() for name, path in (lora_map or {}).items()
        }
        self.host = host
        self.port = port
        self.backend = backend
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout

        # 定位配套的 audiocpp_cli.exe
        if cli_exe is not None:
            self.cli_exe = Path(cli_exe).resolve()
        else:
            default_cli = self.server_exe.parent / "audiocpp_cli.exe"
            self.cli_exe = default_cli if default_cli.is_file() else None

        self._proc: subprocess.Popen | None = None
        self._config_path: Path | None = None
        self._owned_proc = False  # 本桥接层是否是 server 的启动方
        self._lock = threading.Lock()
        self._base_url = f"http://{host}:{port}"

        # 校验路径
        if not self.server_exe.is_file():
            raise FileNotFoundError(
                f"audiocpp_server 不存在：{self.server_exe}\n"
                "请先运行 integrations/audiocpp/scripts/download_audiocpp.ps1"
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"GGUF 模型不存在：{self.model_path}\n"
                "请先运行 scripts/convert_to_gguf.py"
            )
        for name, path in self.lora_map.items():
            if not name or not path.is_file():
                raise FileNotFoundError(f"LoRA adapter 不存在 [{name}]：{path}")

        # 注册退出清理
        atexit.register(self._atexit_cleanup)

        if preload:
            self.start_server()

    # ── 进程管理 ──────────────────────────────────────────────────────────────

    def _build_server_cmd(self) -> list[str]:
        """构造 audiocpp_server 启动命令。"""
        session_options = {
            f"index_tts2.lora.{name}": str(path)
            for name, path in self.lora_map.items()
        }
        models = [{"id": self.family, "family": self.family,
                   "path": str(self.model_path), "task": "tts", "mode": "offline",
                   "session_options": session_options}]
        config = {
            "host": self.host, "port": self.port, "backend": self.backend,
            "lazy_load": True,
            "max_loaded_models": 1,
            # 多人请求在同一模型实例前无限期排队，不因默认 300 秒等待上限
            # 返回 503。模型内部仍由 BusyGuard 保证串行执行。
            "busy_timeout_ms": 0,
            "models": models,
        }
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="indextts2-audiocpp-",
            encoding="utf-8", delete=False,
        )
        with handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        self._config_path = Path(handle.name)
        cmd = [str(self.server_exe), "--config", str(self._config_path)]
        cmd.extend(self.extra_args)
        return cmd

    def _is_server_healthy(self) -> bool:
        """检查 server 是否响应健康检查。"""
        try:
            r = requests.get(f"{self._base_url}{_HEALTH_ENDPOINT}", timeout=2)
            return r.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _wait_for_health(self, timeout: int) -> None:
        """轮询健康检查，直到 server 就绪或超时。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_server_healthy():
                return
            # 检查子进程是否已异常退出
            if self._proc and self._proc.poll() is not None:
                raise ServerStartupError(
                    f"audiocpp_server 进程提前退出（退出码 {self._proc.returncode}）。"
                    "请检查终端日志。"
                )
            time.sleep(_HEALTH_POLL_INTERVAL)
        raise ServerStartupError(
            f"audiocpp_server 在 {timeout}s 内未就绪（{self._base_url}）。"
        )

    def start_server(self) -> None:
        """启动 audiocpp_server（如果尚未运行）。"""
        with self._lock:
            # 先检查是否已有可用 server（包括外部启动的）
            if self._is_server_healthy():
                log.info("检测到 %s 已有运行中的 server，直接复用", self._base_url)
                return

            cmd = self._build_server_cmd()
            log.info("启动 audiocpp_server：%s", " ".join(cmd))
            self._proc = subprocess.Popen(cmd)
            self._owned_proc = True
            log.info("audiocpp_server PID=%d，等待就绪...", self._proc.pid)

            try:
                self._wait_for_health(self.startup_timeout)
            except ServerStartupError:
                self._proc.kill()
                self._proc = None
                self._owned_proc = False
                raise
            finally:
                if self._config_path and self._config_path.exists():
                    self._config_path.unlink()
                self._config_path = None

            log.info("audiocpp_server 已就绪（%s）", self._base_url)

    def _ensure_running(self) -> None:
        """确保 server 正在运行（首次调用时懒启动）。"""
        if self._is_server_healthy():
            return
        self.start_server()

    def shutdown(self) -> None:
        """终止由本桥接层启动的 audiocpp_server 子进程。"""
        with self._lock:
            if self._proc and self._owned_proc:
                log.info("终止 audiocpp_server（PID=%d）", self._proc.pid)
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                self._proc = None
                self._owned_proc = False

    def _atexit_cleanup(self) -> None:
        """atexit 钩子：进程退出时清理子进程。"""
        try:
            self.shutdown()
        except Exception:
            pass

    # ── 推理接口 ──────────────────────────────────────────────────────────────

    def generate_via_cli(
        self,
        text: str,
        voice_ref: str | Path,
        output_wav: str | Path | None = None,
        language: str = "zh",
        lora: str | None = None,
        lora_scale: float = 1.0,
        emotion: str | None = None,
        emotion_audio: str | Path | None = None,
        emotion_alpha: float = 1.0,
        emotion_vector: list[float] | None = None,
        duration_factor: float = 1.0,
        interval_silence_ms: int = 200,
        use_random_emotion: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        seed: int | None = None,
        extra_args: list[str] | None = None,
    ) -> bytes:
        """直接调用 audiocpp_cli.exe 执行推理（无需 HTTP Server，直接本地执行）。"""
        if not self.cli_exe or not self.cli_exe.is_file():
            raise FileNotFoundError(
                f"audiocpp_cli.exe 不存在（路径：{self.cli_exe}）。"
                "请先运行 integrations/audiocpp/scripts/download_audiocpp.ps1"
            )

        voice_ref_path = Path(voice_ref).resolve()
        if not voice_ref_path.is_file():
            raise ValueError(f"说话人参考音频不存在：{voice_ref_path}")

        staged_voice_ref = _stage_audio_ascii(voice_ref_path)
        staged_emotion_audio = None
        if emotion_audio:
            emotion_source = Path(emotion_audio).resolve()
            if not emotion_source.is_file():
                staged_voice_ref.unlink(missing_ok=True)
                raise ValueError(f"情感参考音频不存在：{emotion_source}")
            staged_emotion_audio = _stage_audio_ascii(emotion_source)

        temp_out = False
        if output_wav is None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            out_path = Path(tmp.name)
            temp_out = True
        else:
            out_path = Path(output_wav).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.cli_exe),
            "--task", "clon",
            "--family", "index_tts2",
            "--model", str(self.model_path),
            "--backend", self.backend,
            "--language", language,
            "--voice-ref", str(staged_voice_ref),
            "--text", text,
            "--out", str(out_path),
            "--temperature", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--num-beams", str(num_beams),
            "--repetition-penalty", str(repetition_penalty),
            "--max-tokens", str(max_mel_tokens),
        ]

        if lora:
            if lora not in self.lora_map:
                raise ValueError(
                    f"未知的 LoRA 名称 '{lora}'。可用：{list(self.lora_map.keys())}"
                )
        for name, path in self.lora_map.items():
            cmd.extend(["--session-option", f"index_tts2.lora.{name}={path}"])
        if lora:
            cmd.extend(["--request-option", f"index_tts2.lora={lora}"])
            cmd.extend(["--request-option", f"index_tts2.lora_scale={float(lora_scale)}"])

        if staged_emotion_audio:
            cmd.extend(["--audio", str(staged_emotion_audio)])

        if emotion:
            cmd.extend(["--emotion", emotion])
            cmd.extend(["--request-option", "use_emotion_text=true"])

        if emotion_vector is not None:
            vec_str = ",".join(f"{float(v):.4f}" for v in emotion_vector)
            cmd.extend(["--request-option", f"emotion_vector={vec_str}"])

        if emotion_alpha != 1.0:
            cmd.extend(["--request-option", f"emotion_alpha={emotion_alpha:.2f}"])

        if duration_factor != 1.0:
            cmd.extend(["--request-option", f"duration_factor={duration_factor:.2f}"])

        if interval_silence_ms != 200:
            cmd.extend(["--request-option", f"interval_silence_ms={interval_silence_ms}"])

        if use_random_emotion:
            cmd.extend(["--request-option", "use_random_emotion=true"])

        if length_penalty != 0.0:
            cmd.extend(["--request-option", f"length_penalty={length_penalty:.2f}"])

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        if extra_args:
            cmd.extend(extra_args)

        log.info("运行 audiocpp_cli：%s", " ".join(cmd))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                if temp_out and out_path.exists():
                    out_path.unlink()
                raise InferenceError(f"audiocpp_cli 执行失败 (code {res.returncode}): {res.stderr}")

            if not out_path.is_file() or out_path.stat().st_size == 0:
                if temp_out and out_path.exists():
                    out_path.unlink()
                raise InferenceError("audiocpp_cli 执行完成但未生成有效音频文件")

            wav_data = out_path.read_bytes()
            if temp_out:
                out_path.unlink()
            _validate_wav_audio(wav_data, "audiocpp_cli")
            return wav_data
        finally:
            staged_voice_ref.unlink(missing_ok=True)
            if staged_emotion_audio:
                staged_emotion_audio.unlink(missing_ok=True)

    def generate(
        self,
        text: str,
        voice_ref: str | Path,
        language: str = "zh",
        lora: str | None = None,
        lora_scale: float = 1.0,
        emotion: str | None = None,
        emotion_audio: str | Path | None = None,
        emotion_alpha: float = 1.0,
        emotion_vector: list[float] | None = None,
        duration_factor: float = 1.0,
        interval_silence_ms: int = 200,
        use_random_emotion: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        fallback_to_cli: bool = True,
        **extra_request_options: Any,
    ) -> bytes:
        """调用 audiocpp_server 进行 TTS 推理，返回 WAV 字节流。

        支持 IndexTTS2 全部原生情感控制与语速参数。
        若 HTTP 失败且 fallback_to_cli=True，可自动降级至本地 audiocpp_cli。
        """
        voice_ref_path = Path(voice_ref).resolve()
        if not voice_ref_path.is_file():
            raise ValueError(f"说话人参考音频不存在：{voice_ref_path}")

        if lora:
            if lora not in self.lora_map:
                raise ValueError(
                    f"未知的 LoRA 名称 '{lora}'。可用：{list(self.lora_map.keys())}"
                )

        emo_path = None
        if emotion_audio:
            emo_path = Path(emotion_audio).resolve()
            if not emo_path.is_file():
                raise ValueError(f"情感参考音频不存在：{emo_path}")

        voice_ref_payload, staged_voice_ref = _server_voice_ref(voice_ref_path)
        staged_emo_path = _stage_audio_ascii(emo_path) if emo_path else None

        try:
            self._ensure_running()

            request_body: dict[str, Any] = {
                "model": self.family,
                "input": text,
                # /v1/audio/speech 从顶层 language 构造 Transcript；同时保留
                # options.language 以兼容只读取 request option 的模型族。
                "language": language,
                "voice_ref": voice_ref_payload,
                "response_format": "wav",
            }

            opts: dict[str, Any] = {
                "language": language,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "max_tokens": max_mel_tokens,
                "emotion_alpha": emotion_alpha,
                "duration_factor": duration_factor,
                "interval_silence_ms": interval_silence_ms,
                "use_random_emotion": use_random_emotion,
            }

            if emotion:
                opts["emotion_text"] = emotion
                opts["use_emotion_text"] = True

            if emotion_vector is not None:
                opts["emotion_vector"] = [float(v) for v in emotion_vector]

            if lora:
                opts["index_tts2.lora"] = lora
                opts["index_tts2.lora_scale"] = float(lora_scale)

            opts.update(extra_request_options)
            request_body["options"] = opts
            if staged_emo_path:
                request_body["audio"] = str(staged_emo_path)
            response = requests.post(
                # IndexTTS2 长文本/CPU 推理可能远超两分钟。这里不设置客户端
                # 总超时，由 Gradio 队列一直等待 server 返回，避免误触发 CLI
                # 回退后与仍在运行的 server 同时争抢 GPU。
                f"{self._base_url}{_TTS_ENDPOINT}", json=request_body, timeout=None,
            )

            if response.status_code == 200:
                _validate_wav_audio(response.content, "audiocpp_server")
                return response.content

            error_text = response.content.decode("utf-8", errors="replace")
            raise InferenceError(f"HTTP {response.status_code}: {error_text[:300]}")

        except Exception as e:
            if fallback_to_cli and self.cli_exe and self.cli_exe.is_file():
                log.warning("audiocpp_server 推理失败 (%s)，降级尝试 audiocpp_cli.exe...", e)
                return self.generate_via_cli(
                    text=text,
                    voice_ref=voice_ref_path,
                    language=language,
                    lora=lora,
                    lora_scale=lora_scale,
                    emotion=emotion,
                    emotion_audio=emotion_audio,
                    emotion_alpha=emotion_alpha,
                    emotion_vector=emotion_vector,
                    duration_factor=duration_factor,
                    interval_silence_ms=interval_silence_ms,
                    use_random_emotion=use_random_emotion,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    max_mel_tokens=max_mel_tokens,
                )
            raise InferenceError(f"audiocpp 推理失败：{e}") from e
        finally:
            if staged_voice_ref:
                staged_voice_ref.unlink(missing_ok=True)
            if staged_emo_path:
                staged_emo_path.unlink(missing_ok=True)

    def health_check(self) -> dict[str, Any]:
        """返回 server 健康状态信息。"""
        try:
            r = requests.get(f"{self._base_url}{_HEALTH_ENDPOINT}", timeout=5)
            r.raise_for_status()
            return r.json() if r.content else {"status": "ok"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def list_models(self) -> dict[str, Any]:
        """返回 server 已加载的模型信息（含 LoRA 列表）。"""
        try:
            r = requests.get(f"{self._base_url}{_INFO_ENDPOINT}", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    @property
    def is_running(self) -> bool:
        """server 是否当前可访问。"""
        return self._is_server_healthy()

    @property
    def base_url(self) -> str:
        return self._base_url

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return (
            f"AudioCppBridge(family={self.family!r}, "
            f"model={self.model_path.name!r}, "
            f"url={self._base_url!r}, "
            f"status={status!r})"
        )

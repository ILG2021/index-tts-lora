"""IndexTTS2 audio.cpp GGUF 推理流程端到端测试。

分为两类：
  单元测试（无需模型权重 / audio.cpp 二进制）：
    - test_parse_lora_args            — webui.py --lora 参数解析
    - test_bridge_init_missing_files  — AudioCppBridge 校验缺失文件
    - test_convert_check_model_dir    — convert_to_gguf 输入校验
    - test_lora_checkpoint_format     — LoRA checkpoint 格式校验

  集成测试（需要 audio.cpp 二进制 + GGUF 权重，用 -k integration 运行）：
    - test_integration_audiocpp_health      — server 启动 + 健康检查
    - test_integration_generate             — 基础 TTS 推理
    - test_integration_generate_with_lora   — LoRA adapter TTS 推理

用法：
    # 仅单元测试（无需下载任何模型）
    python -m pytest tests/test_gguf_pipeline.py -v

    # 集成测试（需要已运行 setup_audiocpp.ps1）
    python -m pytest tests/test_gguf_pipeline.py -v -k integration

    # 完整端到端
    python -m pytest tests/test_gguf_pipeline.py -v --timeout=120
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

# 将 scripts/ 加入路径以便直接导入桥接层
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _test_wav_bytes(frames: int = 16) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── 常量：集成测试所需路径 ──────────────────────────────────────────────────
_AUDIOCPP_EXE = PROJECT_ROOT / "integrations" / "audiocpp" / "bin" / "audiocpp_server.exe"
_GGUF_Q8      = PROJECT_ROOT / "integrations" / "audiocpp" / "weights" / "base-q8_0.gguf"
_GGUF_F16     = PROJECT_ROOT / "integrations" / "audiocpp" / "weights" / "base.gguf"
_REFERENCE_WAV = PROJECT_ROOT / "assets" / "reference.wav"

# ─────────────────────────────────────────────────────────────────────────────
# 单元测试（无依赖）
# ─────────────────────────────────────────────────────────────────────────────


class TestParseLoraArgs(unittest.TestCase):
    """webui.py 中 _parse_lora_args 的参数解析测试。"""

    def setUp(self):
        # 动态导入以避免 gradio 等依赖在 CI 中不可用时的 ImportError
        sys.path.insert(0, str(PROJECT_ROOT))

    def _import_fn(self):
        # 从 webui 模块导入（不执行 __main__ 块）
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "webui_module", PROJECT_ROOT / "webui.py"
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        return getattr(mod, "_parse_lora_args", None)

    def test_empty_returns_empty_dict(self):
        fn = self._import_fn()
        if fn is None:
            self.skipTest("_parse_lora_args 未找到（可能 gradio 未安装）")
        self.assertEqual(fn(None), {})
        self.assertEqual(fn([]), {})

    def test_single_lora(self):
        fn = self._import_fn()
        if fn is None:
            self.skipTest("_parse_lora_args 未找到")
        result = fn(["speaker-a=weights/loras/speaker-a.gguf"])
        self.assertEqual(result, {"speaker-a": "weights/loras/speaker-a.gguf"})

    def test_multiple_loras(self):
        fn = self._import_fn()
        if fn is None:
            self.skipTest("_parse_lora_args 未找到")
        result = fn(["a=path/a.gguf", "b=path/b.gguf"])
        self.assertEqual(result, {"a": "path/a.gguf", "b": "path/b.gguf"})

    def test_invalid_format_raises(self):
        fn = self._import_fn()
        if fn is None:
            self.skipTest("_parse_lora_args 未找到")
        with self.assertRaises(ValueError):
            fn(["invalid-no-equals-sign"])


class TestBridgeInitValidation(unittest.TestCase):
    """AudioCppBridge 初始化校验（不实际启动进程）。"""

    def test_missing_exe_raises(self):
        try:
            from audiocpp_bridge import AudioCppBridge
        except ImportError:
            self.skipTest("audiocpp_bridge 不可导入（依赖 requests）")

        with self.assertRaises(FileNotFoundError) as ctx:
            AudioCppBridge(
                server_exe="/nonexistent/audiocpp_server.exe",
                model_path="/nonexistent/model.gguf",
            )
        self.assertIn("audiocpp_server", str(ctx.exception).lower())

    def test_missing_model_raises(self):
        try:
            from audiocpp_bridge import AudioCppBridge
        except ImportError:
            self.skipTest("audiocpp_bridge 不可导入")

        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            fake_exe = f.name

        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                AudioCppBridge(
                    server_exe=fake_exe,
                    model_path="/nonexistent/model.gguf",
                )
            self.assertIn("gguf", str(ctx.exception).lower())
        finally:
            os.unlink(fake_exe)

    def test_repr_contains_family(self):
        """repr() 不启动进程，但需要 exe 和 model 都存在。"""
        try:
            from audiocpp_bridge import AudioCppBridge
        except ImportError:
            self.skipTest("audiocpp_bridge 不可导入")

        with (
            tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f1,
            tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f2,
        ):
            exe, model = f1.name, f2.name

        try:
            bridge = AudioCppBridge(server_exe=exe, model_path=model)
            r = repr(bridge)
            self.assertIn("index_tts2", r)
            self.assertIn("stopped", r)
        finally:
            bridge.shutdown()
            os.unlink(exe)
            os.unlink(model)


class TestLoraSwitching(unittest.TestCase):
    """测试 AudioCppBridge 中 LoRA adapter 映射与切换逻辑。"""

    def setUp(self):
        try:
            from audiocpp_bridge import AudioCppBridge
            self.BridgeClass = AudioCppBridge
        except ImportError:
            self.skipTest("audiocpp_bridge 不可导入")

        self.fake_exe = tempfile.NamedTemporaryFile(suffix=".exe", delete=False)
        self.fake_model = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        self.fake_lora_a = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        self.fake_lora_b = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        self.fake_exe.close()
        self.fake_model.close()
        self.fake_lora_a.close()
        self.fake_lora_b.close()

    def tearDown(self):
        for path in (self.fake_exe.name, self.fake_model.name, self.fake_lora_a.name, self.fake_lora_b.name):
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_lora_map_build_server_config(self):
        """所有 runtime adapter 应挂在同一个基础模型 session 上。"""
        lora_map = {
            "speaker-a": self.fake_lora_a.name,
            "speaker-b": self.fake_lora_b.name,
        }
        bridge = self.BridgeClass(
            server_exe=self.fake_exe.name,
            model_path=self.fake_model.name,
            lora_map=lora_map,
        )
        try:
            cmd = bridge._build_server_cmd()
            self.assertEqual(cmd[1], "--config")
            self.assertNotIn("--lora", cmd)
            import json
            config = json.loads(Path(cmd[2]).read_text(encoding="utf-8"))
            self.assertEqual([m["id"] for m in config["models"]], ["index_tts2"])
            self.assertEqual(config["busy_timeout_ms"], 0)
            self.assertEqual(
                config["models"][0]["session_options"],
                {
                    "index_tts2.lora.speaker-a": str(Path(self.fake_lora_a.name).resolve()),
                    "index_tts2.lora.speaker-b": str(Path(self.fake_lora_b.name).resolve()),
                },
            )
            self.assertEqual(config["max_loaded_models"], 1)
        finally:
            bridge.shutdown()

    def test_unknown_lora_raises_in_generate(self):
        """测试请求未注册的 LoRA 名称时抛出 ValueError。"""
        lora_map = {"speaker-a": self.fake_lora_a.name}
        bridge = self.BridgeClass(
            server_exe=self.fake_exe.name,
            model_path=self.fake_model.name,
            lora_map=lora_map,
        )
        try:
            # 模拟 running 状态以测试参数校验
            with patch.object(bridge, "_ensure_running"):
                with self.assertRaises(ValueError) as ctx:
                    bridge.generate(
                        text="测试",
                        voice_ref=self.fake_model.name,
                        lora="unknown-lora",
                        fallback_to_cli=False,
                    )
                self.assertIn("未知的 LoRA 名称", str(ctx.exception))
        finally:
            bridge.shutdown()

    def test_base_model_has_no_lora_param(self):
        """测试选择基础模型 (lora=None) 时，请求不包含 lora 字段。"""
        lora_map = {"speaker-a": self.fake_lora_a.name}
        bridge = self.BridgeClass(
            server_exe=self.fake_exe.name,
            model_path=self.fake_model.name,
            lora_map=lora_map,
        )
        try:
            with patch.object(bridge, "_ensure_running"), patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = _test_wav_bytes()
                mock_post.return_value = mock_resp

                import json
                bridge.generate(
                    text="测试基础模型",
                    voice_ref=self.fake_model.name,
                    lora=None,
                )
                self.assertTrue(mock_post.called)
                req_json = mock_post.call_args[1]["json"]
                self.assertEqual(req_json["model"], "index_tts2")
                self.assertEqual(req_json["language"], "zh")
                # 切换为 speaker-a
                bridge.generate(
                    text="测试 LoRA",
                    voice_ref=self.fake_model.name,
                    lora="speaker-a",
                )
                req_json2 = mock_post.call_args[1]["json"]
                self.assertEqual(req_json2["model"], "index_tts2")
                self.assertEqual(req_json2["options"]["index_tts2.lora"], "speaker-a")
        finally:
            bridge.shutdown()

    def test_empty_server_wav_falls_back_to_cli(self):
        """HTTP 200 但 WAV 无音频帧时，应改走 CLI，不能交给 WebUI 播放 0:00。"""
        bridge = self.BridgeClass(
            server_exe=self.fake_exe.name,
            model_path=self.fake_model.name,
        )
        bridge.cli_exe = Path(self.fake_exe.name)
        empty_wav = _test_wav_bytes(frames=0)
        valid_wav = _test_wav_bytes()
        try:
            with (
                patch.object(bridge, "_ensure_running"),
                patch.object(bridge, "generate_via_cli", return_value=valid_wav) as fallback,
                patch("requests.post") as mock_post,
            ):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = empty_wav
                mock_post.return_value = mock_resp

                result = bridge.generate(
                    text="测试空音频回退",
                    voice_ref=self.fake_model.name,
                    language="zh",
                )

                self.assertEqual(result, valid_wav)
                fallback.assert_called_once()
        finally:
            bridge.shutdown()

    def test_runtime_lora_key_normalization(self):
        """PEFT GPT2 键应转换成 audio.cpp 的 GPT tensor 命名。"""
        from convert_lora_adapter import _split_key
        self.assertEqual(
            _split_key("base_model.model.h.3.attn.c_attn.lora_A.weight"),
            ("gpt.h.3.attn.c_attn", "A"),
        )
        self.assertEqual(
            _split_key("base_model.model.h.3.mlp.c_proj.lora_B.weight"),
            ("gpt.h.3.mlp.c_proj", "B"),
        )

    def test_runtime_lora_requires_complete_layer_coverage(self):
        """Checkpoint metadata must match every expected GPT projection."""
        from convert_lora_adapter import _validate_coverage
        pairs = {
            f"gpt.h.{layer}.attn.c_attn": {"A": object(), "B": object()}
            for layer in range(24)
        }
        _validate_coverage(pairs, ("c_attn",))
        del pairs["gpt.h.23.attn.c_attn"]
        with self.assertRaisesRegex(ValueError, "missing"):
            _validate_coverage(pairs, ("c_attn",))


class TestConvertCheckModelDir(unittest.TestCase):
    """convert_to_gguf.py 的输入校验测试。"""

    def _get_check_fn(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_to_gguf",
            PROJECT_ROOT / "scripts" / "convert_to_gguf.py",
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        return getattr(mod, "check_model_dir", None)

    def test_missing_files_raises(self):
        fn = self._get_check_fn()
        if fn is None:
            self.skipTest("check_model_dir 未找到")
        with self.assertRaises(FileNotFoundError):
            fn(Path("/nonexistent/checkpoints"))

    def test_partial_files_raises(self):
        fn = self._get_check_fn()
        if fn is None:
            self.skipTest("check_model_dir 未找到")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "config.yaml").write_text("dummy")
            # 缺少 gpt.pth 等文件
            with self.assertRaises(FileNotFoundError):
                fn(p)

    def test_v2_detected(self):
        fn = self._get_check_fn()
        if fn is None:
            self.skipTest("check_model_dir 未找到")
        required = [
            "config.yaml", "bpe.model", "gpt.pth", "s2mel.pth",
            "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt"
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            for f in required:
                (p / f).write_text("dummy")
            version = fn(p)
            self.assertEqual(version, "v2")

    def test_find_staging_script_finds_convert_index_tts2(self):
        """校验 find_audiocpp_staging_script 能找到 convert_index_tts2.py。"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_to_gguf",
            PROJECT_ROOT / "scripts" / "convert_to_gguf.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "find_audiocpp_staging_script", None)
        self.assertIsNotNone(fn)
        script_path = fn(PROJECT_ROOT / "integrations" / "audiocpp" / "bin")
        self.assertTrue(script_path.is_file())
        self.assertEqual(script_path.name, "convert_index_tts2.py")


class TestInferAudiocppUtils(unittest.TestCase):
    """infer_audiocpp.py 工具函数测试。"""

    def test_parse_vector_valid(self):
        from infer_audiocpp import parse_vector
        vec = parse_vector("0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8")
        self.assertEqual(len(vec), 8)
        self.assertAlmostEqual(vec[0], 0.1)
        self.assertAlmostEqual(vec[7], 0.8)

    def test_parse_vector_invalid_length(self):
        from infer_audiocpp import parse_vector
        with self.assertRaises(ValueError):
            parse_vector("0.1, 0.2, 0.3")

    def test_parse_vector_none(self):
        from infer_audiocpp import parse_vector
        self.assertIsNone(parse_vector(None))
        self.assertIsNone(parse_vector(""))


# ─────────────────────────────────────────────────────────────────────────────
# 集成测试（需要 audio.cpp 二进制 + GGUF 权重）
# ─────────────────────────────────────────────────────────────────────────────


def _integration_skip_reason() -> str | None:
    """返回需要跳过的原因，或 None（表示可以运行）。"""
    if not _AUDIOCPP_EXE.is_file():
        return f"audiocpp_server.exe 不存在：{_AUDIOCPP_EXE}"
    if not _GGUF_Q8.is_file() and not _GGUF_F16.is_file():
        return f"GGUF 权重不存在（{_GGUF_Q8} 或 {_GGUF_F16}）"
    return None


class TestIntegrationAudiocpp(unittest.TestCase):
    """端到端集成测试（需要 audio.cpp 二进制 + GGUF 权重）。"""

    @classmethod
    def setUpClass(cls):
        reason = _integration_skip_reason()
        if reason:
            raise unittest.SkipTest(reason)

        try:
            from audiocpp_bridge import AudioCppBridge
        except ImportError as e:
            raise unittest.SkipTest(f"audiocpp_bridge 不可导入：{e}")

        model = _GGUF_Q8 if _GGUF_Q8.is_file() else _GGUF_F16
        cls.bridge = AudioCppBridge(
            server_exe=_AUDIOCPP_EXE,
            model_path=model,
            port=18080,  # 避免与生产 server 冲突
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "bridge"):
            cls.bridge.shutdown()

    def test_integration_server_health(self):
        """启动 server 并检查健康接口。"""
        self.bridge.start_server()
        self.assertTrue(self.bridge.is_running, "Server 应处于运行状态")
        health = self.bridge.health_check()
        self.assertIn("status", health)

    def test_integration_list_models(self):
        """列出已加载的模型信息。"""
        self.bridge.start_server()
        info = self.bridge.list_models()
        # audio.cpp server 返回格式可能多样，至少不应报错
        self.assertIsInstance(info, dict)

    def test_integration_generate_basic(self):
        """基础 TTS 推理：文本 → WAV 字节流。"""
        if not _REFERENCE_WAV.is_file():
            self.skipTest(f"参考音频不存在：{_REFERENCE_WAV}")

        self.bridge.start_server()
        wav_bytes = self.bridge.generate(
            text="你好，这是 audio.cpp GGUF 推理流程的端到端测试。",
            voice_ref=_REFERENCE_WAV,
            language="zh",
        )
        self.assertIsInstance(wav_bytes, bytes)
        self.assertGreater(len(wav_bytes), 1000, "生成的 WAV 数据量过小，疑似无效")
        # WAV 文件应以 RIFF 头开始
        self.assertTrue(
            wav_bytes[:4] == b"RIFF",
            f"生成数据不是有效 WAV（起始字节：{wav_bytes[:4]!r}）",
        )

    def test_integration_server_reuse(self):
        """重复调用 start_server() 不应启动新进程。"""
        self.bridge.start_server()
        pid1 = self.bridge._proc.pid if self.bridge._proc else None
        self.bridge.start_server()  # 第二次调用
        pid2 = self.bridge._proc.pid if self.bridge._proc else None
        self.assertEqual(pid1, pid2, "重复启动应复用同一进程")


if __name__ == "__main__":
    unittest.main(verbosity=2)

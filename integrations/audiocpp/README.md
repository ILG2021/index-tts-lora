# audio.cpp IndexTTS2 集成

本目录承载针对 **IndexTTS2**（IndexTTS-2）的 audio.cpp C++ 推理后端的全部配置与脚本，与 MOSS-TTS 项目中 `integrations/openmoss/` 的架构对等。

> 本集成专注支持 **IndexTTS2** 官方模型，包含完整的音色克隆、4 种情感控制模式（音色自适应、情感音频、8 维情感向量、情感文本描述）以及语速调节。

## 目录结构

```text
integrations/audiocpp/
├── README.md                   本文件
├── bin/                        audio.cpp 预编译二进制（gitignore）
│   ├── audiocpp_cli.exe        命令行推理二进制
│   ├── audiocpp_server.exe     常驻 HTTP 服务二进制
│   └── audiocpp_gguf.exe       GGUF 打包与量化工具
├── weights/                    GGUF 权重文件（gitignore）
│   ├── base.gguf               原精度基础权重（f16）
│   ├── base-q8_0.gguf          Q8_0 量化（推荐首选，降低 ~50% 显存且无损）
│   └── voices/                 小体积运行时 LoRA adapters
│       └── speaker-a.safetensors
├── patches/
│   └── 0001-index-tts2-runtime-lora.patch
├── configs/
│   └── default.yaml            audiocpp_server 常用参数与 IndexTTS2 选项
└── scripts/
    ├── download_audiocpp.ps1   下载 Windows CUDA 预编译二进制
    ├── convert_index_tts2.py   IndexTTS2 专属 Safetensors staging 与转换脚本
    └── build.ps1               从源码编译（进阶，需 CMake + CUDA Toolkit）
```

---

## 1. 获取 audio.cpp 二进制

中国大陆网络说明：audio.cpp 官方给模型权重提供了 [ModelScope 镜像](https://www.modelscope.cn/models/HereIsMark/audio.cpp-gguf)，可避免从 Hugging Face 下载 GGUF；但源码和 Windows Release 二进制目前仍由 GitHub 官方仓库发布，本项目没有发现同等的官方大陆镜像。受限网络建议在可访问 GitHub 的机器下载二进制后离线拷贝，或从源码包本地编译。

### 方案 A：下载预编译二进制（仅基础模型推理）

```powershell
.\integrations\audiocpp\scripts\download_audiocpp.ps1
```

脚本会自动从 audio.cpp 官方 Releases 下载适配 Windows CUDA 的最新包并解压到 `integrations\audiocpp\bin\`。

### 方案 B：从源码编译（LoRA 热切换必需）

```powershell
.\integrations\audiocpp\scripts\build.ps1
```

---

## 2. 模型权重转换

### 2.1 基础权重转换为 GGUF

使用封装脚本（优先使用内置针对 IndexTTS2 定制的转换器）：

```powershell
python scripts\convert_to_gguf.py `
    --model-dir checkpoints `
    --audiocpp-dir integrations\audiocpp\bin `
    --output integrations\audiocpp\weights\base.gguf `
    --dtype f16 `
    --quantize-q8
```

> `--quantize-q8` 参数会在生成 `base.gguf` 后自动额外生成 `base-q8_0.gguf`（约减少 50% 体积，显存大幅降低且音质几无损失）。

也可以直接从 staging 权重生成 Q8_0（`audiocpp_gguf` 没有 llama-quantize 风格的 GGUF→GGUF `--quantize` 接口）：

```powershell
python scripts\convert_to_gguf.py `
    --model-dir checkpoints `
    --audiocpp-dir integrations\audiocpp\bin `
    --output integrations\audiocpp\weights\base-q8_0.gguf `
    --dtype q8_0
```

### 2.2 LoRA 转换为运行时 adapter

本仓库的 audio.cpp v0.8.1 patch 在 GPT-2 的 `c_attn`、`c_proj` 和 `c_fc` 图中加入低秩分支。基础 GGUF 常驻且不改写，每个 adapter 只保存 A/B 张量；`alpha/r` 在转换时折入 B，请求仍可额外指定 scale：

```powershell
python scripts\convert_lora_adapter.py `
    --checkpoint trained_ckpts\你的checkpoint.pth `
    --output integrations\audiocpp\weights\voices\speaker-a.safetensors
```

WebUI 的 `--lora NAME=PATH` 指向上述 `.safetensors`。桥接层只注册一个基础 model id，并把 adapter 路径作为 session options 传入；每个请求通过 `index_tts2.lora=NAME` 切换。切换会丢弃并重建 GPT prefill/decode/forward 图与 KV 状态，但不会重载基础权重或其他 IndexTTS2 组件。

当前 adapter 格式是本仓库定义的 IndexTTS2 safetensors，不是 llama.cpp 的 GGUF adapter；运算结构借鉴 llama.cpp/OpenMOSS 的“基础权重 + 运行时低秩增量”，但直接接入 audio.cpp 自己的 GPT-2 GGML 图。

adapter 上传到后端时使用 F16。按默认 rank 16、24 层、四个目标投影计算，每个 adapter 约占 15 MiB GPU 权重（另有少量 allocator 开销），而不是一份完整模型。切换 adapter 会重建生成图，因此首个 token 有一次图构建开销；同一 adapter 的后续请求可复用图。

---

## 3. 命令行推理（两种模式）

### 3.1 独立 Python 命令行推理工具（推荐，开箱即用）

本项目提供 `scripts/infer_audiocpp.py`，无需启动浏览器即可一键测试或批量跑音频：

```powershell
# 基础音色克隆
python scripts\infer_audiocpp.py `
    --text "你好，这是使用 audio.cpp 高性能后端的 IndexTTS2 语音。" `
    --voice-ref assets\reference.wav `
    --out outputs\test.wav

# 情感描述文本控制
python scripts\infer_audiocpp.py `
    --text "太不可思议了！我们居然真的做到了！" `
    --voice-ref assets\reference.wav `
    --emotion-text "极度惊喜与激动" `
    --emotion-alpha 0.85 `
    --out outputs\surprise.wav

# 8 维情感向量控制 (喜, 怒, 哀, 惧, 厌恶, 低落, 惊喜, 平静)
python scripts\infer_audiocpp.py `
    --text "今天的天气真好，心情非常舒畅。" `
    --voice-ref assets\reference.wav `
    --emotion-vector "0.9,0.0,0.0,0.0,0.0,0.0,0.1,0.2" `
    --out outputs\joy.wav

# 语速调节（duration_factor < 1.0 加快，> 1.0 减慢）
python scripts\infer_audiocpp.py `
    --text "这是一段快速播报的新闻提示。" `
    --voice-ref assets\reference.wav `
    --duration-factor 0.85 `
    --out outputs\fast.wav
```

### 3.2 直调 audiocpp_cli.exe 原生命令行

```powershell
.\integrations\audiocpp\bin\audiocpp_cli.exe `
    --task clon `
    --family index_tts2 `
    --model integrations\audiocpp\weights\base-q8_0.gguf `
    --backend cuda `
    --language zh `
    --voice-ref assets\reference.wav `
    --text "你好，这是 audio.cpp 基座连通性测试。" `
    --out outputs\test_audiocpp.wav
```

---

## 4. 常驻服务（audiocpp_server）

```powershell
# 启动常驻服务
.\integrations\audiocpp\bin\audiocpp_server.exe `
    --family index_tts2 `
    --model integrations\audiocpp\weights\base-q8_0.gguf `
    --backend cuda `
    --host 127.0.0.1 `
    --port 8080
```

---

## 5. WebUI 界面启动（Gradio 前端）

WebUI 前端已完全对齐 PyTorch 后端的功能，提供完整的音色参考、4 种情感模式、8 维情感滑块、语速倍率调节与 LoRA 下拉：

```powershell
# 一键 PowerShell 脚本启动
.\scripts\windows\webui_audiocpp.ps1 `
    -Model integrations\audiocpp\weights\base-q8_0.gguf

# 或直接运行 Python 命令：
python webui.py `
    --backend audiocpp `
    --audiocpp-exe integrations\audiocpp\bin\audiocpp_server.exe `
    --model integrations\audiocpp\weights\base-q8_0.gguf
```

---

## 6. IndexTTS2 专用参数与优化

| 参数 | 适用模式 | 默认值 | 作用说明 |
|------|----------|--------|----------|
| `duration_factor` | 请求选项 | `1.0` | 语速调节（`<1.0` 加快，`>1.0` 减慢） |
| `emotion_alpha` | 请求选项 | `1.0` | 情感控制强度 `[0.0, 1.0]` |
| `emotion_vector` | 请求选项 | 8 维浮点 | `[喜, 怒, 哀, 惧, 厌恶, 低落, 惊喜, 平静]` |
| `emotion_text` | 请求选项 | 字符串 | 引导情感风格的自然语言描述 |
| `interval_silence_ms`| 请求选项 | `200` | 标点符号分段间的静音时长（毫秒） |
| `index_tts2.mem_saver` | 会话选项 | `false` | 请求处理完成后立即释放中间暂存图，节省显存 |
| `index_tts2.speaker_cache_slots` | 会话选项 | `1` | 说话人特征缓存槽位（复用参考音频时加速） |

---

## 7. 与 MOSS-TTS openmoss 的架构对比

| 要素 | MOSS-TTS openmoss | 本项目 audio.cpp 集成 |
|------|-------------------|----------------------|
| 目标模型 | MOSS-TTS | **IndexTTS2**（官方 v2 架构） |
| C++ 后端 | libllama + ggml（openmoss 内嵌） | audio.cpp（ggml 框架） |
| 权重格式 | backbone.gguf + extras.gguf | 单个 GGUF（包含全量多模块） |
| 情感控制 | 简易控制 | 4 种模式：音色自适应 / 情感音频 / 8 维向量 / 描述文本 |
| 推理通道 | Gradio HTTP → server | Gradio HTTP → server，并备选 CLI 直接推理 |
| 量化支持 | llama-quantize Q4_K_M | audiocpp_gguf（IndexTTS2 已验证 Q8_0） |

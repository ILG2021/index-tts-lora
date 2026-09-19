# audio.cpp IndexTTS2 集成

本目录承载针对 **IndexTTS2**（IndexTTS-2）的 audio.cpp C++ 推理后端的全部配置与脚本，与 MOSS-TTS 项目中 `integrations/openmoss/` 的架构对等。

> 本集成专注支持 **IndexTTS2** 官方模型，包含完整的音色克隆、4 种情感控制模式（音色自适应、情感音频、8 维情感向量、情感文本描述）以及语速调节。

## 目录结构

```text
integrations/audiocpp/
├── README.md                   本文件
├── bin/                        audio.cpp 运行二进制（gitignore）
│   ├── audiocpp_cli.exe        命令行推理二进制
│   ├── audiocpp_server.exe     常驻 HTTP 服务二进制
│   └── audiocpp_gguf.exe       GGUF 打包与量化工具
├── weights/                    GGUF 权重文件（gitignore）
│   ├── base.gguf               原精度基础权重（f16）
│   ├── base-q8_0.gguf          Q8_0 量化基础权重（需在目标模型上验收音质）
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

网络来源约束：本项目禁止链接或自动访问中国大陆网站。audio.cpp 源码和 Release 二进制仅从 [GitHub 官方仓库](https://github.com/0xShug0/audio.cpp) 获取；模型脚本只使用 Hugging Face 等项目明确允许的来源。需要离线部署时，应在允许访问这些上游的环境下下载并校验，再拷贝到目标机器。

### 方案 A：下载预编译二进制（仅基础模型推理）

```powershell
.\integrations\audiocpp\scripts\download_audiocpp.ps1
```

脚本会自动从 audio.cpp 官方 Releases 下载适配 Windows CUDA 的最新包并解压到 `integrations\audiocpp\bin\`。

> **重要：**官方 audio.cpp v0.8.1 不包含本项目的 IndexTTS2 运行时 LoRA 扩展。预编译二进制不能用于 LoRA 热切换，必须使用下面的源码构建。

### 方案 B：从源码编译（LoRA 热切换必需）

构建依赖：

- Visual Studio 2022 或 Build Tools 2022，安装“使用 C++ 的桌面开发”、MSVC v143 x64/x86 和 Windows 10/11 SDK。
- CMake 3.20+、Git。
- 本项目当前的 Windows CUDA 构建使用 CUDA Toolkit 12.4，默认安装路径为 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`。

推荐显式指定 Toolkit：

```powershell
.\integrations\audiocpp\scripts\build.ps1 `
    -CudaToolkitRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

脚本会自动通过 `vswhere`/`VsDevCmd.bat` 加载 MSVC x64 环境，所以可以从普通 PowerShell 启动；也可以在“Developer PowerShell for VS 2022”中运行。CMake 会被明确配置为：

```text
-G "Visual Studio 17 2022" -A x64
-T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

可先检查 Toolkit：

```powershell
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" --version
```

`Test-Path` 应返回 `True`，`nvcc` 应显示 `release 12.4`。这里的本地 CUDA Toolkit 用于编译 audio.cpp，与 Python 训练环境中 PyTorch wheel 自带的 CUDA runtime 版本是两个独立概念。

构建脚本固定检出 audio.cpp `v0.8.1`，幂等应用
`patches/0001-index-tts2-runtime-lora.patch`，然后运行
`scripts/sanitize_source.ps1` 清理不符合项目网络策略的上游链接，再生成带热切换能力的 CLI、server 和 GGUF 工具。audio.cpp 不作为本仓库的 submodule；`integrations/audiocpp/src/` 是可删除、可重建的本地构建目录，真正提交和维护的上游功能修改是 patch，网络策略清理由脚本完成。
构建时显式设置 `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF`，不编译上游 server 的模型管理和网络下载功能；本项目运行时只从本地路径加载 GGUF 和 adapter。

#### Windows 构建故障排查

**`No CUDA toolset found`**

这通常表示 CMake 找到了 `nvcc.exe`，但 Visual Studio generator 没有获得 CUDA Toolkit 位置。使用上面的 `-CudaToolkitRoot` 参数；不要只依赖 PATH 中的 `nvcc`。

**CMake 提示 generator/toolset 与现有 cache 不一致**

失败的首次配置可能在 `build-cuda` 留下没有 `-T cuda=...` 的 cache。删除的只是可重建构建目录，不包含 GGUF、LoRA 或项目源码：

```powershell
Remove-Item `
    -LiteralPath "integrations\audiocpp\src\audio.cpp\build-cuda" `
    -Recurse `
    -Force
```

然后重新执行带 `-CudaToolkitRoot` 的构建命令。

**提示缺少 MSVC**

先确认 Visual Studio Installer 中安装了“使用 C++ 的桌面开发”。更新后的 `build.ps1` 会自动定位 VS 2022；如果仍然失败，改在“Developer PowerShell for VS 2022”中执行同一命令。

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

> `--quantize-q8` 参数会在生成 `base.gguf` 后额外生成 `base-q8_0.gguf`。Q8_0 可降低权重体积和显存占用，但应先用 F16 完成 LoRA 等价性对照，再单独验收量化音质。

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

每个 checkpoint 转换一次，例如：

```powershell
python scripts\convert_lora_adapter.py `
    --checkpoint trained_ckpts\speaker-b.pth `
    --output integrations\audiocpp\weights\voices\speaker-b.safetensors
```

转换器会校验 `r`/`alpha`/`target_modules`、24 层目标投影的完整覆盖、A/B 成对关系和张量形状。缺层、重复键或不支持的目标会立即报错，不会生成部分 adapter。

WebUI 的 `--lora NAME=PATH` 指向上述 `.safetensors`。桥接层只注册一个基础 model id，并把 adapter 路径作为 session options 传入；每个请求通过 `index_tts2.lora=NAME` 切换。切换会丢弃并重建 GPT prefill/decode/forward 图与 KV 状态，但不会重载基础权重或其他 IndexTTS2 组件。

当前 adapter 格式是本仓库定义的 IndexTTS2 safetensors，不是 llama.cpp 的 GGUF adapter；运算结构借鉴 llama.cpp/OpenMOSS 的“基础权重 + 运行时低秩增量”，但直接接入 audio.cpp 自己的 GPT-2 GGML 图。

adapter 上传到后端时使用 F16。按默认 rank 16、24 层、四个目标投影计算，每个 adapter 约占 15 MiB GPU 权重（另有少量 allocator 开销），而不是一份完整模型。切换 adapter 会重建生成图，因此首个 token 有一次图构建开销；同一 adapter 的后续请求可复用图。

所有在 `--lora` 中注册的 adapter 会在 server session 创建时一次性读取并上传；请求间的“热切换”不重读文件，也不重载基础 GGUF。当前不支持 server 运行期间新增/卸载 adapter；要更改注册列表需重启 server。`base` 和 `none` 是禁用 LoRA 的保留选择，不要用作 adapter 名称。

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

`infer_audiocpp.py --lora NAME=PATH` 每次只注册一个 adapter，适合单次冒烟测试；进程结束时它会关闭自己启动的 server。要验证多 LoRA 热切换，必须用下面的 WebUI/常驻 server 在同一进程中注册所有 adapter。

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

推荐让 `AudioCppBridge`/WebUI 生成 server JSON 配置并管理进程。如果需要手工启动，先创建 `audiocpp-server.json`：

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "lazy_load": true,
  "max_loaded_models": 1,
  "models": [
    {
      "id": "index_tts2",
      "family": "index_tts2",
      "path": "D:/vibecoding/index-tts-lora/integrations/audiocpp/weights/base-q8_0.gguf",
      "task": "tts",
      "mode": "offline",
      "session_options": {
        "index_tts2.lora.A": "D:/vibecoding/index-tts-lora/integrations/audiocpp/weights/voices/speaker-a.safetensors",
        "index_tts2.lora.B": "D:/vibecoding/index-tts-lora/integrations/audiocpp/weights/voices/speaker-b.safetensors"
      }
    }
  ]
}
```

然后执行：

```powershell
.\integrations\audiocpp\bin\audiocpp_server.exe `
    --config audiocpp-server.json
```

JSON 中建议使用绝对路径。同一 model session 一次只执行一个请求，`prepare()` 和 `run()` 在 server 的 busy guard 内串行，因此不会在两个并发请求之间交叉切换 adapter。

---

## 5. WebUI 界面启动（Gradio 前端）

WebUI 前端已完全对齐 PyTorch 后端的功能，提供完整的音色参考、4 种情感模式、8 维情感滑块、语速倍率调节与 LoRA 下拉：

```powershell
# 一键 PowerShell 脚本启动
.\scripts\windows\webui_audiocpp.ps1 `
    -Model integrations\audiocpp\weights\base-q8_0.gguf `
    -Lora @(
        "A=integrations\audiocpp\weights\voices\speaker-a.safetensors",
        "B=integrations\audiocpp\weights\voices\speaker-b.safetensors"
    ) `
    -Preload

# 或直接运行 Python 命令：
python webui.py `
    --backend audiocpp `
    --audiocpp-exe integrations\audiocpp\bin\audiocpp_server.exe `
    --model integrations\audiocpp\weights\base-q8_0.gguf `
    --lora "A=integrations\audiocpp\weights\voices\speaker-a.safetensors" `
    --lora "B=integrations\audiocpp\weights\voices\speaker-b.safetensors" `
    --preload
```

`--lora` 可重复指定；界面中选择“基础模型”或 adapter 名称时，始终使用同一个 C++ server 和同一份基础 GGUF。

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
| `index_tts2.lora.NAME` | 会话选项 | 无 | 启动时注册 `NAME=adapter.safetensors` |
| `index_tts2.lora` | 请求选项 | 空 | 当前 adapter 名称；空/`base`/`none` 禁用 LoRA |
| `index_tts2.lora_scale` | 请求选项 | `1.0` | 运行时 LoRA 额外缩放系数（必须为有限数） |

---

## 7. 与 MOSS-TTS openmoss 的架构对比

| 要素 | MOSS-TTS openmoss | 本项目 audio.cpp 集成 |
|------|-------------------|----------------------|
| 目标模型 | MOSS-TTS | **IndexTTS2**（官方 v2 架构） |
| C++ 后端 | libllama + ggml（openmoss 内嵌） | audio.cpp（ggml 框架） |
| 权重格式 | backbone.gguf + extras.gguf | 单个 GGUF（包含全量多模块） |
| 情感控制 | 简易控制 | 4 种模式：音色自适应 / 情感音频 / 8 维向量 / 描述文本 |
| 推理通道 | Gradio HTTP → server | Gradio HTTP → server，并备选 CLI 直接推理 |
| 量化支持 | llama-quantize Q4_K_M | audiocpp_gguf Q8_0（需在目标 GPU/模型上验收） |

---

## 8. 提交前热切换验收

验收必须在**同一个 server 进程**中完成；每次重启 CLI/server 不能证明热切换正常。固定参考音频、文本、情感参数、采样参数和 seed，至少完成：

1. `base → A → B → base`，检查切回 base 后无 adapter 残留。
2. `A → B → A`，检查两次 A 的听感、时长和内容一致；CUDA 非确定性可能使 WAV 不能逐字节相同。
3. 循环切换 100 次，用 `nvidia-smi -l 1` 观察显存；首次建图后应趋于稳定，不应随请求数持续单调增长。
4. 先用 F16 `base.gguf` 与 PyTorch PEFT 做对照，再测 Q8_0，避免将 LoRA 差异与量化误差混在一起。

当前接口只输出 WAV，因此 PyTorch/audio.cpp 不应以“波形逐样本相同”为标准。建议同时比较音色/韵律听感、ASR 内容、时长、RMS 和 mel 频谱；若要严格定位 LoRA 数值差异，还需额外暴露 GPT logits 或 hidden-state 调试接口。

> 本仓库已完成补丁适用性和静态检查，但不把它等同于目标 GPU 上的整模验收。发布生产版前应在实际模型、adapter 和部署 GPU 上执行上述测试。

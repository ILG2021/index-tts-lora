# Windows + LJSpeech：从数据到 LoRA 推理

本文以 PowerShell、Python 3.10 和 NVIDIA CUDA 为主路径。训练只微调 GPT 的 LoRA 层与说话人条件；DVAE 负责生成离散音频 token，BigVGAN 负责把 GPT 输出还原为波形，它们不参与微调。

## 1. 准备环境

建议使用 Windows 10/11、Python 3.10、Git、FFmpeg 和支持 CUDA 12.8 版 PyTorch 的 NVIDIA 驱动。安装脚本会固定安装 `torch==2.8.0`、`torchaudio==2.8.0` 和匹配的 `torchvision`：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\setup.ps1
.\.venv\Scripts\Activate.ps1
```

脚本实际执行的 PyTorch 安装命令如下。URL 是命令参数，不要保留 Markdown 链接的方括号和圆括号：

```powershell
pip install torch==2.8.0 torchaudio==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

如果安装后不能识别显卡，先检查 NVIDIA 驱动是否满足 CUDA 12.8 runtime 的要求。

Windows 默认不编译 BigVGAN 的可选 CUDA 扩展，而使用兼容性更好的 PyTorch 实现。这不影响训练正确性。确实配置好了 Visual Studio C++、CUDA Toolkit 与匹配的 PyTorch 时，可在安装前设置 `$env:INDEXTTS_BUILD_CUDA_EXT="1"`。

验证 GPU：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. 放置基础模型

把官方 IndexTTS 1.5 基础模型文件放到 `finetune_models`：

```text
finetune_models/
├── config.yaml
├── bpe.model
├── dvae.pth
├── gpt.pth
└── bigvgan_generator.pth
```

文件名必须与 `finetune_models/config.yaml` 一致。预处理需要 `dvae.pth`、`gpt.pth` 和 `bpe.model`；最终推理还需要 `bigvgan_generator.pth`。不要混用其他 IndexTTS 版本的配置与权重。

## 3. 准备 LJSpeech 数据

标准结构如下：

```text
D:\datasets\LJSpeech-1.1\
├── metadata.csv
└── wavs\
    ├── LJ001-0001.wav
    └── ...
```

`metadata.csv` 每行是：

```text
LJ001-0001|原始文本|规范化文本
```

转换并提取训练特征：

```powershell
.\scripts\windows\prepare_ljspeech.ps1 -DatasetDir "D:\datasets\LJSpeech-1.1"
```

这个脚本先生成 `finetune_data/ljspeech.lst`，每行使用绝对音频路径和文本并以 Tab 分隔，然后执行：

```powershell
python tools\extract_codec.py --audio_list finetune_data\ljspeech.lst --extract_condition
```

如需保留 LJSpeech 原始文本而不是规范化文本，可分两步执行：

```powershell
python tools\prepare_ljspeech.py --dataset-dir "D:\datasets\LJSpeech-1.1" --output finetune_data\ljspeech.lst --text-column raw
python tools\extract_codec.py --audio_list finetune_data\ljspeech.lst --extract_condition
```

预处理会：

1. 将音频重采样到 24 kHz；
2. 提取 mel 频谱和 DVAE code；
3. 为每条音频提取 condition latent；
4. 计算最能代表说话人的 medoid condition；
5. 按 90%/10% 生成训练集和验证集；
6. 更新 `finetune_data/processed_data/speaker_info.json`。

输出目录带时间戳，例如：

```text
finetune_data/processed_data/ljspeech_20260917_120000/
├── metadata.jsonl
├── metadata_train.jsonl
├── metadata_valid.jsonl
├── medoid_condition.npy
├── *_codes.npy
├── *_mel.npy
└── *_condition.npy
```

说话人 ID 默认是清单文件名的 stem，因此 `ljspeech.lst` 对应 `ljspeech`。重复处理同名说话人时，`speaker_info.json` 中的旧记录会被替换。

数据建议：音频应为单说话人、少噪声、无背景音乐，文本必须逐字对应。训练加载器会过滤短于 1 秒或长于 20 秒的样本。先用几十条做通路测试，再处理完整数据集。

## 4. 配置 LoRA

主要配置位于 `finetune_models/config.yaml`：

```yaml
train:
  epochs: 15
  batch_size: 4
  valid_batch_size: 4
  num_workers: 0
  gradient_accumulation_steps: 4
  data_path: "finetune_data/processed_data"
  optimizer:
    learning_rate: 5.0e-5
    warmup_ratio: 0.1
    loraplus_lr_ratio: 8.0
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
```

Windows 默认 `num_workers: 0`，可避免 DataLoader spawn、重复加载与共享内存问题。显存不足时依次减小 `batch_size` 到 2 或 1，再增大 `gradient_accumulation_steps` 维持有效批大小。有效批大小约为 `batch_size × gradient_accumulation_steps`。

目标模块包括 GPT 注意力和 MLP 投影层。训练时基础模型被冻结，采用 LoRA+ 优化器；说话人 medoid condition 也作为可学习参数更新。

## 5. 开始微调

```powershell
.\scripts\windows\train.ps1
```

等价的可覆盖参数命令：

```powershell
python train.py --config finetune_models\config.yaml --device cuda --batch-size 4 --valid-batch-size 4 --num-workers 0
```

每轮训练后都会验证并输出：

```text
finetune_models/checkpoints/
├── gpt_epoch_1.pth
├── ...
├── gpt_best.pth
└── gpt_finetuned.pth
```

同时生成 `finetune_models/config_finetuned.yaml`。保存时 LoRA 已合并进 GPT，因此这些 `.pth` 是可直接由普通 IndexTTS 推理类加载的完整 GPT 权重，不需要推理时再次挂载 PEFT adapter。`gpt_best.pth` 通常适合正式评估；默认最终配置指向 `gpt_finetuned.pth`。

建议同时观察验证集 mel loss 与试听结果。如果验证 loss 持续恶化，应减少 epoch、降低学习率或使用 `gpt_best.pth`。如要改用最佳权重，把 `config_finetuned.yaml` 中的 `gpt_checkpoint` 改为 `checkpoints/gpt_best.pth`。

## 6. 在 WebUI 中使用微调模型

参考音频建议取自目标说话人，保持 3–15 秒、音质干净。启动集成了微调推理的 WebUI：

```powershell
.\scripts\windows\webui_finetuned.ps1
```

或直接启动：

```powershell
python webui.py --model_dir finetune_models --config finetune_models\config_finetuned.yaml --speaker_info finetune_data\processed_data\speaker_info.json
```

打开 `http://127.0.0.1:7860`。如果 checkpoint 包含微调说话人，界面会显示“微调说话人”下拉框；单说话人清单 `ljspeech.lst` 对应 `ljspeech`。普通推理和批次推理都会使用选中的说话人 condition。基础模型没有说话人信息时，下拉框自动隐藏。

## 7. 多说话人

为每位说话人生成单独清单，例如 `alice.lst`、`bob.lst`，分别执行特征提取。所有条目会汇总进同一个 `speaker_info.json`，然后只需运行一次训练。推理时用 `--speaker alice` 或 `--speaker bob` 选择条件。

## 8. 常见问题

- `CUDA was requested...`：确认安装的是上述 cu128 构建，并检查 NVIDIA 驱动是否可用。
- `FileNotFoundError`：检查基础模型是否位于 `finetune_models`，以及配置中的文件名。
- 显存不足：降低 batch size，保持或提高梯度累积步数；关闭其他 GPU 程序。
- 预处理很慢：DVAE 与 condition 提取本身计算量较大；确认 `torch.cuda.is_available()` 为 `True`。
- 文本质量差：LJSpeech 的 normalized 列通常更稳定；不要让音频与文本错位。
- PowerShell 禁止脚本：仅对当前窗口执行 `Set-ExecutionPolicy -Scope Process Bypass`。

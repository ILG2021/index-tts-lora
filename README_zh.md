# IndexTTS2 LoRA 微调（Windows 优先）

本项目基于 [instavar/indextts2-finetuning](https://github.com/instavar/indextts2-finetuning) 的 IndexTTS2 数据预处理和 GPT 训练流程，将全参数 SFT 改为 PEFT LoRA，并提供 LJSpeech 转换、Windows 脚本和 Gradio WebUI。

训练只更新 IndexTTS2 GPT Transformer 内的 LoRA 参数。基础模型保持冻结，checkpoint 保存 adapter、优化器、调度器和恢复状态。WebUI 在官方基础 GPT 上挂载选定的 LoRA checkpoint。

## 1. 安装

需要 Windows 10/11、Python 3.10、Git、FFmpeg、NVIDIA 驱动和 CUDA GPU。在 PowerShell 中执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\setup.ps1
.\.venv\Scripts\Activate.ps1
```

脚本会执行以下关键操作：

```powershell
pip install torch==2.8.0 torchaudio==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/index-tts/index-tts.git vendor\index-tts
git -C vendor\index-tts checkout ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c
pip install -e vendor\index-tts
```

## 2. 下载 IndexTTS2 模型

直接运行：

```powershell
.\scripts\windows\download_models.ps1
```

它调用官方 IndexTTS2 下载器，会同时补齐主模型和辅助模型。等价命令是：

```powershell
indextts2 download --source huggingface --model-dir checkpoints
```

模型位于 `checkpoints`，其中包含 `config.yaml`、`bpe.model`、`gpt.pth`、`s2mel.pth`、`wav2vec2bert_stats.pt`、`feat1.pt`、`feat2.pt`、`qwen0.6bemo4-merge` 和 `hf_cache` 下的辅助模型。

## 3. LJSpeech 预处理

数据目录应包含 `metadata.csv` 和 `wavs`：

```text
D:\datasets\LJSpeech-1.1\
├── metadata.csv
└── wavs\LJ001-0001.wav
```

运行完整预处理：

```powershell
.\scripts\windows\prepare_ljspeech.ps1 -DatasetDir "D:\datasets\LJSpeech-1.1"
```

流程依次执行：

1. 将 `metadata.csv` 转为 IndexTTS2 JSONL；
2. 提取文本 token、semantic code、conditioning latent 和 emotion vector；
3. 生成训练及验证 manifest；
4. 为同一说话人的不同音频构建 prompt/target 配对。

最终训练文件为：

```text
processed_data/ljspeech/gpt_pairs_train.jsonl
processed_data/ljspeech/gpt_pairs_val.jsonl
```

## 4. LoRA 训练

```powershell
.\scripts\windows\train.ps1
```

默认参数适合先验证流程：batch size 2、梯度累积 8、10 epochs、AMP、LoRA rank 16。可覆盖：

```powershell
.\scripts\windows\train.ps1 -BatchSize 1 -GradAccumulation 16 -Epochs 5
```

也可直接运行：

```powershell
python train.py `
  --train-manifest processed_data\ljspeech\gpt_pairs_train.jsonl `
  --val-manifest processed_data\ljspeech\gpt_pairs_val.jsonl `
  --tokenizer checkpoints\bpe.model `
  --config checkpoints\config.yaml `
  --base-checkpoint checkpoints\gpt.pth `
  --output-dir trained_ckpts `
  --batch-size 2 `
  --grad-accumulation 8 `
  --epochs 10 `
  --learning-rate 1e-4 `
  --lora-r 16 `
  --lora-alpha 32 `
  --lora-dropout 0.05 `
  --amp
```

checkpoint 写入 `trained_ckpts`。不要默认选择 `latest.pth`；应结合验证 loss 和多组试听，明确选择表现最好的 step 或 epoch checkpoint。

## 5. WebUI 推理

把选定 checkpoint 传给 WebUI：

```powershell
.\scripts\windows\webui_finetuned.ps1 `
  -LoraCheckpoint "trained_ckpts\你实际生成的checkpoint文件名.pth"
```

访问 `http://127.0.0.1:7860`，上传说话人参考音频、输入文本并生成。也可以测试基础模型：

```powershell
python webui.py --model-dir checkpoints --config checkpoints\config.yaml
```

## 6. 目录说明

```text
trainers/train_gpt_v2_lora.py   IndexTTS2 LoRA 训练器
tools/prepare_ljspeech.py       LJSpeech 转 IndexTTS2 JSONL
tools/preprocess_data_v2.py     IndexTTS2 特征预处理
tools/build_gpt_prompt_pairs.py prompt/target 配对
webui.py                        基础模型或 LoRA 推理界面
scripts/windows/                Windows 一键脚本
```

参考项目是全参数 SFT。本项目对训练器做了实质修改：冻结基础 `UnifiedVoice`，只向 GPT Transformer 的 `c_attn`、`c_proj` 和 `c_fc` 注入 LoRA。相关来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 7. 数据、训练与权重使用细节

所有命令在项目根目录的 PowerShell 执行。不要混用 IndexTTS 1.5 的 `dvae.pth`、`bigvgan_generator.pth` 或旧预处理结果；本项目使用 IndexTTS2 的 semantic codec 与 S2Mel。

你的数据可直接使用 UTF-8 编码、竖线分隔的两栏格式。第一栏包含 `.wav` 后缀：

```text
LJ001-0001.wav|朗读文本
LJ001-0002.wav|朗读文本
```

第一栏也可以包含 `wavs` 下的相对子目录：

```text
文件夹/file0001.wav|朗读文本
speaker_b/session_01/file0002.wav|另一段文本
```

转换器会在 `wavs` 中按相对路径查找文件，不会再次追加后缀；正斜杠和反斜杠都可识别。禁止绝对路径、盘符和 `../`，文件必须位于 `wavs` 内。普通文件生成的样本 ID 会去掉 `.wav`；包含子目录时，内部 ID 由路径和短哈希生成，因此不会把斜杠带进特征文件名。Windows 下路径不能大小写重复。音频建议为真正的 PCM WAV，不要只给 MP3 改扩展名。空文本会跳过。自定义音色数据必须让音频与文本逐条对应，避免截断、背景音乐及多说话人混合。

同时兼容官方三栏格式 `ID|原始文本|规范化文本`（第一栏可省略 `.wav`）。三栏默认使用第三栏；只有三栏格式需要用 `--text-column raw` 切换到第二栏。不要在同一个文件中混用两栏和三栏。

JSONL 的 `speaker` 是配对用的说话人分组 ID，不是声音嵌入，也不需要旧版 `speaker_info`。单说话人可以统一使用 `ljspeech`；多说话人必须分别标注真实 ID。配对不会跨说话人，也不会将同一条音频同时作为自身的 prompt 和 target。

默认约 1% 数据按 ID 哈希分到验证集。**每个需要参与配对的说话人，在对应 split 中至少要有两条有效音频**；两条原始数据不代表能完成训练。小数据集可能产生空验证集，此时应换新的输出目录，提高 `--val-ratio`（例如 `0.1`）并检查拆分结果，再分别生成配对文件。不能通过复制训练配对到验证集解决。预处理使用 `--batch-size 1`，目前拒绝变长音频的批量特征提取，避免把 padding 当作真实语义码保存。

重新预处理时已有 ID 保持原来的 train/val 归属。修改文本、模型、分词器、说话人或拆分比例后，应使用全新的 `--output-dir`，避免旧特征、旧 manifest 混入。`--skip-existing` 仅用于同一份输入及同一套模型下的断点续提，不验证文件内容哈希。

训练 batch size 与预处理 batch size 独立。默认有效 batch 约为 `2 × 8 = 16` 个配对，末尾不足一组仍更新；显存不足先把训练 batch 降为 1，再增加梯度累积。LoRA 只减少可训练参数及优化器状态，不保证低显存：基础模型、激活、预处理辅助模型仍占显存。目前关闭 GPT 梯度检查点，避免冻结输入及移除 `wte` 后与 PEFT 的输入梯度钩子冲突。

仅训练 GPT 的 LoRA；文本嵌入、音频头、conditioning/emotion 模块和 S2Mel 不训练。不支持借此替换词表或直接扩展语言。`--language` 只是数据标记，文本规范化使用官方实现。

### 查看训练与选择 checkpoint

```powershell
.\.venv\Scripts\tensorboard.exe --logdir trained_ckpts\logs
Get-ChildItem trained_ckpts\model_*.pth | Sort-Object LastWriteTime
```

每 1000 个优化器步骤保存 step 权重，每个完整 epoch 保存 epoch 权重，另外维护 `latest.pth`；不会自动删除旧 checkpoint，请留足磁盘空间。`latest.pth` 不是自动选出的最佳权重。按验证集 loss 与固定的多组试听共同选择，检查漏字、重复、音色和发音，避免只看训练 loss。

可用以下命令选取最新的完整 epoch 权重先做冒烟测试（“最新”不表示“最好”）：

```powershell
$Checkpoint = Get-ChildItem trained_ckpts\model_epoch*_step*.pth | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Checkpoint) { throw "还没有完整 epoch checkpoint" }
.\scripts\windows\webui_finetuned.ps1 -LoraCheckpoint $Checkpoint.FullName
```

WebUI 是**推理界面**，不是训练界面；训练通过命令行完成。启动时加载一个 LoRA，将其合并到内存中的 GPT，并同步推理 transformer；不改磁盘上的基础模型。暂不支持热切换，换 LoRA 请停止并重新启动 WebUI。LoRA 不是独立语音模型，仍需要原版基础权重及说话人参考音频。基础模型必须与训练时一致；键名/形状校验不能证明基础权重内容一致。

生成音频写入 `outputs`，默认只监听本机。不要随意使用 `--share` 或监听 `0.0.0.0` 暴露服务。checkpoint 可能包含 pickle 恢复状态，**只加载自己训练或可信来源的文件**。

### 恢复训练

在第 4 节的完整训练命令中追加 `--resume auto --trust-resume-state`，其余数据、模型、训练配置及输出目录保持一致。恢复仅支持带完整元数据和恢复附件的 epoch checkpoint；不要只搬一个 `.pth` 文件，也不要用 `latest.pth` 或中途 step checkpoint 恢复。`--epochs` 是总目标轮数，不是再训练多少轮。`auto` 找不到可恢复 epoch 时会从头开始，务必检查终端输出。

## 8. 自检与验收边界

无需下载权重即可执行轻量回归测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.version.cuda, torch.cuda.is_available())"
```

版本应为 torch/torchaudio `2.8.0`（可能带 `+cu128`），CUDA runtime `12.8`。训练使用 GPU 时最后一项应为 `True`。无需为了预编译 wheel 单独安装完整 CUDA Toolkit；本项目默认不启用自定义 CUDA kernel。

轻量测试覆盖数据转换失败不覆盖原文件、重复/非法 ID、配对路径、同说话人配对、LoRA 键名及形状、训练入口导入副作用、Python 语法及安装版本判断。它们**不等于真实模型训练/推理已经通过**。完整验收还需要下载模型，在目标 GPU 上预处理数据、运行至少一个完整 epoch、重新加载 checkpoint 试听，并测试 epoch 恢复。当前修订未在完整模型/GPU 环境完成上述验收，不能承诺零 bug 或音质。

### 常见问题

测试还包含一个不下载模型的小型 GPT2/PEFT 梯度、保存重载及合并一致性用例。缺少 torch、transformers 或 peft 时会明确跳过；应在安装好的项目环境中重新运行。它验证 LoRA 机制，不替代 IndexTTS2 整模验收。

- `No valid prompt-target pairs`：检查该 split 的有效样本数和 speaker 分组，不要混入训练集充当验证集。
- `Missing ... model`：先运行模型下载脚本；只下载 `gpt.pth` 不够。
- `CUDA out of memory`：降低训练 batch、缩短过长音频；LoRA 不会消除激活显存占用。
- `LoRA keys/shape mismatch`：检查是否是本项目生成的 IndexTTS2 LoRA，而非完整 SFT、IndexTTS 1.5 或其他模块配置。
- PowerShell 执行被阻止：仅对当前会话使用第 1 节的 `Set-ExecutionPolicy -Scope Process Bypass`，不需要全局关闭策略。

安装及模型下载需要网络。下载脚本显式选择 Hugging Face，但官方运行时和依赖也可能自行下载辅助资源；此文档不保证全链路仅访问某一域名或完全离线。部署前应单独审核官方运行时、模型缓存和网络策略。

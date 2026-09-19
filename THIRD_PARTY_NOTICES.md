# Third-party notices

Parts of the IndexTTS2 preprocessing, prompt-pair construction, guarded resume,
training and inference workflow are adapted from
[`instavar/indextts2-finetuning`](https://github.com/instavar/indextts2-finetuning),
Copyright 2026 Instavar, licensed under the Apache License 2.0.

The runtime model implementation is installed separately from
[`index-tts/index-tts`](https://github.com/index-tts/index-tts) and remains subject
to its own source and model licenses.

Changes in this project include Windows launchers, LJSpeech conversion, PEFT LoRA
injection and adapter checkpoint loading in the Gradio interface.

The optional native GGUF inference workflow integrates with
[`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp), distributed under
the MIT License. audio.cpp binaries and model weights are downloaded separately
and remain subject to their respective licenses.

# Vendored audio.cpp source

This directory is a repository-owned source snapshot based on audio.cpp
`v0.8.1` at commit `f2b4937306daa25f5c78520f3c626ed31495a37a`.

It is intentionally not a Git submodule. Local changes add runtime IndexTTS2
GGUF LoRA adapters and MSVC UTF-8 compilation. Upstream tests, documentation,
WebUI source/demo assets, model-manager fixtures, and optional server frontend
modules are omitted because this integration builds only the `index_tts2`
custom model composite.

The retained third-party sources and their licenses remain governed by the
upstream `LICENSE` and `THIRD_PARTY_NOTICES.md` files.

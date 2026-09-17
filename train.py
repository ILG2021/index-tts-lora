"""Windows-friendly IndexTTS2 LoRA training entry point."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    trainer_dir = Path(__file__).parent / "trainers"
    sys.path.insert(0, str(trainer_dir))
    runpy.run_path(str(trainer_dir / "train_gpt_v2_lora.py"), run_name="__main__")

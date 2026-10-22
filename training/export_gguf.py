"""Merge the LoRA adapter into the base model and shrink it to a phone-sized GGUF.

  python training/export_gguf.py                 # -> outputs/phone-agent-q4_k_m.gguf (~1 GB for 1.5B)
  python training/export_gguf.py --quant Q8_0    # bigger, slightly more accurate

Builds llama.cpp (pinned to the same commit the Android app uses) the first time.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"  # keep in sync with android/setup_llama.sh


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def ensure_llama_cpp(path: Path) -> Path:
    if not (path / "convert_hf_to_gguf.py").exists():
        path.mkdir(parents=True, exist_ok=True)
        sh(["git", "init", "-q"], cwd=path)
        sh(["git", "fetch", "-q", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_CPP_COMMIT], cwd=path)
        sh(["git", "checkout", "-q", "FETCH_HEAD"], cwd=path)
    exe = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    quantize = next((p for p in (path / "build" / "bin").rglob(exe)), None) if (path / "build").exists() else None
    if quantize is None:
        sh(["cmake", "-S", str(path), "-B", str(path / "build"), "-DCMAKE_BUILD_TYPE=Release"])
        sh(["cmake", "--build", str(path / "build"), "--config", "Release", "-j", "--target", "llama-quantize", "llama-server"])
        quantize = next((path / "build" / "bin").rglob(exe))
    return quantize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path, default=ROOT / "outputs" / "lora")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "outputs" / "llama.cpp")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = (a.lora / "BASE_MODEL").read_text().strip()
    merged = a.out / "merged"
    print(f"Merging {a.lora} into {base}...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(a.lora)).merge_and_unload()
    model.save_pretrained(str(merged), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(a.lora)).save_pretrained(str(merged))

    quantize = ensure_llama_cpp(a.llama_cpp)
    # Not llama.cpp's requirements file: it pins a CPU-only torch and an older transformers, which would
    # replace the GPU setup used for training. The converter only needs its own gguf package on top.
    sh([sys.executable, "-m", "pip", "install", "-q", str(a.llama_cpp / "gguf-py"), "sentencepiece"])
    f16 = a.out / "phone-agent-f16.gguf"
    sh([sys.executable, str(a.llama_cpp / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"])
    final = a.out / f"phone-agent-{a.quant.lower()}.gguf"
    sh([str(quantize), str(f16), str(final), a.quant])
    f16.unlink()
    print(f"\nPhone model ready: {final} ({final.stat().st_size / 1024 ** 3:.2f} GB)")
    print("Copy it to the phone and pick it in the app (see android/README.md).")


if __name__ == "__main__":
    main()

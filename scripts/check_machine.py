"""Check this computer and find the LLMs on it, then say what it can handle.

  python scripts/check_machine.py                      # looks in the usual places
  python scripts/check_machine.py "D:\\AI\\models"      # also scan your models folder

Needs only Python 3.9+. Writes machine_report.json next to the repo.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GB = 1024 ** 3
MODEL_EXT = {".gguf", ".safetensors", ".bin", ".pt", ".onnx", ".tflite", ".task", ".mlmodel"}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def ram_gb() -> float:
    if sys.platform == "win32":
        class MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEM()
        m.dwLength = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / GB
    if sys.platform == "darwin":
        out = run(["sysctl", "-n", "hw.memsize"]).strip()
        return int(out) / GB if out.isdigit() else 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return 0.0


def gpus() -> list[dict]:
    found = []
    out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    for line in out.strip().splitlines():
        name, _, mem = line.rpartition(",")
        if name:
            found.append({"name": name.strip(), "vram_gb": round(float(mem) / 1024, 1), "kind": "nvidia"})
    if sys.platform == "darwin" and platform.machine() == "arm64":
        found.append({"name": "Apple Silicon (shares system RAM)", "vram_gb": round(ram_gb() * 0.7, 1), "kind": "apple"})
    if not found and sys.platform == "win32":
        out = run(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
        for name in out.strip().splitlines():
            if name.strip():
                found.append({"name": name.strip(), "vram_gb": 0, "kind": "other"})
    return found


def ollama_models() -> list[dict]:
    out = run(["ollama", "list"])
    models = []
    for line in out.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4:
            size = parts[2] + parts[3] if parts[3] in ("GB", "MB") else parts[2]
            models.append({"name": parts[0], "size": size, "where": "ollama"})
    return models


def model_dirs(extra: list[str]) -> list[Path]:
    home = Path.home()
    dirs = [Path(p) for p in extra] + [
        home / ".ollama" / "models", home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models",
        home / ".cache" / "huggingface" / "hub", home / "Downloads", home / "models", ROOT.parent,
    ]
    if os.environ.get("OLLAMA_MODELS"):
        dirs.append(Path(os.environ["OLLAMA_MODELS"]))
    return [d for d in dirs if d.is_dir()]


def scan_files(dirs: list[Path]) -> list[dict]:
    """Model files over 100 MB. Ollama blobs are reported by `ollama list` instead."""
    seen, found = set(), []
    for d in dirs:
        for root, subdirs, files in os.walk(d):
            subdirs[:] = [s for s in subdirs if not s.startswith(".") and s not in ("node_modules", "blobs")]
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() not in MODEL_EXT:
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size < 100 * 1024 ** 2 or p in seen:
                    continue
                seen.add(p)
                found.append({"name": p.name, "path": str(p), "size_gb": round(size / GB, 2), "where": "file"})
    return found


def size_gb(m: dict) -> float:
    if "size_gb" in m:
        return m["size_gb"]
    s = m.get("size", "0GB")
    try:
        return float(s[:-2]) / (1024 if s.endswith("MB") else 1)
    except ValueError:
        return 0.0


def advise(ram: float, gpu: list[dict], disk_free: float, models: list[dict]) -> list[str]:
    tips = []
    vram = max((g["vram_gb"] for g in gpu if g["kind"] in ("nvidia", "apple")), default=0)
    usable = [m for m in models if m["where"] == "ollama" or m["name"].endswith(".gguf")]
    fits = [m for m in usable if 0 < size_gb(m) < ram * 0.6]
    too_big = [m for m in usable if size_gb(m) >= ram * 0.6]

    if not models:
        tips.append("No models found. Pass your models folder: python scripts/check_machine.py \"<folder>\"")
    elif not usable:
        tips.append("Found model files, but none in GGUF/Ollama form. The council needs them served by Ollama or "
                    "LM Studio. Safetensors models can be converted, or pull GGUF versions in Ollama.")
    if fits:
        tips.append(f"Council (models teaching each other): can run here with {len(fits)} model(s): "
                    + ", ".join(m["name"] for m in fits) + ". Run them one at a time (--workers 1).")
    if too_big:
        tips.append("Too big for this RAM, skip or use on Colab: " + ", ".join(m["name"] for m in too_big))
    if ram < 8:
        tips.append("Under 8 GB RAM: only 1-3B models are practical here. The simulator data needs no LLM at all, "
                    "so you can skip the council and still train a good model.")
    if vram >= 8 and any(g["kind"] == "nvidia" for g in gpu):
        tips.append(f"Training: your NVIDIA GPU ({vram} GB) can fine-tune the 1.5B phone model locally.")
    elif vram >= 6 and any(g["kind"] == "nvidia" for g in gpu):
        tips.append(f"Training: {vram} GB GPU is tight; use --base Qwen/Qwen2.5-0.5B-Instruct locally, "
                    "or the free Colab notebook for the 1.5B model.")
    else:
        tips.append("Training: no suitable GPU here. Use the free Google Colab notebook "
                    "(training/colab_pipeline.ipynb) - it trains on Google's GPU and gives you the phone model file.")
    if disk_free < 10:
        tips.append(f"Only {disk_free:.0f} GB disk free; you need about 10 GB for data, model and export.")
    return tips


def main():
    extra = sys.argv[1:]
    ram = ram_gb()
    gpu = gpus()
    disk_free = shutil.disk_usage(ROOT).free / GB
    print("Scanning for models (can take a minute)...")
    dirs = model_dirs(extra)
    models = ollama_models() + scan_files(dirs)
    report = {
        "os": f"{platform.system()} {platform.release()}", "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(), "ram_gb": round(ram, 1), "gpus": gpu, "disk_free_gb": round(disk_free, 1),
        "ollama_installed": bool(shutil.which("ollama")), "scanned": [str(d) for d in dirs], "models": models,
    }
    report["advice"] = advise(ram, gpu, disk_free, models)

    print(f"\nOS: {report['os']}   CPU cores: {report['cpu_cores']}   RAM: {report['ram_gb']} GB   Disk free: {report['disk_free_gb']} GB")
    print("GPU: " + (", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in gpu) or "none found"))
    print(f"Ollama installed: {'yes' if report['ollama_installed'] else 'no'}")
    print(f"\nModels found ({len(models)}):")
    for m in models:
        print(f"  - {m['name']}  [{m['where']}, {m.get('size') or str(m.get('size_gb')) + ' GB'}]" + (f"\n      {m['path']}" if "path" in m else ""))
    print("\nWhat this laptop can do:")
    for t in report["advice"]:
        print("  * " + t)
    out = ROOT / "machine_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

# Notes for Claude

This repo builds an offline phone agent: big local LLMs (the "council") create and
cross-check training data, a small Qwen2.5 model is fine-tuned on it, exported to GGUF,
and run by the Android app in `android/`. README.md explains the whole pipeline.

## The owner's setup

- Laptop: Windows 11, 16 GB RAM, 8 cores, NVIDIA MX250 (2 GB, too small for LLM work), CPU only.
- Repo at `D:\handsfree-phone-agent`. Ollama is installed.
- `D:\LocalAI` holds Gemma4-E2B, Gemma4-E4B and Qwen3-4B as raw Hugging Face folders
  (not usable by Ollama as-is), Whisper `base.pt`/`small.pt`, and a multilingual embedding model.
  LM Studio has Llama-3.2-1B Q8 GGUF (too weak to teach).
- Free GPUs: Kaggle (30 h/week, T4 x2) is the main place for the council and training; Colab as backup.
- The owner prefers Claude to drive things end to end and report results, not hand them long instructions.

## The plan to drive

1. `git pull`, `python -m pip install requests pytest`, `python -m pytest tests`,
   `python scripts\check_machine.py "D:\LocalAI"`.
2. Run `training/kaggle_pipeline.ipynb` on Kaggle with the Kaggle CLI (`pip install kaggle`;
   the key goes in `C:\Users\DRmoh\.kaggle\kaggle.json`; if it's missing, ask the owner to create one
   at kaggle.com/settings → API → Create New Token). Push it with `kaggle kernels push` and a
   `kernel-metadata.json` that enables GPU and internet. The repo is private and secrets can't be
   attached through the CLI, so upload the repo as a private Kaggle dataset
   (`kaggle datasets create`, excluding `.git`, `data`, `outputs`) and change the notebook's first
   cell to copy it from `/kaggle/input/...` instead of cloning. Check Gemma 4 / Qwen 3 tag names on
   ollama.com/library before setting `MODELS`.
3. Poll `kaggle kernels status`; when done, `kaggle kernels output` to get
   `phone-agent-q4_k_m.gguf` and `council-data/`. Show the owner the teacher leaderboard and the
   student's score per task type. Fix any errors and rerun. Ask before going past 10 GPU hours in a week.
4. Build the Android app (`android/README.md`). Ask before installing Android Studio or other
   large tools. The owner must turn on the accessibility service and USB debugging on the phone themselves.

## Rules for changes

- `pipeline/spec.py` and `android/app/src/main/java/ai/handsfree/agent/AgentPrompt.kt` must stay
  identical in prompt and grammar; `pytest tests` checks this.
- The llama.cpp commit is pinned in both `training/export_gguf.py` and `android/setup_llama.sh`;
  change both together.
- Seed 999 is reserved for leaderboard tasks; never train on it.

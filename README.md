# Hands-free phone agent

A small AI model that runs **on the phone, offline**, listens to your voice and drives the
phone for you: opens apps, taps, types, scrolls, calls, sends WhatsApps, sets alarms,
navigates. It understands English and Egyptian Arabic.

The big models on your laptop (Gemma 4, Qwen 3 via Ollama) are the **teachers**. They never
go on the phone. They test each other, write and check training examples, and a small
**student** model (~1 GB) learns from the result. Only the student goes on the phone.

```
 Laptop / Colab                                              Phone
 ┌───────────────────────────────────────────────┐          ┌──────────────────────┐
 │ phone simulator ──► known-right examples      │          │ voice ► student model │
 │ Gemma 4 ◄──council──► Qwen 3  (Ollama)        │  1 file  │   ▲          │        │
 │   score each other, rewrite requests,         │ ───────► │ screen     action     │
 │   invent new tasks, vote on every step        │  .gguf   │   │          ▼        │
 │ ──► train student ──► test ──► retrain        │          │ accessibility service │
 └───────────────────────────────────────────────┘          └──────────────────────┘
```

## How the models "talk to each other"

| Step | What happens | Command |
|---|---|---|
| Leaderboard | Every model drives a simulated phone; each answer is checked against the known right action. You see which model is the best teacher and what each is weak at. | `council.py leaderboard` |
| Paraphrase | The best teacher rewrites requests the way people really talk ("ابعت لأحمد إني جاي"); a different model must confirm each rewrite means the same thing. | `council.py paraphrase` |
| Invent | A teacher invents tasks for apps the simulator lacks (Talabat, InstaPay, Uber…). The other models solve every step independently. Disagreements go back to the author, who may change its mind; only tasks the council agrees on are kept. | `council.py invent` |
| Self-improvement | The trained student takes the leaderboard test. The next dataset gets more examples of whatever it failed. | `build_dataset.py --focus` |

The simulator alone (no LLM needed) already produces correct training data for 13 kinds of task,
so even a laptop that can't run the teachers can still produce a working model.

## Step by step (Windows laptop)

### 0. Get the code and check the laptop

Open PowerShell:

```powershell
cd D:\
git clone https://github.com/Mo3gza21/handsfree-phone-agent.git
cd handsfree-phone-agent
python -m pip install requests pytest
python scripts\check_machine.py "D:\LocalAI"
```

It lists your RAM, GPU and every model it finds in `D:\LocalAI` and Ollama, then says what the
laptop can handle. It writes `machine_report.json`.

### 1. Make sure Ollama serves your models

```powershell
ollama list
```

If your Gemma 4 / Qwen 3 are GGUF files in `D:\LocalAI` that aren't in that list yet, add each one:

```powershell
"FROM D:\LocalAI\<file>.gguf" | Out-File -Encoding ascii Modelfile
ollama create qwen3-local -f Modelfile
```

### 2. Let the models teach each other

Use the exact names from `ollama list`:

```powershell
cd pipeline
python council.py leaderboard --models gemma4:<tag> qwen3:<tag> --n 40
python council.py paraphrase  --models <best> <other> --n 200
python council.py invent      --models <best> <other> --n 100
```

On a weak laptop this is slow but fine to leave running overnight. Keep `--workers 1`. Each
command appends to `data\` as it goes, so you can stop it any time and keep what's done.

### 3. Build the training data

```powershell
python build_dataset.py --sim 5000 --extra ..\data\paraphrased.jsonl ..\data\invented.jsonl
```

### 4. Train + shrink + test: on Kaggle or Colab (free GPU)

Your laptop doesn't need a GPU.

- **Kaggle (recommended):** 30 free GPU hours a week, sessions up to 12 h. Create a notebook,
  **File → Import Notebook**, upload `training/kaggle_pipeline.ipynb`, and follow the first cell
  (T4 x2, internet on, a `GITHUB_TOKEN` secret). It can also run the whole council on the GPU
  with bigger teachers, which is much faster than a laptop CPU. That makes steps 1-3 optional.
- **Colab:** upload `training/colab_pipeline.ipynb` and pick a **T4 GPU** runtime.

Either one trains the student, converts it to `phone-agent-q4_k_m.gguf` (~1 GB), tests it on 150
unseen tasks, and gives you the file. To train on data your laptop made, upload the `data`
folder first.

If you have an NVIDIA GPU with 8 GB+ you can train locally instead:
`pip install -r requirements.txt`, then `python training\train_lora.py` and `python training\export_gguf.py`.

### 5. Put it on the phone

See [android/README.md](android/README.md): build the app, choose the `.gguf`, turn on the
accessibility service, and say "Maya, open WhatsApp and tell Ahmed I'm on my way".

## Which student model?

`--base` picks the starting model for the student.

| Base | Phone file | Phones |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (default) | ~1 GB | most phones from ~2020 on, 6 GB+ RAM |
| `Qwen/Qwen2.5-0.5B-Instruct` | ~400 MB | older / 4 GB RAM phones; less accurate |
| `Qwen/Qwen2.5-3B-Instruct` | ~2 GB | flagships; most accurate |

Qwen 2.5 is the tested path: the phone formats prompts the way it expects (ChatML). Other bases
(Qwen 3, Gemma) can work but check that `llama_chat_apply_template` supports their template
before relying on them.

## Repo layout

```
pipeline/spec.py           action format + system prompt (the phone app mirrors it)
pipeline/scenarios.py      simulated phone: 13 task families, English + Egyptian Arabic
pipeline/council.py        leaderboard / paraphrase / invent
pipeline/build_dataset.py  merge into data/train.jsonl, data/val.jsonl
training/train_lora.py     LoRA / QLoRA fine-tune
training/export_gguf.py    merge + convert + quantize to GGUF for the phone
training/kaggle_pipeline.ipynb everything above on Kaggle's free GPU
training/colab_pipeline.ipynb  same, for Google Colab
scripts/check_machine.py   what can this laptop do?
android/                   the phone app (Kotlin + llama.cpp)
tests/                     pytest; runs the council against fake model servers
```

## Status

| Part | State |
|---|---|
| Simulator, council, dataset builder | Tested here (`pytest tests`) against fake model servers; not yet run with your real models. |
| Training + export | Written against current TRL/PEFT/llama.cpp, **not yet run**: this build machine had no GPU. The Colab notebook is the first real run. |
| Android app | Kotlin type-checks against the Android 16 API and the C++ engine compiles against the pinned llama.cpp, but the APK hasn't been built or tried on a phone yet. |

Expect the first real runs to need small fixes.

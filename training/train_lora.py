"""Fine-tune the small phone model on data/train.jsonl with LoRA.

  python training/train_lora.py                                   # Qwen2.5-1.5B (default)
  python training/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct # weaker phones / 6 GB GPUs

On an NVIDIA GPU it loads the base model in 4-bit (QLoRA), so a free Colab T4
is enough. Only the action JSON is trained on, not the prompt.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent


def warmup(ratio: float) -> dict:
    """transformers 5 dropped warmup_ratio; there warmup_steps takes a fraction of the total."""
    if "warmup_ratio" in {f.name for f in dataclasses.fields(SFTConfig)}:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps for a quick trial run")
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    quant = None
    if cuda:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base, quantization_config=quant, device_map="auto" if cuda else None,
        torch_dtype=(torch.bfloat16 if bf16 else torch.float16) if cuda else torch.float32)

    ds = load_dataset("json", data_files={"train": str(a.data / "train.jsonl"), "val": str(a.data / "val.jsonl")})
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "completion")])

    cfg = SFTConfig(
        output_dir=str(a.out), num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, lr_scheduler_type="cosine", **warmup(0.03),
        logging_steps=20, eval_strategy="steps", eval_steps=200, save_steps=200, save_total_limit=2,
        max_length=a.max_len, completion_only_loss=True, bf16=bf16, fp16=cuda and not bf16,
        gradient_checkpointing=cuda, report_to="none",
    )
    lora = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(a.out))
    tok.save_pretrained(str(a.out))
    (a.out / "BASE_MODEL").write_text(a.base)
    print(f"LoRA adapter saved to {a.out}. Next: python training/export_gguf.py")


if __name__ == "__main__":
    main()

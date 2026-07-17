import argparse, random, math, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling)
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_sample(path, n=5000, seed=0):
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    random.seed(seed)
    if n and n < len(lines): lines = random.sample(lines, n)
    return Dataset.from_dict({"text": lines})

def eval_model(model_path, data_path, n=5000, seq_len=512, mlm_prob=0.15, batch_size=8):
    ds = load_sample(data_path, n=n)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    def tok_fn(ex): return tok(ex["text"], truncation=True, max_length=seq_len)
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=mlm_prob)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    losses, total = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Eval"):
            for k in batch: batch[k] = batch[k].to(model.device)
            out = model(**batch)
            losses += out.loss.item() * batch["labels"].ne(-100).sum().item()
            total  += batch["labels"].ne(-100).sum().item()
    avg_loss = losses / max(total,1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seq_len", type=int, default=512)
    args = ap.parse_args()
    loss, ppl = eval_model(args.model, args.data, n=args.n, seq_len=args.seq_len)
    print(f"avg_masked_loss={loss:.4f}  pseudo_ppl={ppl:.2f}")

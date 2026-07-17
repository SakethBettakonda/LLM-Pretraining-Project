# scripts/eval_mlm_accuracy.py
import argparse, json, random, os
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling)
from torch.utils.data import DataLoader
from functools import partial

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_lexicon_ids(tokenizer, lexicon_path):
    if not lexicon_path: return None
    with open(lexicon_path, "r", encoding="utf-8") as f:
        terms = json.load(f)
    ids = set()
    specials = set(tokenizer.all_special_ids)
    for t in terms:
        toks = tokenizer(t, add_special_tokens=False)["input_ids"]
        for tid in toks:
            if tid not in specials:
                ids.add(tid)
    return torch.tensor(sorted(ids), dtype=torch.long)


def tokenize_fn(tokenizer, ex, max_len):
    return tokenizer(ex["text"], truncation=True, padding=False, max_length=max_len)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lexicon_path", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    raw = load_dataset("text", data_files={"train": args.data})
    ds = raw["train"].select(range(min(args.n, len(raw["train"]))))

    tok_fn = partial(tokenize_fn, tokenizer, max_len=args.seq_len)
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    dl = DataLoader(ds_tok, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collator, num_workers=0)

    lex_ids_tensor = build_lexicon_ids(tokenizer, args.lexicon_path)
    if lex_ids_tensor is not None:
        lex_ids_tensor = lex_ids_tensor.to(device)

    total_masked = 0
    correct1 = 0
    correctk = 0
    total_lex = 0
    correct1_lex = 0
    loss_sum = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dl:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, labels=labels)
            logits = out.logits
            loss_sum += out.loss.item()
            steps += 1

            # Only evaluate positions that were masked (labels != -100)
            mask = labels.ne(-100)
            if mask.sum().item() == 0:
                continue

            preds = logits.argmax(dim=-1)
            correct1 += preds.masked_select(mask).eq(labels.masked_select(mask)).sum().item()

            if args.topk and args.topk > 1:
                topk_idx = torch.topk(logits, k=args.topk, dim=-1).indices
                # Compare each masked label against top-k predictions
                masked_labels = labels.masked_select(mask)
                masked_topk = topk_idx.masked_select(mask.unsqueeze(-1)).view(-1, args.topk)
                correctk += (masked_topk.eq(masked_labels.unsqueeze(-1))).any(dim=-1).sum().item()

            total_masked += mask.sum().item()

            if lex_ids_tensor is not None:
                # Count masked positions whose ground-truth token is in the lexicon set
                # (This treats multi-token phrases as “any of their sub-tokens”.)
                lex_mask = mask & torch.isin(labels, lex_ids_tensor)
                if lex_mask.sum().item() > 0:
                    correct1_lex += preds.masked_select(lex_mask).eq(labels.masked_select(lex_mask)).sum().item()
                    total_lex += lex_mask.sum().item()

    avg_loss = loss_sum / max(steps, 1)
    acc1 = correct1 / max(total_masked, 1)
    acck = correctk / max(total_masked, 1) if (args.topk and args.topk > 1) else None
    acc1_lex = (correct1_lex / total_lex) if total_lex > 0 else None

    print(f"masked_tokens={total_masked}")
    print(f"avg_masked_loss={avg_loss:.4f}")
    print(f"Acc@1={acc1*100:.2f}%")
    if acck is not None:
        print(f"Acc@{args.topk}={acck*100:.2f}%")
    if acc1_lex is not None:
        print(f"Lexicon Acc@1 (on {total_lex} masked lexicon tokens) = {acc1_lex*100:.2f}%")

if __name__ == "__main__":
    main()

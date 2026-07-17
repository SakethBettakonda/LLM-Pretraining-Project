# scripts/dapt_train.py
import argparse, os, json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

class DataCollatorSelectiveMasking(DataCollatorForLanguageModeling):
    """
    MLM collator that increases masking probability for tokens in a given lexicon.
    Keeps expected overall mask rate approximately at target_mask_rate.
    """
    def __init__(self, tokenizer, mlm_probability, lex_ids_tensor,
                 mask_boost=3.0, target_mask_rate=0.15):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.lex_ids_tensor = lex_ids_tensor
        self.mask_boost = mask_boost
        self.target_mask_rate = target_mask_rate

    def torch_mask_tokens(self, inputs, special_tokens_mask=None, **kwargs):
        device = inputs.device
        # Mark positions whose ID is in our lexicon
        is_lex = (inputs.unsqueeze(-1) == self.lex_ids_tensor.to(device)).any(dim=-1)  # (B, L) bool

        base_p = self.mlm_probability
        p = torch.full_like(inputs, base_p, dtype=torch.float, device=device)
        boosted = min(base_p * self.mask_boost, 0.90)
        p[is_lex] = boosted


        # Rescale so global mean ~ target_mask_rate (prevents over-masking)
        cur_mean = p.mean()
        if cur_mean > 0:
            p = torch.clamp(p * (self.target_mask_rate / cur_mean), max=0.95)

        mask = torch.bernoulli(p).bool()

        # Respect special tokens/padding
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in inputs.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)

        mask = mask & ~special_tokens_mask & (inputs != self.tokenizer.pad_token_id)

        labels = inputs.clone()
        labels[~mask] = -100

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & mask
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% -> random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & mask & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=device)
        inputs[indices_random] = random_words[indices_random]

        # 10% -> keep original (still counted in loss)
        return inputs, labels


def build_trainer(args):
    raw = load_dataset("text", data_files={"train": args.train_path})
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    def tok_fn(ex): return tokenizer(ex["text"], truncation=True, max_length=args.seq_len)
    tok = raw.map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # ----- Variant B (optional loss weighting) -----
    logic_ids = None
    lex_ids_tensor = None  # <--- NEW: tensor used by selective masking (and could be used by weighting too)
    if os.path.exists(args.lexicon_path):
        with open(args.lexicon_path, encoding="utf-8-sig") as f:
            lex = json.load(f)

        # Build a *token-id* set from the lexicon phrases
        vocab = tokenizer.get_vocab()
        logic_id_set = set()
        for t in lex:
            if t in vocab:
                logic_id_set.add(vocab[t])
            # also add all subword ids that make up the phrase
            for i in tokenizer.encode(t, add_special_tokens=False):
                logic_id_set.add(i)

        if logic_id_set:
            lex_ids_tensor = torch.tensor(sorted(list(logic_id_set)), dtype=torch.long)
        if args.weighting:
            logic_ids = set(logic_id_set)
            print(f"[weighting] logic_ids={len(logic_ids)} from {len(lex)} terms")

    # ----- Choose collator (Variant C if requested) -----
    if args.selective_masking and lex_ids_tensor is not None:
        collator = DataCollatorSelectiveMasking(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            lex_ids_tensor=lex_ids_tensor,
            mask_boost=args.mask_boost,
            target_mask_rate=args.target_mask_rate,
        )
        print("[selective_masking] enabled with mask_boost="
              f"{args.mask_boost}, target_mask_rate={args.target_mask_rate}")
    else:
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            labels = inputs.get("labels")
            loss = outputs.loss
            if logic_ids is None or labels is None:
                return (loss, outputs) if return_outputs else loss

            logits = outputs.logits  # [B, T, V]
            vocab_size = logits.size(-1)

            logic_ids_tensor = torch.tensor(sorted(list(logic_ids)), device=labels.device)
            # positions where gold label is a logic token
            mask = (labels != -100) & torch.isin(labels, logic_ids_tensor)

            weights = torch.ones_like(labels, dtype=logits.dtype)
            weights = torch.where(mask, torch.full_like(weights, args.weight_factor), weights)

            ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = ce(logits.view(-1, vocab_size), labels.view(-1)).view_as(labels)
            token_loss = token_loss * weights

            denom = torch.sum((labels != -100).to(token_loss.dtype)) + 1e-8
            loss = torch.sum(token_loss) / denom
            return (loss, outputs) if return_outputs else loss

    TrainerClass = WeightedTrainer if (args.weighting and logic_ids is not None) else Trainer

    workers = 0 if os.name == "nt" else 4
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=workers,
        report_to=["none"],
    )

    return TrainerClass(model=model, args=training_args, train_dataset=tok["train"], data_collator=collator), tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="roberta-base")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--grad_checkpoint", action="store_true")
    # Variant B (optional)
    p.add_argument("--weighting", action="store_true")
    p.add_argument("--weight_factor", type=float, default=2.0)
    p.add_argument("--lexicon_path", type=str, default="lexicon.json")
    # Variant C (selective masking)
    p.add_argument("--selective_masking", action="store_true",
                   help="Bias MLM masking probability toward lexicon tokens.")
    p.add_argument("--mask_boost", type=float, default=3.0,
                   help="Multiply base mask prob for tokens in lexicon.")
    p.add_argument("--target_mask_rate", type=float, default=0.15,
                   help="Target overall masked fraction per sequence.")
    return p.parse_args()


def main():
    args = parse_args()
    trainer, tokenizer = build_trainer(args)
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("✅ Training complete ->", args.out_dir)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

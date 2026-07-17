# Logic-Aware Language Model Pretraining: DAPT + ELECTRA-Style Discriminator Training

This project explores two complementary pretraining strategies for improving a language model's sensitivity to **logical and reasoning-related language** — connectives, conditionals, causation, quantifiers, implications, and related linguistic markers. It combines **Domain-Adaptive Pretraining (DAPT)** on RoBERTa with a custom **ELECTRA-style generator/discriminator** pipeline trained from scratch, both targeted at a curated lexicon of logical-reasoning terms.

## Motivation

Standard masked language modeling (MLM) treats all tokens equally when deciding what to mask and how heavily to weight the loss. But tokens that carry logical structure (e.g. *"because," "unless," "implies," "therefore"*) are often rarer and more semantically load-bearing than the average token. This project tests whether **biasing pretraining toward these tokens** — via selective masking, loss reweighting, and adversarial discriminator training — improves a model's ability to recognize and predict logic-bearing language, without materially hurting general-purpose performance.

## Project Structure

```
.
├── scripts/
│   ├── dapt_train.py           # Domain-Adaptive Pretraining (RoBERTa) with 3 variants
│   ├── eval_mlm_accuracy.py    # Masked-token accuracy eval (overall + lexicon-specific)
│   └── eval_mlm_loss.py        # Masked-LM loss / pseudo-perplexity eval
├── notebooks/
│   └── Electra_style_Pretraining.ipynb   # ELECTRA-style generator/discriminator training from scratch
├── results/
│   ├── weighted_dapt_diagram.png   # DAPT before/after accuracy & loss comparison
│   ├── electra_graphs.png          # ELECTRA training loss curves + accuracy gains
│   └── electra_results.png         # ELECTRA discriminator accuracy by logic category
└── README.md
```

## Part 1: Domain-Adaptive Pretraining (DAPT) on RoBERTa

`scripts/dapt_train.py` fine-tunes `roberta-base` on a logic-focused corpus using masked language modeling, with three configurable strategies:

- **Variant A — Baseline DAPT:** standard MLM fine-tuning (15% random masking).
- **Variant B — Loss Weighting:** up-weights the cross-entropy loss for tokens found in a curated logic lexicon (`--weighting --weight_factor`), so the model is penalized more for getting logic-bearing tokens wrong.
- **Variant C — Selective Masking:** biases the *masking probability itself* toward lexicon tokens (`--selective_masking --mask_boost`), while rescaling probabilities to keep the overall mask rate near the target (default 15%), so logic tokens are seen (and predicted) more often during training without over-masking the sequence.

```bash
python scripts/dapt_train.py \
    --model roberta-base \
    --train_path data/logic_corpus.txt \
    --out_dir out/weighted_dapt \
    --lexicon_path lexicon.json \
    --selective_masking --mask_boost 3.0 \
    --weighting --weight_factor 2.0 \
    --max_steps 20000 --bf16
```

### Evaluation

`eval_mlm_accuracy.py` reports overall Top-1/Top-5 masked-token accuracy *and* accuracy restricted to lexicon tokens, so you can see whether gains are concentrated where intended:

```bash
python scripts/eval_mlm_accuracy.py \
    --model out/weighted_dapt \
    --data data/holdout.txt \
    --lexicon_path lexicon.json \
    --topk 5
```

`eval_mlm_loss.py` reports average masked-LM loss and pseudo-perplexity for quick before/after comparison.

### Results

| Model | Lexicon Top-1 Acc | Overall Top-1 Acc | Overall Top-5 Acc | Avg MLM Loss |
|---|---|---|---|---|
| RoBERTa-Base (before) | 89.56% | 70.33% | 85.70% | 1.43 |
| DAPT 20k (after) | 90.22% | 70.41% | 85.74% | 1.41 |
| Weighted DAPT 15k (after) | **91.75%** | **70.48%** | **85.80%** | **1.41** |

Weighted/selective-masking DAPT improves accuracy specifically on logic-lexicon tokens (+2.2 pts) while general-purpose masked-token accuracy stays essentially flat — showing the targeted approach doesn't trade off broad language modeling ability for domain focus, even with fewer training steps (15k vs 20k) than the baseline DAPT run.

## Part 2: ELECTRA-Style Generator/Discriminator Pretraining

The notebook (`Electra_style_Pretraining.ipynb`) implements an ELECTRA-style setup **from scratch** on top of `bert-base-uncased` tokenization:

1. **Generator** — a scaled-down BERT (25% hidden size/layer ratio) trained with standard MLM to propose plausible replacement tokens for masked positions.
2. **Discriminator** — a full-size BERT encoder + binary classification head, trained adversarially to detect which tokens in a sequence were *replaced by the generator* vs. original.
3. **Training data** — ~200k samples combining general web/Wikipedia text with logic-focused text, split 180k train / 20k validation, with masking targeted at a 14-category logical-reasoning taxonomy (basic connectives, conditionals, implications, causation, quantifiers, temporal logic, contrasts, alternatives, modals, comparisons, negations, existential, certainty, and additions).
4. **Loss** — combined generator MLM loss + discriminator replaced-token-detection loss (discriminator loss weighted 50x to counter its low base magnitude relative to the generator).

### Results

| Metric | Before Training | After Training | Δ |
|---|---|---|---|
| Generator Loss | 10.36 | 6.03 | -4.33 |
| Discriminator Loss | 0.71 | 0.14 | -0.57 |
| Total Weighted Loss | 45.87 | 13.01 | -32.86 |
| Generator Accuracy | 0.0% | 9.4% | +9.4 pts |
| **Discriminator Accuracy** | **38.5%** | **96.4%** | **+57.9 pts** |

The discriminator's ability to distinguish real vs. generator-replaced tokens improves dramatically (38.5% → 96.4%), with per-category detection scores showing the strongest signal on **causation** and **quantifier** tokens — the categories with the richest surface-level cues for a discriminator to key off of.

![Training curves](results/electra_graphs.png)
![Category-level detection](results/electra_results.png)

## Key Takeaways

- Targeting pretraining toward a specific linguistic phenomenon (logical reasoning language) via selective masking and loss weighting yields measurable, targeted gains without degrading general MLM performance.
- An ELECTRA-style discriminator is highly effective at learning to detect replaced tokens once trained, but detection strength varies significantly by logical category — suggesting some reasoning categories (causation, quantifiers) are linguistically more "detectable" than others (existential, modals).
- Together, these experiments suggest that **lightweight, targeted interventions during pretraining** (rather than full architecture changes) can meaningfully shift what a language model is sensitive to.

## Tech Stack

`PyTorch` · `HuggingFace Transformers` · `HuggingFace Datasets` · `RoBERTa` / `BERT` · Custom ELECTRA-style training loop

## Future Work

- Extend the logic lexicon and evaluate on downstream reasoning benchmarks (e.g. logical entailment, NLI).
- Ablate the discriminator loss weighting factor (currently fixed at 50x) to find a more principled balance.
- Compare against a full from-scratch ELECTRA pretraining run at larger scale.

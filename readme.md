# The Format Is the Circuit

**How prompt structure externalizes computation and creates attention circuits in transformer models.**

Independent mechanistic interpretability research · Ryan Brady · 2026

---

## TL;DR

Structured prompt format is not neutral scaffolding. It determines what circuits a model builds, what computations get externalized into tokens, and what the model actually has to compute internally.

Path patching identifies circuits that correlate with a computation. Correlation does not distinguish between circuits that **compute** something and circuits that **read** a pre-computed externalized value. Causal ablation is necessary to tell them apart — and the distinction matters for interpreting what a model has actually learned.

## Core finding

GPT-2 XL's base weights — with zero fine-tuning — already show operation-specific diagonal attention structure the moment a geometric constraint routing (GCR) format is applied. Training sharpens this structure; it does not create it.

The attention figure (Figure 1, center column) is the primary mechanistic evidence. The circuits that path patching later identifies as "carry-specific" were not learned from arithmetic training. They were imposed by token geometry and then amplified by training.

## Seven experiments

| # | Experiment | Result |
|---|---|---|
| 1 | Arithmetic (GPT-2 XL) | 35% → 80% with LoRA; ADD/SUB circuits share 0/10 heads; 16-head and 48-MLP ablations produce zero selective accuracy change |
| 2 | Logic (standard format) | 55% overall; near-uniform True-prediction bias (19/20 prompts have positive True–False gap) |
| 3 | Format relocation | Layer-level carry specialization is format-invariant (peaks fixed at L26, L35); head-level specialization shifts with carry token position |
| 4 | Opaque format probe | True-bias identical across 1/0, T/F, cat/dog; inverted labels collapse AND to 0% — bias is a pretraining semantic prior, not a token-identity shortcut |
| 5 | Contrastive vs RL training | Contrastive: 100% AND trained-format, 50% inverted. RL (REINFORCE, no NTP loss): 67% AND, format-invariant across standard, numeric, inverted |
| 6 | Vowel counting (no externalization possible) | 8% → 50% with 0.17% of parameters changed; residual-stream probes show count information developing L4 → L11 |
| 7 | Cross-architecture (Mistral-7B) | Format-driven routing begins at layer 0, same as GPT-2 XL |

## Two regimes

| Regime | Circuit property | Ablation | Example |
|---|---|---|---|
| **Externalized** | Distributed, redundant token readers | Robust (0% change under 16-head / 48-MLP ablation) | Arithmetic carry, logic variable values |
| **Internal** | Organized, probeable representations | Expected fragile | Vowel counting, operator semantics (when trained via RL/contrastive) |
| **Mixed shortcut** | Readers that fail when the shortcut contradicts the answer | Robust to ablation (still reading the wrong tokens) | Logic AND/OR under standard NTP training |

## Why the training objective matters

Standard next-token prediction on well-formatted data tends to install shortcuts — the model learns to read the format's explicit value tokens rather than evaluate operators. Contrastive data (where value-token shortcuts fail 50% of the time by construction) and REINFORCE reward signals (gradient only from answer correctness) both push toward operator evaluation. RL produces a False-leaning prior (−1.43 gap vs. the original +0.70) that is **operator-appropriate** rather than format-appropriate, generalizing across format variants because it was learned from correctness rather than token patterns.

## Practical implications

- **Prompt format selection is circuit selection.** Writing `c=1` as an explicit token vs. leaving carry implicit chooses between a model that reads carry from context and a model that must compute it internally.
- **Pretraining priors survive format variation.** Using True/False as logic labels activates pretraining associations that fine-tuning cannot easily override — swapping to 1/0 or cat/dog does not remove them.
- **Capability evaluation under opaque formats** reveals the degree to which a demonstrated capability is genuine vs. format-dependent. Models that appear to have learned arithmetic or logic via structured prompting may have learned token-pattern matching on the intermediate values the format supplies.

## Repository contents

- **`GCR_paper_v8.md`** — full paper (v8)
- **`figures/`** — attention maps (Figure 1), path-patching heatmaps, layer-wise specificity profiles, residual-stream divergence plots, training-comparison figures

## Status

Complete. Targeting BlackboxNLP and Findings of ACL.

## Citation

```bibtex
@misc{brady2026format,
  title  = {The Format Is the Circuit: How Prompt Structure Externalizes
            Computation and Creates Attention Circuits in Transformer Models},
  author = {Brady, Ryan},
  year   = {2026},
  note   = {Independent research},
  url    = {https://github.com/ardinpo/<repo-name>}
}
```

## Contact

Ryan Brady — independent researcher
GitHub: [@ardinpo](https://github.com/ardinpo)

# OOD Improvement Plan for CPC Finetuning

## Round 1 Results: Finetuning Techniques (completed)

Rankings by OOD average accuracy (mean over seeds 42-44):

### HumanActivityRecognition

| # | Trial | ID Acc | X-Env | X-User | X-Dev | OOD Avg |
|---|---|---|---|---|---|---|
| 1 | **dann_mixup** | 0.776 | **0.433** | **0.533** | 0.406 | **0.458** |
| 2 | dann_xdomain_mixup | 0.777 | 0.430 | 0.529 | 0.399 | 0.453 |
| 3 | mixup | 0.786 | 0.423 | 0.528 | 0.406 | 0.452 |
| 4 | dann_sam | 0.738 | 0.418 | 0.517 | 0.400 | 0.445 |
| 5 | freeze | 0.729 | 0.439 | 0.514 | 0.380 | 0.444 |
| 6 | dann | 0.738 | 0.419 | 0.524 | 0.390 | 0.444 |
| 7 | dann_swa | 0.741 | 0.417 | 0.520 | 0.392 | 0.443 |
| 8 | dann_coral | 0.746 | 0.407 | 0.518 | 0.379 | 0.435 |
| 9 | sam | **0.793** | 0.403 | 0.512 | 0.352 | 0.422 |
| 10 | baseline | 0.789 | 0.396 | 0.513 | 0.345 | 0.418 |

### HumanIdentification

| # | Trial | ID Acc | X-Env | X-User | X-Dev | OOD Avg |
|---|---|---|---|---|---|---|
| 1 | **freeze** | 0.919 | 0.243 | 0.000 | **0.418** | **0.330** |
| 2 | dann_xdomain_mixup | 0.989 | **0.259** | 0.000 | 0.372 | 0.315 |
| 3 | dann_coral | 0.986 | 0.254 | 0.000 | 0.369 | 0.312 |
| 4 | dann | 0.988 | 0.253 | 0.000 | 0.364 | 0.308 |
| 5 | dann_sam | 0.990 | 0.244 | 0.000 | 0.362 | 0.303 |
| 6 | dann_mixup | 0.988 | 0.249 | 0.000 | 0.360 | 0.304 |
| 7 | mixup | 0.990 | 0.269 | 0.000 | 0.334 | 0.301 |
| 8 | baseline | 0.989 | 0.231 | 0.000 | 0.333 | 0.282 |

*Cross-user = 0.0 everywhere: test users are unseen identities (expected).*

### ProximityRecognition

| # | Trial | ID Acc | X-Env | X-User | X-Dev | OOD Avg |
|---|---|---|---|---|---|---|
| 1 | **dann_sam** | 0.671 | 0.307 | 0.291 | **0.358** | **0.319** |
| 2 | dann_mixup | 0.706 | 0.312 | 0.293 | 0.350 | 0.318 |
| 3 | mixup | 0.716 | 0.302 | 0.290 | 0.359 | 0.317 |
| 4 | dann | 0.667 | **0.318** | 0.290 | 0.342 | 0.317 |
| 5 | dann_manifold_mixup | 0.707 | 0.309 | **0.295** | 0.345 | 0.317 |
| 6 | dann_xdomain_mixup | 0.702 | 0.303 | 0.295 | 0.351 | 0.316 |
| 7 | baseline | 0.712 | 0.299 | 0.294 | 0.341 | 0.312 |

---

## Key Takeaway from Round 1

Finetuning-stage techniques give **marginal** improvements (<1pp OOD avg). The ceiling
is in the **CPC pretraining itself** — the encoder has already learned domain-specific
features, and no amount of finetuning regularization will fully undo that.

What *did* help: techniques that limit overfitting to training domains (freeze, DANN,
mixup). But they're all fighting the same upstream problem.

---

## Round 2: What To Do Next (highest impact first)

### 1. TENT: Test-Time Entropy Minimization (implemented, ready to run)

**Paper:** Wang et al., "Fully Test-Time Adaptation by Entropy Minimization" (ICLR 2021)

At inference time, adapt the model's BatchNorm parameters by minimizing the entropy of
predictions on each test batch. This is **free performance** — no retraining needed, just
run it on your existing best checkpoints.

CPC's `g_enc` has BatchNorm1d layers, so TENT can directly adapt them to the target domain's
statistics. This should help cross-device and cross-environment most, since those shifts
change the signal distribution that BatchNorm captures.

```bash
# Run TENT on your best checkpoints (dann_mixup example):
python scripts/evaluate_ood.py \
  --weights results/HumanActivityRecognition/cpc/dann_mixup_s42/best_model.pt \
  --config results/HumanActivityRecognition/cpc/dann_mixup_s42/cpc_HumanActivityRecognition_config.yaml \
  --pipeline supervised --model cpc --use_dann \
  --tasks HumanActivityRecognition \
  --tent --tent_steps 1 --tent_lr 1e-3

# Also try TTA (test-time augmentation) for comparison:
python scripts/evaluate_ood.py \
  --weights results/HumanActivityRecognition/cpc/dann_mixup_s42/best_model.pt \
  --config results/HumanActivityRecognition/cpc/dann_mixup_s42/cpc_HumanActivityRecognition_config.yaml \
  --pipeline supervised --model cpc --use_dann \
  --tasks HumanActivityRecognition \
  --tta --tta_rounds 10
```

Try TENT on your top checkpoints across all 3 tasks. Tune `--tent_lr` (1e-4 to 1e-2)
and `--tent_steps` (1 to 3).

### 2. Freeze + Domain Alignment Combos (ready to run)

`freeze` was #1 for HumanID OOD. Combining frozen backbone with DANN+CORAL or
DANN+cross-domain mixup could stack the benefits: frozen features prevent overfitting,
DANN/CORAL heads push for domain invariance in the classification heads.

```bash
python scripts/run_experiment_trials.py \
  --encoder pretrain_results/all_tasks/cpc/6b8fd6c79b/encoder_weights.pt \
  --tasks all \
  --only freeze_coral freeze_xdomain_mixup freeze_sam
```

### 3. Kitchen Sink (bug fixed, ready to re-run)

The manifold_mixup + SAM bug has been fixed. Re-run:

```bash
python scripts/run_experiment_trials.py \
  --encoder pretrain_results/all_tasks/cpc/6b8fd6c79b/encoder_weights.pt \
  --tasks all \
  --only kitchen_sink
```

### 4. Domain-Aware CPC Pretraining (IMPLEMENTED, ready to run)

The root cause of the OOD gap is that CPC pretraining learns domain-specific features.
Domain-aware CPC uses domain labels as **hard negatives** to fix this at the source:

- **Hard negatives:** Same (env, device) domain, different content — forces the encoder
  to learn features that distinguish *activity* from *domain-specific signal characteristics*
  (e.g., device noise floor, environment multipath).
- **Same-user masking:** Negatives from the same user are masked out to prevent false
  negatives (two samples from the same user doing the same activity aren't true negatives).
- `domain_neg_ratio` controls the fraction of negatives from the same domain (default 0.5).
  The rest are random for diversity.

**How it works:** Standard CPC negatives are random across the whole batch, so the model
can "cheat" by using domain cues (device hardware signature, room acoustics) to reject
negatives without learning semantic content. By forcing half the negatives to come from
the *same* domain, the model can't distinguish positive from negative based on domain alone
— it must learn the actual content (activity, proximity, identity).

```bash
# Retrain CPC encoder with domain-aware hard negatives:
python scripts/pretrain.py \
  --pretrain_method cpc \
  --all_tasks \
  --domain_aware \
  --domain_neg_ratio 0.5 \
  --cpc_num_negatives 256 \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --save_dir pretrain_results

# Then finetune with your best technique (e.g., dann_mixup):
python scripts/run_experiment_trials.py \
  --encoder pretrain_results/all_tasks/cpc/<new_experiment_id>/encoder_weights.pt \
  --tasks all \
  --only dann_mixup freeze dann
```

**Tuning tips:**
- `--domain_neg_ratio 0.5` is a good starting point. Higher (0.7-0.8) = harder negatives,
  may slow convergence but produce more domain-invariant features.
- `--cpc_num_negatives 256` can be increased to 512 if batch sizes are large.
- Compare val loss curves: domain-aware should converge slower but to a similar final loss.
  If it diverges, lower `domain_neg_ratio` or increase `cpc_num_negatives`.

**Files modified:** `engine/cpc_trainer.py` (hard negative sampling in `compute_cpc_loss`),
`load/pretrain_dataset.py` (returns env/device labels), `scripts/pretrain.py` (CLI flags).

---

## Techniques Implemented (reference)

| Technique | Paper | Flag | Where |
|---|---|---|---|
| SAM optimizer | Foret et al. ICLR 2021 | `--use_sam --sam_rho 0.05` | `utils/sam.py` |
| CORAL loss | Sun & Saenko ECCV 2016 | `--coral_weight 0.5` | `engine/dann_trainer.py` |
| Manifold Mixup | Verma et al. ICML 2019 | `--manifold_mixup` | `model/models.py` |
| Cross-Domain Mixup | Xu et al. AAAI 2020 | `--cross_domain_mixup` | `engine/dann_trainer.py` |
| DANN weight tuning | - | `--lambda_user/env/device` | `engine/dann_trainer.py` |
| TENT | Wang et al. ICLR 2021 | `--tent --tent_lr 1e-3` | `scripts/evaluate_ood.py` |
| TTA (augmentation) | - | `--tta --tta_rounds 10` | `scripts/evaluate_ood.py` |
| Domain-Aware CPC | - | `--domain_aware --domain_neg_ratio 0.5` | `engine/cpc_trainer.py` |

# Experiment configs for CPC finetune trials

Use these with a pretrained CPC encoder and the trial runner script.

## Configs

| File | Description |
|------|-------------|
| `cpc_baseline.yaml` | Base: 100 epochs, lr=0.001, no mixup, no SWA, label_smoothing=0.1 |
| `cpc_mixup.yaml` | Same as baseline + mixup_alpha=0.2 |
| `cpc_swa.yaml` | Same as baseline + swa_epochs=20, swa_lr=1e-4 |

## Trial runner

From project root:

```bash
# All 8 trials (baseline, freeze, dann, freeze_dann, mixup, dann_mixup, swa, dann_swa), 1 task, 3 seeds
python scripts/run_experiment_trials.py --encoder pretrain_results/all_tasks/cpc/6b8fd6c79b

# All 3 OOD tasks
python scripts/run_experiment_trials.py --encoder pretrain_results/all_tasks/cpc/6b8fd6c79b --tasks all

# Only baseline and DANN
python scripts/run_experiment_trials.py --encoder path/to/encoder.pt --only baseline dann

# Custom seeds
python scripts/run_experiment_trials.py --encoder path/to/encoder.pt --seeds 42 43 44 45
```

Results go under `results/<task>/cpc/params_*/`. A summary is written to `results/experiment_trials_summary.json`.

To evaluate with TTA after training:

```bash
python scripts/evaluate_ood.py --weights results/<Task>/cpc/params_XXX/best_model.pt \
  --config configs/experiments/cpc_baseline.yaml --pipeline supervised --tasks <Task> --tta --tta_rounds 10
```

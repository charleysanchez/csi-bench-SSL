"""
Central registry of model types for supervised and pretrain pipelines.
Scripts should import MODEL_TYPES or PRETRAIN_MODEL_TYPES from here.
"""
from model.models import (
    MLPClassifier,
    LSTMClassifier,
    ResNet18Classifier,
    TransformerClassifier,
    ViTClassifier,
    PatchTST,
    TimesFormer1D,
    CPCClassifier,
    MaskedLSTM,
    MaskedTransformer,
    MaskedPatchTST,
    MaskedTimesFormer1D,
    CPCModel,
)

# Classification models: used by train_supervised, train_multitask_adapter, evaluate_ood
MODEL_TYPES = {
    "mlp": MLPClassifier,
    "lstm": LSTMClassifier,
    "resnet18": ResNet18Classifier,
    "transformer": TransformerClassifier,
    "vit": ViTClassifier,
    "patchtst": PatchTST,
    "timesformer1d": TimesFormer1D,
    "cpc": CPCClassifier,
}

# Pretraining models: used by pretrain.py (masked or CPC)
PRETRAIN_MODEL_TYPES = {
    "lstm": MaskedLSTM,
    "patchtst": MaskedPatchTST,
    "timesformer1d": MaskedTimesFormer1D,
    "transformer": MaskedTransformer,
    "cpc": CPCModel,
}

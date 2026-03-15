"""
Shared model wrappers for multitask (Transformer embedding) and DANN.
Used by train_multitask_adapter, evaluate_ood, and train_supervised when wrapping for DANN.
"""
import torch.nn as nn
from model.models import DANNTransformer


class ConfigDict(dict):
    """Dict with attribute-style access (e.g. backbone.config.hidden_size)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")


class TransformerEmbedding(nn.Module):
    """Wraps TransformerClassifier to output a single embedding vector (mean over time)."""

    def __init__(self, cls_model):
        super().__init__()
        self.input_proj = cls_model.input_proj
        self.pos_encoder = cls_model.pos_encoder
        self.transformer = cls_model.transformer

    def forward(self, x):
        x = x.squeeze(1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x.mean(dim=1)


def wrap_backbone_for_multitask_transformer(backbone_model, args):
    """
    Wrap a TransformerClassifier for SimpleMultiTaskModel.
    Returns (backbone_module, task_classes_dict) is not returned; caller has it.
    """
    backbone = TransformerEmbedding(backbone_model)
    d_model = getattr(args, "emb_dim", 256)
    num_layers = len(backbone_model.transformer.layers)
    backbone.config = ConfigDict(
        model_type="bert",
        hidden_size=d_model,
        num_attention_heads=8,
        num_hidden_layers=num_layers,
        intermediate_size=d_model * 4,
        hidden_dropout_prob=getattr(args, "dropout", 0.1),
        use_return_dict=False,
    )
    return backbone


def wrap_for_dann(model, num_classes, args, num_users=65, num_envs=27, num_devices=37):
    """
    Strip the classification head from `model` and wrap in DANNTransformer.
    Uses the model's embedding size when available (e.g. CPC has hidden_size=256).
    Domain head sizes must match load/dataset.py GLOBAL_*_MAPPING + Unknown: 65 users, 27 envs, 37 devices.
    """
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    elif hasattr(model, "head"):
        model.head = nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()
    # CPC and others use hidden_size; transformer config uses emb_dim
    embed_dim = getattr(model, "hidden_size", None) or getattr(model, "d_model", None) or getattr(args, "emb_dim", 256)
    return DANNTransformer(
        base_encoder=model,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_users=num_users,
        num_envs=num_envs,
        num_devices=num_devices,
        grl_alpha=1.0,
    )

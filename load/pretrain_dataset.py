from load.dataset import CSIDataset as _CSIDataset

class PretrainDataset(_CSIDataset):
    """Wraps CSIDataset, ignoring label/metadata errors — labels unused during pretraining."""
    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
        except KeyError as e:
            raise  # let the outer try/except in the loading loop handle it

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            return None

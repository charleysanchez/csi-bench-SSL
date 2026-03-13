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
            result = super().__getitem__(index)
            if result is None:
                return None
            
            csi, label = result
            row = self.split_metadata.iloc[index]
            user = row.get("user", f"unknown_user_{index}")

            label_key = str(label)

                # Handle decimal → binary label conversion (e.g. Localization)
            if label_key not in self.label_mapper["label_to_idx"]:
                dec_to_bin = self.label_mapper.get("decimal_to_binary", {})
                label_key = dec_to_bin.get(label_key, label_key)

            label_idx = self.label_mapper["label_to_idx"][label_key]
            
            return csi, label, user
        except Exception:
            return None

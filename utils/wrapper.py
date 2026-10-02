import torch
import torch.nn as nn

from src.model import get_model


class ModelWrapper(nn.Module):
    def __init__(self, name, embed_dim=384, ratio=16, ckpt_path: str = None):
        super().__init__()

        self.name = name
        self.embed_dim = embed_dim
        self.ratio = ratio

        self.model = get_model(name, feature_dim=embed_dim, ratio=ratio)

        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

    def forward(self, image, features, output_size):
        out = self.model(image, features, output_size)
        return out

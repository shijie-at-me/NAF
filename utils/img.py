# import cv2

import torch
from einops import rearrange

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def round_to_nearest_multiple(value, multiple=14):
    return multiple * round(value / multiple)


def create_coordinate(h, w, start=0, end=1, device="cuda", dtype=torch.float32):
    # Create a grid of coordinates
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    # Create a 2D map using meshgrid
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    # Stack the x and y coordinates to create the final map
    coord_map = torch.stack([xx, yy], axis=-1)[None, ...]
    coords = rearrange(coord_map, "b h w c -> b (h w) c", h=h, w=w)
    return coords

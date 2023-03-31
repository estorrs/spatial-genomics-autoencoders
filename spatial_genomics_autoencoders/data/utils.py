import os
import re

import numpy as np
import seaborn as sns
import scanpy as sc
import torch
import torchvision.transforms.functional as TF
import numpy as np
from einops import rearrange


def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_means_and_stds(adatas):
    means, stds = None, None
    for a in adatas:
        x = rearrange(next(iter(a.uns['spatial'].values()))['images']['lowres'], 'h w c -> c h w')
        x = TF.convert_image_dtype(torch.tensor(x))

        if means is None:
            means = x.mean((1, 2))
            stds = x.std((1, 2))
        else:
            means = (means + x.mean((1, 2))) / 2
            stds = (stds + x.std((1, 2))) / 2
    return means, stds


def flexible_rescale(img, scale=.5, size=None):
    if size is None:
        size = int(img.shape[-2] * scale), int(img.shape[-1] * scale)

    if not isinstance(img, torch.Tensor):
        is_tensor = False
        img = torch.tensor(img)
    else:
        is_tensor = True

    img = TF.resize(img, size=size)

    if not is_tensor:
        img = img.numpy()

    return img

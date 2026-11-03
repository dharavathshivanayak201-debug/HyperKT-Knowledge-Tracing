"""Strict, CPU-safe loading of the final Stage 9E/10 model family."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .models import DKT, HyperKT, HyperKTGated


def read_structure(path, n_skills=151):
    with np.load(path, allow_pickle=False) as data:
        assignments = data["assignments"].astype(np.float32)
        ids = data["skill_ids"].astype(int)
    matrix = np.zeros((n_skills, assignments.shape[1]), dtype=np.float32)
    matrix[ids] = assignments
    return matrix


def make_model(variant, structure, n_skills=151):
    if variant == "DKT_matched":
        return DKT(n_skills)
    if variant == "HyperKT_matched":
        return HyperKT(n_skills, structure, hyper=True, decay=True)
    if variant == "HyperKT_gated":
        return HyperKTGated(n_skills, structure)
    raise ValueError(f"Unknown final variant: {variant}")


def load_checkpoint(path, variant, structure, expected_sha256):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Checkpoint hash mismatch: {path.name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    if not all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value)):
        raise ValueError("Nonfinite checkpoint state")
    model = make_model(variant, structure)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint

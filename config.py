"""
config.py — Configuración centralizada de hiperparámetros.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Datos ──────────────────────────────────────────────────────────────────
    drive_root:  str = "data/DRIVE"
    stare_root:  str = "data/STARE"
    chase_root:  str = "data/CHASE_DB1"

    # ── Modelo ─────────────────────────────────────────────────────────────────
    arch:         str   = "unet"         # unet | att_unet | unetpp
    in_channels:  int   = 3
    out_channels: int   = 1
    base_filters: int   = 64
    dropout:      float = 0.1
    bilinear:     bool  = True

    # ── Entrenamiento ──────────────────────────────────────────────────────────
    epochs:        int   = 100
    batch_size:    int   = 4
    lr:            float = 1e-4
    weight_decay:  float = 1e-5
    grad_clip:     float = 1.0
    patience:      int   = 15           # early stopping
    amp:           bool  = True         # mixed precision (FP16)
    num_workers:   int   = 4

    # ── Pérdida ────────────────────────────────────────────────────────────────
    loss:       str   = "combined"       # bce | dice | combined | focal | tversky
    loss_alpha: float = 0.5             # peso BCE en pérdida combinada

    # ── Preprocesamiento ───────────────────────────────────────────────────────
    use_clahe:   bool  = True
    clahe_clip:  float = 2.0
    img_size:    int   = 512

    # ── Evaluación ─────────────────────────────────────────────────────────────
    threshold:   float = 0.5
    use_tta:     bool  = True
    tta_n:       int   = 8

    # ── Salida ─────────────────────────────────────────────────────────────────
    save_dir:    str   = "checkpoints"
    results_dir: str   = "results"

    # ── Reproducibilidad ───────────────────────────────────────────────────────
    seed:        int   = 42

    def __post_init__(self):
        import torch
        import random
        import numpy as np
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def describe(self):
        """Imprime un resumen legible de la configuración."""
        lines = [
            "=" * 50,
            "CONFIGURACIÓN DEL EXPERIMENTO",
            "=" * 50,
            f"  Arquitectura : {self.arch}",
            f"  Base filters : {self.base_filters}",
            f"  Pérdida      : {self.loss} (alpha={self.loss_alpha})",
            f"  CLAHE        : {self.use_clahe}",
            f"  Img size     : {self.img_size}",
            f"  Epochs       : {self.epochs}",
            f"  Batch size   : {self.batch_size}",
            f"  LR           : {self.lr}",
            f"  Dropout      : {self.dropout}",
            f"  TTA          : {self.use_tta} (n={self.tta_n})",
            "=" * 50,
        ]
        print("\n".join(lines))

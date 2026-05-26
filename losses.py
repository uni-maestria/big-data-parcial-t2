"""
losses.py — Funciones de pérdida para segmentación de vasos retinianos.
Implementa BCE, Dice Loss, combinaciones y Focal Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    fov_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dice Loss para segmentación binaria.

    D = 1 - (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)

    Args:
        pred: logits [B, 1, H, W] (sin sigmoide).
        target: máscara binaria [B, 1, H, W] en {0, 1}.
        smooth: término de suavizado para evitar división por cero.
        fov_mask: si se provee, el cálculo solo considera píxeles dentro del FOV.
    """
    pred_sig = torch.sigmoid(pred)

    if fov_mask is not None:
        pred_sig = pred_sig * fov_mask
        target   = target   * fov_mask

    pred_flat   = pred_sig.view(pred_sig.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union        = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    fov_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Binary Cross-Entropy con logits y peso de clase positiva.

    Args:
        pred: logits [B, 1, H, W].
        target: máscara [B, 1, H, W].
        pos_weight: tensor [1] para compensar desbalance de clases.
        fov_mask: máscara del campo visual para calcular la pérdida solo en el FOV.
    """
    if pos_weight is not None:
        pos_weight = pos_weight.to(pred.device)

    if fov_mask is not None:
        # Reducción manual con máscara
        loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight, reduction='none'
        )
        # Solo los píxeles dentro del FOV contribuyen
        return (loss * fov_mask).sum() / (fov_mask.sum() + 1e-7)
    else:
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )


class CombinedLoss(nn.Module):
    """
    Pérdida combinada: α·BCE + (1-α)·Dice.
    Recomendada por múltiples trabajos de segmentación médica.
    El BCE garantiza convergencia estable; el Dice optimiza directamente el F1.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha      = alpha
        self.pos_weight = pos_weight
        self.smooth     = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fov_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_bce  = bce_loss(pred, target, self.pos_weight, fov_mask)
        l_dice = dice_loss(pred, target, self.smooth, fov_mask)
        return self.alpha * l_bce + (1.0 - self.alpha) * l_dice


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — penaliza más los ejemplos fáciles.
    Especialmente útil para vasos capilares finos.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fov_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce    = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        prob   = torch.sigmoid(pred)
        p_t    = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal  = alpha_t * ((1 - p_t) ** self.gamma) * bce

        if fov_mask is not None:
            return (focal * fov_mask).sum() / (fov_mask.sum() + 1e-7)
        return focal.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalización del Dice que penaliza asimétricamente
    falsos positivos y falsos negativos.
    α > β penaliza más los FN (útil para vasos finos).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                fov_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = torch.sigmoid(pred)
        if fov_mask is not None:
            p, target = p * fov_mask, target * fov_mask

        pf = p.view(p.size(0), -1)
        tf = target.view(target.size(0), -1)

        tp = (pf * tf).sum(dim=1)
        fp = (pf * (1 - tf)).sum(dim=1)
        fn = ((1 - pf) * tf).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


def build_loss(name: str, pos_weight: Optional[torch.Tensor] = None, **kwargs) -> nn.Module:
    """Instancia la función de pérdida por nombre."""
    registry = {
        "bce":      lambda: nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "dice":     lambda: (lambda pred, target, fov=None: dice_loss(pred, target, fov_mask=fov)),
        "combined": lambda: CombinedLoss(pos_weight=pos_weight, **kwargs),
        "focal":    lambda: FocalLoss(**kwargs),
        "tversky":  lambda: TverskyLoss(**kwargs),
    }
    if name not in registry:
        raise ValueError(f"Pérdida '{name}' no reconocida. Opciones: {list(registry)}")
    return registry[name]()

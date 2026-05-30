"""
evaluate.py — Evaluación completa: métricas, TTA y experimento de domain shift.
Implementa: Sensibilidad, Especificidad, F1, AUC-ROC, y Test-Time Augmentation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional, Dict, List
from sklearn.metrics import roc_auc_score, roc_curve


# ──────────────────────────────────────────────────────────────────────────────
# Métricas píxel a píxel
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_probs: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    fov_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calcula métricas de segmentación binaria.

    Args:
        pred_probs: probabilidades predichas [H, W] en [0,1].
        target: máscara binaria [H, W] en {0,1}.
        threshold: umbral de binarización.
        fov_mask: si se provee, solo se evalúan los píxeles dentro del FOV.

    Returns:
        dict con sensitivity, specificity, f1, precision, auc_roc, accuracy.
    """
    if fov_mask is not None:
        idx    = fov_mask.astype(bool).flatten()
        probs  = pred_probs.flatten()[idx]
        labels = target.flatten()[idx]
    else:
        probs  = pred_probs.flatten()
        labels = target.flatten()

    preds = (probs >= threshold).astype(np.float32)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    precision   = tp / (tp + fp + 1e-7)
    f1          = 2 * tp / (2 * tp + fp + fn + 1e-7)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-7)

    try:
        auc_roc = roc_auc_score(labels, probs)
    except ValueError:
        auc_roc = 0.5

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "f1":          float(f1),
        "accuracy":    float(accuracy),
        "auc_roc":     float(auc_roc),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inferencia con TTA (Test-Time Augmentation)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    n_augments: int = 8,
) -> torch.Tensor:
    """
    TTA mediante promediado de predicciones sobre aumentaciones:
    original + flip horizontal + flip vertical + rotaciones 90/180/270.

    Args:
        model: modelo ya en eval().
        image: tensor [1, 3, H, W] o [3, H, W].
        device: dispositivo de cómputo.
        n_augments: número de aumentaciones (máx 8).

    Returns:
        probabilidades promediadas [1, 1, H, W].
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    augmentations = [
        lambda x: x,                                          # original
        lambda x: TF.hflip(x),                               # flip H
        lambda x: TF.vflip(x),                               # flip V
        lambda x: torch.rot90(x, 1, [-2, -1]),               # rot 90
        lambda x: torch.rot90(x, 2, [-2, -1]),               # rot 180
        lambda x: torch.rot90(x, 3, [-2, -1]),               # rot 270
        lambda x: TF.hflip(torch.rot90(x, 1, [-2, -1])),    # rot90 + flipH
        lambda x: TF.vflip(torch.rot90(x, 1, [-2, -1])),    # rot90 + flipV
    ]

    de_augmentations = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: torch.rot90(x, -1, [-2, -1]),
        lambda x: torch.rot90(x, -2, [-2, -1]),
        lambda x: torch.rot90(x, -3, [-2, -1]),
        lambda x: torch.rot90(TF.hflip(x), -1, [-2, -1]),
        lambda x: torch.rot90(TF.vflip(x), -1, [-2, -1]),
    ]

    model.eval()
    preds = []
    for aug, de_aug in zip(augmentations[:n_augments], de_augmentations[:n_augments]):
        aug_img  = aug(image)
        logits   = model(aug_img)
        prob_aug = torch.sigmoid(logits)
        preds.append(de_aug(prob_aug))

    return torch.stack(preds, dim=0).mean(dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluación sobre un DataLoader
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
    use_tta: bool = False,
    n_augments: int = 8,
) -> Dict[str, float]:
    """
    Evalúa el modelo sobre todo el loader y devuelve métricas agregadas.
    """
    model.eval()
    all_probs   = []
    all_targets = []
    all_fovs    = []

    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].numpy()
        fov   = batch.get("fov")

        if use_tta:
            probs = predict_with_tta(model, imgs, device, n_augments).cpu().numpy()
        else:
            logits = model(imgs)
            probs  = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs.squeeze(1))    # [B, H, W]
        all_targets.append(masks.squeeze(1))  # [B, H, W]
        if fov is not None:
            all_fovs.append(fov.numpy().squeeze(1))

    all_probs   = np.concatenate(all_probs,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    fov_combined = np.concatenate(all_fovs, axis=0) if all_fovs else None

    # Métricas agregadas (todo el dataset como un solo conjunto de píxeles)
    agg = compute_metrics(
        all_probs.reshape(-1),
        all_targets.reshape(-1),
        threshold=threshold,
        fov_mask=fov_combined.reshape(-1) if fov_combined is not None else None,
    )

    # Métricas por imagen (promedio)
    per_image = []
    for i in range(len(all_probs)):
        fov_i = fov_combined[i] if fov_combined is not None else None
        m = compute_metrics(all_probs[i], all_targets[i], threshold, fov_i)
        per_image.append(m)

    for key in ["sensitivity","specificity","f1","precision","auc_roc","accuracy"]:
        agg[f"mean_{key}"] = float(np.mean([m[key] for m in per_image]))
        agg[f"std_{key}"]  = float(np.std( [m[key] for m in per_image]))

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Experimento de domain shift (DRIVE → CHASE_DB1)
# ──────────────────────────────────────────────────────────────────────────────

def domain_shift_experiment(
    model: torch.nn.Module,
    source_loader,
    target_loader,
    device: torch.device,
    threshold: float = 0.5,
    use_tta: bool = True,
    source_name: str = "fuente",
    target_name: str = "objetivo",
) -> Dict[str, Dict]:
    """
    Evalúa el modelo en dominio fuente y dominio objetivo.
    Calcula y reporta la brecha de rendimiento.

    Returns:
        dict con métricas de 'source', 'target' y 'gap'.
    """
    print(f"Evaluando en dominio fuente ({source_name})...")
    source_metrics = evaluate_loader(model, source_loader, device, threshold, use_tta)

    print(f"Evaluando en dominio objetivo ({target_name})...")
    target_metrics = evaluate_loader(model, target_loader, device, threshold, use_tta)

    gap = {}
    for key in ["f1", "sensitivity", "specificity", "auc_roc"]:
        gap[key] = source_metrics[key] - target_metrics[key]

    results = {
        "source": source_metrics,
        "target": target_metrics,
        "gap":    gap,
    }

    print("\n" + "="*60)
    print(f"EXPERIMENTO DE DOMAIN SHIFT ({source_name} → {target_name})")
    print("="*60)
    src_w = max(len(source_name), 8)
    tgt_w = max(len(target_name), 9)
    header = f"{'Métrica':>15} | {source_name:>{src_w}} | {target_name:>{tgt_w}} | {'Brecha':>7}"
    print(header)
    print("-" * (15 + src_w + tgt_w + 16))
    for key in ["f1", "sensitivity", "specificity", "auc_roc"]:
        src = source_metrics[key]
        tgt = target_metrics[key]
        g   = gap[key]
        print(f"{key:>15} | {src:>{src_w}.4f} | {tgt:>{tgt_w}.4f} | {g:>+7.4f}")
    print("="*60)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Análisis de curva ROC y umbral óptimo
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    pred_probs: np.ndarray,
    target: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    criterion: str = "f1",
) -> float:
    """
    Encuentra el umbral óptimo que maximiza el F1 (o la métrica especificada)
    mediante búsqueda en la curva ROC.
    """
    if fov_mask is not None:
        idx    = fov_mask.astype(bool).flatten()
        probs  = pred_probs.flatten()[idx]
        labels = target.flatten()[idx]
    else:
        probs  = pred_probs.flatten()
        labels = target.flatten()

    fpr, tpr, thresholds = roc_curve(labels, probs)

    if criterion == "f1":
        # F1 = 2*TP / (2*TP + FP + FN)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            m = compute_metrics(probs, labels, threshold=t)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_t  = t
        return float(best_t)
    elif criterion == "youden":
        j = tpr - fpr
        return float(thresholds[np.argmax(j)])
    else:
        raise ValueError(f"Criterio '{criterion}' no soportado. Use 'f1' o 'youden'.")

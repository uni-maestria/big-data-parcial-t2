"""
visualize.py — Análisis cualitativo de fallos y visualización de segmentaciones.
Identifica y categoriza errores: capilares finos vs. arterias grandes.
"""

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, List, Dict
from scipy import ndimage


# ──────────────────────────────────────────────────────────────────────────────
# Análisis de componentes conectados (tamaño de vasos)
# ──────────────────────────────────────────────────────────────────────────────

def classify_vessels_by_size(
    mask: np.ndarray,
    thin_threshold: int = 5,    # diámetro en píxeles
    thick_threshold: int = 15,
) -> Dict[str, np.ndarray]:
    """
    Clasifica vasos en capilares finos, vasos medios y arterias/venas grandes
    usando distancia de transformada (aprox. del diámetro del vaso).

    Args:
        mask: máscara binaria uint8/float32 [H, W].
        thin_threshold: radio máximo (píxeles) para considerar vaso fino.
        thick_threshold: radio mínimo para vaso grueso.

    Returns:
        dict con máscaras 'thin', 'medium', 'thick'.
    """
    mask_bin = (mask > 0.5).astype(np.uint8)

    # Distancia de transformada: cada píxel del vaso sabe su distancia al borde
    dist = ndimage.distance_transform_edt(mask_bin)

    thin   = mask_bin & (dist <= thin_threshold)
    thick  = mask_bin & (dist >= thick_threshold)
    medium = mask_bin & (dist > thin_threshold) & (dist < thick_threshold)

    return {"thin": thin, "medium": medium, "thick": thick}


def compute_vessel_type_metrics(
    pred_prob: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict]:
    """
    Calcula F1 por tipo de vaso (fino, medio, grueso).
    """
    vessel_masks = classify_vessels_by_size(target)
    results = {}

    for vtype, vmask in vessel_masks.items():
        if vmask.sum() < 10:
            continue
        pred_in_region = (pred_prob >= threshold).astype(np.float32) * vmask
        target_in_region = target * vmask

        tp = (pred_in_region * target_in_region).sum()
        fp = (pred_in_region * (1 - target_in_region)).sum()
        fn = ((1 - pred_in_region) * target_in_region).sum()

        f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        results[vtype] = {
            "f1":     float(f1),
            "recall": float(recall),
            "n_pixels": int(vmask.sum()),
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Error maps (falsos positivos / negativos por región)
# ──────────────────────────────────────────────────────────────────────────────

def compute_error_map(
    pred_prob: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Genera un mapa de error de 3 clases:
    0 = verdadero negativo (fondo correcto)
    1 = verdadero positivo (vaso correcto)
    2 = falso positivo (predicho vaso, es fondo)
    3 = falso negativo (predicho fondo, es vaso) — más crítico
    """
    pred = (pred_prob >= threshold).astype(np.uint8)
    gt   = (target > 0.5).astype(np.uint8)

    error_map = np.zeros_like(pred, dtype=np.uint8)
    error_map[(pred == 1) & (gt == 1)] = 1   # TP
    error_map[(pred == 1) & (gt == 0)] = 2   # FP
    error_map[(pred == 0) & (gt == 1)] = 3   # FN

    return error_map


# ──────────────────────────────────────────────────────────────────────────────
# Visualización de resultados
# ──────────────────────────────────────────────────────────────────────────────

def plot_segmentation_results(
    images: List[np.ndarray],
    targets: List[np.ndarray],
    preds: List[np.ndarray],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    threshold: float = 0.5,
    max_cols: int = 4,
):
    """
    Genera una grilla de visualizaciones: imagen | GT | predicción | error.
    """
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (img, gt, pred_prob) in enumerate(zip(images, targets, preds)):
        # Desnormalizar imagen para visualización
        img_vis = img.copy()
        if img_vis.max() <= 1.0 and img_vis.min() < 0:
            img_vis = (img_vis * np.array([0.229, 0.224, 0.225]) +
                       np.array([0.485, 0.456, 0.406]))
        img_vis = np.clip(img_vis, 0, 1)

        pred_bin   = (pred_prob >= threshold).astype(np.float32)
        error_map  = compute_error_map(pred_prob, gt, threshold)

        # Colorear el mapa de error
        error_rgb = np.zeros((*error_map.shape, 3), dtype=np.float32)
        error_rgb[error_map == 1] = [0.2, 0.8, 0.2]   # TP: verde
        error_rgb[error_map == 2] = [0.9, 0.6, 0.1]   # FP: naranja
        error_rgb[error_map == 3] = [0.9, 0.1, 0.1]   # FN: rojo

        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title("Imagen original", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground truth", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_prob, cmap="hot")
        axes[i, 2].set_title("Predicción (prob.)", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(error_rgb)
        axes[i, 3].set_title("Error: TP=verde FP=naranja FN=rojo", fontsize=9)
        axes[i, 3].axis("off")

        if titles:
            axes[i, 0].set_title(f"{titles[i]}\n{axes[i,0].get_title()}", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualización guardada en: {save_path}")
    plt.close()


def plot_vessel_type_analysis(
    pred_prob: np.ndarray,
    target: np.ndarray,
    save_path: Optional[str] = None,
    threshold: float = 0.5,
):
    """
    Visualiza la segmentación por tipo de vaso y las métricas por categoría.
    """
    vessel_masks = classify_vessels_by_size(target)
    metrics      = compute_vessel_type_metrics(pred_prob, target, threshold)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Fila 1: Tipos de vasos en GT
    colors = {"thin": [1, 0.5, 0], "medium": [0.2, 0.8, 0.2], "thick": [0.1, 0.3, 0.9]}
    for j, (vtype, vmask) in enumerate(vessel_masks.items()):
        rgb = np.zeros((*vmask.shape, 3))
        rgb[vmask > 0] = colors[vtype]
        axes[0, j].imshow(rgb)
        n_px = vmask.sum()
        axes[0, j].set_title(f"Vasos {vtype}\n({n_px} px)", fontsize=11)
        axes[0, j].axis("off")

    # Fila 2: Error por tipo de vaso
    pred_bin = (pred_prob >= threshold).astype(np.float32)
    for j, (vtype, vmask) in enumerate(vessel_masks.items()):
        error = compute_error_map(pred_prob, target * vmask, threshold)
        rgb = np.zeros((*error.shape, 3))
        rgb[error == 1] = [0.2, 0.8, 0.2]
        rgb[error == 2] = [0.9, 0.6, 0.1]
        rgb[error == 3] = [0.9, 0.1, 0.1]
        m = metrics.get(vtype, {})
        f1  = m.get("f1", 0)
        rec = m.get("recall", 0)
        axes[1, j].imshow(rgb)
        axes[1, j].set_title(f"Errores {vtype}\nF1={f1:.3f}  Recall={rec:.3f}", fontsize=11)
        axes[1, j].axis("off")

    plt.suptitle("Análisis de segmentación por tipo de vaso", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Análisis guardado en: {save_path}")
    plt.close()

    return metrics


def plot_domain_shift_comparison(
    source_metrics: Dict, target_metrics: Dict,
    save_path: Optional[str] = None,
):
    """Gráfica de barras comparando métricas DRIVE vs CHASE_DB1."""
    keys   = ["f1", "sensitivity", "specificity", "auc_roc"]
    labels = ["F1", "Sensibilidad", "Especificidad", "AUC-ROC"]
    src_v  = [source_metrics[k] for k in keys]
    tgt_v  = [target_metrics[k] for k in keys]

    x    = np.arange(len(keys))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, src_v, w, label="DRIVE (fuente)", color="#2563eb", alpha=0.85)
    b2 = ax.bar(x + w/2, tgt_v, w, label="CHASE_DB1 (objetivo)", color="#dc2626", alpha=0.85)

    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_title("Brecha de rendimiento entre dominios", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Anotar brecha
    for i, (sv, tv) in enumerate(zip(src_v, tgt_v)):
        gap = sv - tv
        ax.annotate(f"Δ={gap:+.3f}", xy=(x[i], min(sv, tv) - 0.005),
                    ha="center", fontsize=9, color="#7c3aed")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfica guardada en: {save_path}")
    plt.close()


def plot_roc_curve(
    pred_probs: np.ndarray,
    targets: np.ndarray,
    labels: List[str] = None,
    save_path: Optional[str] = None,
):
    """Curva ROC para uno o varios modelos."""
    from sklearn.metrics import roc_curve, auc

    if not isinstance(pred_probs, list):
        pred_probs = [pred_probs]
        targets    = [targets]
        labels     = labels or ["Modelo"]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]

    for prob, tgt, lbl, c in zip(pred_probs, targets, labels, colors):
        fpr, tpr, _ = roc_curve(tgt.flatten(), prob.flatten())
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{lbl} (AUC={roc_auc:.4f})")

    ax.plot([0,1],[0,1], "k--", alpha=0.4)
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=12)
    ax.set_title("Curva ROC — Segmentación de vasos retinianos", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

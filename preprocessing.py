"""
preprocessing.py — Preprocesamiento de imágenes de fundus retiniano.
Implementa CLAHE, extracción de canal verde, normalización y ecualización de histograma.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
    apply_to: str = "green",  # "green" | "lab" | "all"
) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) a la imagen.

    Args:
        image_rgb: imagen en formato RGB uint8 [H, W, 3].
        clip_limit: límite de contraste para CLAHE.
        tile_grid: tamaño de la cuadrícula de teselas.
        apply_to: "green" aplica solo al canal verde;
                  "lab"   aplica al canal L del espacio LAB;
                  "all"   aplica a los tres canales por separado.

    Returns:
        Imagen RGB con CLAHE aplicado.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out   = image_rgb.copy()

    if apply_to == "green":
        # Solo canal verde (el más informativo para vasos retinianos)
        g_eq = clahe.apply(out[:, :, 1])
        out  = np.stack([out[:, :, 0], g_eq, out[:, :, 2]], axis=-1)

    elif apply_to == "lab":
        # Espacio LAB: aplica CLAHE al canal de luminancia
        lab       = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        out        = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif apply_to == "all":
        # Aplica a cada canal independientemente
        for c in range(3):
            out[:, :, c] = clahe.apply(out[:, :, c])

    return out


def green_channel_extract(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extrae el canal verde y lo replica en los 3 canales (imagen "verde" RGB).
    El canal verde presenta el mayor contraste entre vasos y fondo.
    """
    g = image_rgb[:, :, 1]
    return np.stack([g, g, g], axis=-1)


def normalize_fundus(
    image_rgb: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Normaliza la imagen de fundus con media/std de ImageNet (apropiado si se usa
    backbone preentrenado) o los estadísticos del canal.

    Returns:
        imagen float32 normalizada en el rango aprox. [-2, 2].
    """
    img = image_rgb.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    img = (img - mean_arr) / (std_arr + 1e-7)
    return img


def histogram_equalization(image_rgb: np.ndarray) -> np.ndarray:
    """
    Ecualización de histograma global en el canal verde (alternativa más simple a CLAHE).
    """
    out      = image_rgb.copy()
    out[:,:,1] = cv2.equalizeHist(out[:,:,1])
    return out


def extract_fov_mask(image_rgb: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Extrae automáticamente la máscara del campo visual (FOV) por umbralización.
    Los bordes negros de las imágenes de fundus no contienen información útil.

    Returns:
        máscara binaria uint8 [H, W] con 255 en FOV y 0 fuera.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Cerrado morfológico para eliminar huecos pequeños
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def gamma_correction(image_rgb: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Corrección gamma para aumentar el contraste en regiones oscuras."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image_rgb, table)


def preprocess_pipeline(
    image_rgb: np.ndarray,
    use_clahe: bool = True,
    use_gamma: bool = False,
    gamma: float = 1.2,
    clahe_mode: str = "green",
) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento.
    Orden: CLAHE → corrección gamma → normalización.
    """
    if use_clahe:
        image_rgb = apply_clahe(image_rgb, apply_to=clahe_mode)
    if use_gamma:
        image_rgb = gamma_correction(image_rgb, gamma=gamma)
    return normalize_fundus(image_rgb)

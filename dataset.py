"""
dataset.py — DataLoaders para DRIVE, STARE y CHASE_DB1.
Soporta aumentación de datos, CLAHE y normalización por dataset.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from preprocessing import apply_clahe, green_channel_extract, normalize_fundus


# ──────────────────────────────────────────────────────────────────────────────
# Augmentaciones conjuntas imagen + máscara
# ──────────────────────────────────────────────────────────────────────────────

class JointAugment:
    """
    Aplica las mismas transformaciones geométricas a imagen y máscara.
    Las transformaciones de color solo se aplican a la imagen.
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        max_angle: float = 30.0,
        p_crop: float = 0.3,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        p_brightness: float = 0.4,
        p_gamma: float = 0.3,
        output_size: int = 512,
    ):
        self.p_flip       = p_flip
        self.p_rotate     = p_rotate
        self.max_angle    = max_angle
        self.p_crop       = p_crop
        self.crop_scale   = crop_scale
        self.p_brightness = p_brightness
        self.p_gamma      = p_gamma
        self.output_size  = output_size

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Volteo horizontal
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Volteo vertical
        if random.random() < self.p_flip:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Rotación
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # Recorte aleatorio
        if random.random() < self.p_crop:
            scale = random.uniform(*self.crop_scale)
            h, w  = image.shape[-2], image.shape[-1]
            ch, cw = int(h * scale), int(w * scale)
            i = random.randint(0, h - ch)
            j = random.randint(0, w - cw)
            image = TF.resized_crop(image, i, j, ch, cw, self.output_size,
                                     interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resized_crop(mask,  i, j, ch, cw, self.output_size,
                                     interpolation=TF.InterpolationMode.NEAREST)
        else:
            image = TF.resize(image, [self.output_size, self.output_size],
                               interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  [self.output_size, self.output_size],
                               interpolation=TF.InterpolationMode.NEAREST)

        # Brillo/contraste (solo imagen)
        if random.random() < self.p_brightness:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            image = TF.adjust_contrast(image,   random.uniform(0.7, 1.3))

        # Corrección gamma (solo imagen)
        if random.random() < self.p_gamma:
            gamma = random.uniform(0.7, 1.5)
            image = TF.adjust_gamma(image, gamma)

        return image, mask


# ──────────────────────────────────────────────────────────────────────────────
# Dataset base
# ──────────────────────────────────────────────────────────────────────────────

class RetinalDataset(Dataset):
    """
    Dataset genérico para imágenes de fundus con máscaras de vasos.

    Args:
        image_paths: lista de rutas a imágenes.
        mask_paths: lista de rutas a máscaras binarias.
        augment: objeto JointAugment o None.
        use_clahe: si True aplica CLAHE al canal verde antes de convertir a RGB.
        output_size: tamaño final de la imagen.
        use_fov_mask: si True aplica máscara de FOV (DRIVE).
        fov_paths: rutas a las máscaras de FOV.
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        augment: Optional[JointAugment] = None,
        use_clahe: bool = True,
        output_size: int = 512,
        use_fov_mask: bool = False,
        fov_paths: Optional[List[Path]] = None,
    ):
        assert len(image_paths) == len(mask_paths), \
            f"Número de imágenes y máscaras no coincide, {len(image_paths)} vs {len(mask_paths)}"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment
        self.use_clahe   = use_clahe
        self.output_size = output_size
        self.use_fov_mask = use_fov_mask
        self.fov_paths   = fov_paths
        self.to_tensor   = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        # Cargar imagen
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocesamiento
        if self.use_clahe:
            img = apply_clahe(img)          # CLAHE en canal verde
        img = normalize_fundus(img)         # normalización por canal

        # Cargar máscara
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        # Convertir a tensor
        img_t  = self.to_tensor(img)        # [3, H, W] float32 en [0,1]
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        # Augmentación
        if self.augment is not None:
            img_t, mask_t = self.augment(img_t, mask_t)
        else:
            img_t  = TF.resize(img_t,  [self.output_size, self.output_size],
                                interpolation=TF.InterpolationMode.BILINEAR)
            mask_t = TF.resize(mask_t, [self.output_size, self.output_size],
                                interpolation=TF.InterpolationMode.NEAREST)

        result = {"image": img_t, "mask": mask_t.float()}

        # Máscara de FOV (región de interés)
        if self.use_fov_mask and self.fov_paths is not None:
            fov = cv2.imread(str(self.fov_paths[idx]), cv2.IMREAD_GRAYSCALE)
            fov = (fov > 127).astype(np.float32)
            fov_t = torch.from_numpy(fov).unsqueeze(0)
            fov_t = TF.resize(fov_t, [self.output_size, self.output_size],
                               interpolation=TF.InterpolationMode.NEAREST)
            result["fov"] = fov_t.float()

        result["path"] = str(self.image_paths[idx])
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de carga por dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_drive(
    root: str,
    split: str = "train",
    augment: bool = True,
    use_clahe: bool = True,
    output_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """
    Carga el dataset DRIVE con su división oficial train/test.

    Estructura esperada:
        root/
          training/
            images/   *.tif
            1st_manual/ *.gif
            mask/     *.gif
          test/
            images/   *.tif
            1st_manual/ *.gif
            mask/     *.gif
    """
    root = Path(root)
    sub  = "training" if split == "train" else "test"

    images = sorted((root / sub / "images").glob("*.tif"))
    masks  = sorted((root / sub / "1st_manual").glob("*.gif"))
    fovs   = sorted((root / sub / "mask").glob("*.gif"))

    assert len(images) > 0, f"No se encontraron imágenes en {root/sub/'images'}"

    aug = JointAugment(output_size=output_size) if augment and split == "train" else None

    ds = RetinalDataset(
        image_paths=images,
        mask_paths=masks,
        augment=aug,
        use_clahe=use_clahe,
        output_size=output_size,
        use_fov_mask=True,
        fov_paths=fovs,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, pin_memory=True, drop_last=(split == "train"))


def load_stare(
    root: str,
    augment: bool = False,
    use_clahe: bool = True,
    output_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    annotator: int = 1,  # 1 o 2 (dos anotadores expertos)
) -> DataLoader:
    """
    Carga STARE (20 imágenes .ppm con anotaciones de 2 expertos).

    Estructura esperada:
        root/
          images/ *.ppm
          labels-ah/ *.ppm   (anotador 1)
          labels-vk/ *.ppm   (anotador 2)
    """
    root   = Path(root)
    images = sorted((root / "images").glob("*.ppm"))
    label_dir = "labels-ah" if annotator == 1 else "labels-vk"
    masks  = sorted((root / label_dir).glob("*.ppm"))

    aug = JointAugment(output_size=output_size) if augment else None
    ds  = RetinalDataset(images, masks, augment=aug, use_clahe=use_clahe,
                          output_size=output_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def load_chase_db1(
    root: str,
    augment: bool = False,
    use_clahe: bool = True,
    output_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """
    Carga CHASE_DB1 (28 imágenes .jpg con anotaciones _1stHO.png).

    Estructura esperada:
        root/
          Image_*.jpg
          Image_*_1stHO.png
    """
    root   = Path(root)
    images = sorted(root.glob("Image_*.jpg"))
    masks  = sorted(root.glob("Image_*_1stHO.png"))

    aug = JointAugment(output_size=output_size) if augment else None
    ds  = RetinalDataset(images, masks, augment=aug, use_clahe=use_clahe,
                          output_size=output_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(loader: DataLoader) -> torch.Tensor:
    """
    Calcula el peso positivo para BCE con logits (pos_weight = n_neg / n_pos).
    Útil para manejar el desbalance entre vasos y fondo.
    """
    n_pos, n_neg = 0, 0
    for batch in loader:
        m = batch["mask"]
        n_pos += m.sum().item()
        n_neg += (1 - m).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    print(f"  n_pos={n_pos:.0f}, n_neg={n_neg:.0f}, pos_weight={pos_weight.item():.2f}")
    return pos_weight

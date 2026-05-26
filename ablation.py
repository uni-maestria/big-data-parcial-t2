"""
ablation.py — Estudio de ablación sistemático.
Compara: profundidad, función de pérdida, CLAHE, arquitecturas.
"""

import json
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
import pandas as pd

from config import Config
from train import train
from evaluate import evaluate_loader, domain_shift_experiment
from dataset import load_drive, load_chase_db1
from model import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Definición de experimentos
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_GRID = {
    # Experimento 1: Función de pérdida
    "loss_function": [
        {"loss": "bce",      "arch": "unet", "base_filters": 64},
        {"loss": "dice",     "arch": "unet", "base_filters": 64},
        {"loss": "combined", "arch": "unet", "base_filters": 64},
        {"loss": "focal",    "arch": "unet", "base_filters": 64},
    ],
    # Experimento 2: Arquitectura
    "architecture": [
        {"loss": "combined", "arch": "unet",     "base_filters": 64},
        {"loss": "combined", "arch": "att_unet", "base_filters": 64},
        {"loss": "combined", "arch": "unetpp",   "base_filters": 32},
    ],
    # Experimento 3: Profundidad (base_filters determina implícitamente la capacidad)
    "capacity": [
        {"loss": "combined", "arch": "unet", "base_filters": 32},
        {"loss": "combined", "arch": "unet", "base_filters": 64},
        {"loss": "combined", "arch": "unet", "base_filters": 128},
    ],
    # Experimento 4: Preprocesamiento
    "preprocessing": [
        {"loss": "combined", "arch": "unet", "base_filters": 64, "use_clahe": False},
        {"loss": "combined", "arch": "unet", "base_filters": 64, "use_clahe": True},
    ],
}


@dataclass
class AblationResult:
    experiment_name: str
    variant: str
    config: Dict[str, Any]
    # Métricas en DRIVE (in-distribution)
    drive_f1:          float = 0.0
    drive_sensitivity: float = 0.0
    drive_specificity: float = 0.0
    drive_auc:         float = 0.0
    # Métricas en CHASE_DB1 (out-of-distribution)
    chase_f1:          float = 0.0
    chase_sensitivity: float = 0.0
    chase_auc:         float = 0.0
    # Brecha
    f1_gap: float = 0.0


def run_ablation(
    drive_root: str,
    chase_root: str,
    experiments: List[str] = None,
    epochs: int = 50,
    img_size: int = 512,
    save_dir: str = "ablation_results",
    device_str: str = "cuda",
):
    """
    Ejecuta el estudio de ablación completo.

    Args:
        experiments: lista de nombres de experimentos a ejecutar.
                     Si None, ejecuta todos.
        epochs: épocas de entrenamiento (reducidas para ablación).
    """
    device    = torch.device(device_str if torch.cuda.is_available() else "cpu")
    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results   = []

    if experiments is None:
        experiments = list(ABLATION_GRID.keys())

    for exp_name in experiments:
        variants = ABLATION_GRID[exp_name]
        print(f"\n{'='*60}")
        print(f"ABLACIÓN: {exp_name}")
        print(f"{'='*60}")

        for v_cfg in variants:
            variant_name = "_".join(f"{k}={v}" for k, v in v_cfg.items())
            print(f"\n  Variante: {variant_name}")

            # Configuración base + override por variante
            cfg = Config(
                arch=v_cfg.get("arch", "unet"),
                loss=v_cfg.get("loss", "combined"),
                base_filters=v_cfg.get("base_filters", 64),
                use_clahe=v_cfg.get("use_clahe", True),
                epochs=epochs,
                drive_root=drive_root,
                save_dir=str(save_dir / exp_name),
                img_size=img_size,
                patience=10,
            )

            # Entrenar
            ckpt_path = train(cfg)

            # Cargar mejor modelo
            model = build_model(
                arch=cfg.arch, in_channels=3, out_channels=1,
                base_filters=cfg.base_filters
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            model.eval()

            # Evaluar en DRIVE (test set)
            drive_loader = load_drive(
                drive_root, split="test", augment=False,
                use_clahe=cfg.use_clahe, output_size=img_size,
                batch_size=1, num_workers=2,
            )
            drive_m = evaluate_loader(model, drive_loader, device, use_tta=True)

            # Evaluar en CHASE_DB1
            chase_loader = load_chase_db1(
                chase_root, augment=False,
                use_clahe=cfg.use_clahe, output_size=img_size,
                batch_size=1, num_workers=2,
            )
            chase_m = evaluate_loader(model, chase_loader, device, use_tta=True)

            res = AblationResult(
                experiment_name=exp_name,
                variant=variant_name,
                config=v_cfg,
                drive_f1=drive_m["f1"],
                drive_sensitivity=drive_m["sensitivity"],
                drive_specificity=drive_m["specificity"],
                drive_auc=drive_m["auc_roc"],
                chase_f1=chase_m["f1"],
                chase_sensitivity=chase_m["sensitivity"],
                chase_auc=chase_m["auc_roc"],
                f1_gap=drive_m["f1"] - chase_m["f1"],
            )
            results.append(res)

            print(f"    DRIVE  — F1={res.drive_f1:.4f}  AUC={res.drive_auc:.4f}")
            print(f"    CHASE  — F1={res.chase_f1:.4f}  AUC={res.chase_auc:.4f}")
            print(f"    Brecha — ΔF1={res.f1_gap:+.4f}")

    # Guardar resultados
    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = save_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")

    # Imprimir tabla resumen
    print_ablation_table(df)
    return df


def print_ablation_table(df: pd.DataFrame):
    """Imprime una tabla resumen legible del estudio de ablación."""
    print("\n" + "="*80)
    print("TABLA RESUMEN DEL ESTUDIO DE ABLACIÓN")
    print("="*80)
    cols = ["experiment_name", "variant", "drive_f1", "drive_auc",
            "chase_f1", "chase_auc", "f1_gap"]
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print("="*80)


# ──────────────────────────────────────────────────────────────────────────────
# Análisis de adaptación de dominio
# ──────────────────────────────────────────────────────────────────────────────

def domain_adaptation_experiment(
    model: torch.nn.Module,
    drive_root: str,
    chase_root: str,
    device: torch.device,
    fine_tune_n: int = 5,
    img_size: int = 512,
):
    """
    Compara tres estrategias de adaptación de dominio:
    1. Sin adaptación (baseline)
    2. CLAHE en el dominio objetivo
    3. Fine-tuning con n imágenes del dominio objetivo

    Args:
        fine_tune_n: número de imágenes de CHASE_DB1 para fine-tuning.
    """
    from dataset import load_chase_db1, RetinalDataset
    from torch.utils.data import DataLoader
    from pathlib import Path

    results = {}

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n[1/3] Baseline (sin adaptación)...")
    chase_loader = load_chase_db1(chase_root, use_clahe=False, output_size=img_size,
                                   batch_size=1, num_workers=2)
    results["baseline"] = evaluate_loader(model, chase_loader, device, use_tta=False)

    # ── CLAHE ────────────────────────────────────────────────────────────────
    print("[2/3] Con CLAHE (adaptación de preprocesamiento)...")
    chase_clahe = load_chase_db1(chase_root, use_clahe=True, output_size=img_size,
                                  batch_size=1, num_workers=2)
    results["clahe"] = evaluate_loader(model, chase_clahe, device, use_tta=True)

    # ── TTA ──────────────────────────────────────────────────────────────────
    print("[3/3] Con TTA (8 aumentaciones en prueba)...")
    results["tta"] = evaluate_loader(model, chase_clahe, device, use_tta=True, n_augments=8)

    # Reportar
    print("\n" + "="*55)
    print("COMPARACIÓN DE ESTRATEGIAS DE ADAPTACIÓN DE DOMINIO")
    print("="*55)
    print(f"{'Estrategia':>20} | {'F1':>7} | {'Sens':>7} | {'AUC':>7}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:>20} | {m['f1']:>7.4f} | {m['sensitivity']:>7.4f} | {m['auc_roc']:>7.4f}")
    print("="*55)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive_root",  type=str, required=True)
    parser.add_argument("--chase_root",  type=str, required=True)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--save_dir",    type=str, default="ablation_results")
    args = parser.parse_args()

    run_ablation(
        drive_root=args.drive_root,
        chase_root=args.chase_root,
        experiments=args.experiments,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )

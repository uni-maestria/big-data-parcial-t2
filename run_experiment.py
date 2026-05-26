"""
run_experiment.py — Script principal que ejecuta todos los entregables del examen.
Uso: python run_experiment.py --drive_root data/DRIVE --chase_root data/CHASE_DB1
"""

import argparse
import json
from pathlib import Path

import torch

from config import Config
from model import build_model
from dataset import load_drive, load_chase_db1, load_stare
from train import train
from evaluate import evaluate_loader, domain_shift_experiment, find_optimal_threshold
from ablation import run_ablation, domain_adaptation_experiment
from visualize import (
    plot_segmentation_results,
    plot_vessel_type_analysis,
    plot_domain_shift_comparison,
    plot_roc_curve,
    compute_vessel_type_metrics,
)
import numpy as np


def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        drive_root=args.drive_root,
        chase_root=args.chase_root,
        stare_root=args.stare_root,
        arch="att_unet",
        loss="combined",
        base_filters=64,
        use_clahe=True,
        epochs=args.epochs,
        img_size=512,
        use_tta=True,
    )
    cfg.describe()

    # ──────────────────────────────────────────────────────────────────────────
    # ENTREGABLE 1: Entrenamiento del modelo principal
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[ENTREGABLE 1] Entrenamiento de Attention U-Net...")
    ckpt_path = train(cfg)

    model = build_model("att_unet", in_channels=3, out_channels=1,
                         base_filters=64).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ──────────────────────────────────────────────────────────────────────────
    # ENTREGABLE 3: Evaluación in-distribution (DRIVE)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[ENTREGABLE 3] Evaluación en DRIVE (test set)...")
    drive_test = load_drive(args.drive_root, split="test", augment=False,
                             use_clahe=True, output_size=512, batch_size=1)
    drive_metrics = evaluate_loader(model, drive_test, device, use_tta=True)

    print("\nMÉTRICAS EN DRIVE (in-distribution):")
    for k in ["sensitivity", "specificity", "f1", "auc_roc", "accuracy"]:
        print(f"  {k:>15}: {drive_metrics[k]:.4f}  (±{drive_metrics.get('std_'+k, 0):.4f})")

    with open(results_dir / "drive_metrics.json", "w") as f:
        json.dump(drive_metrics, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # ENTREGABLE 4: Experimento de domain shift (DRIVE → CHASE_DB1)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[ENTREGABLE 4] Experimento de domain shift...")
    chase_loader = load_chase_db1(args.chase_root, use_clahe=True,
                                   output_size=512, batch_size=1)
    shift_results = domain_shift_experiment(model, drive_test, chase_loader, device,
                                             use_tta=True)

    plot_domain_shift_comparison(
        shift_results["source"], shift_results["target"],
        save_path=str(results_dir / "domain_shift.png"),
    )
    with open(results_dir / "domain_shift.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, float)}
                   for k, v in shift_results.items()}, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # ENTREGABLE 5: Análisis cualitativo de fallos por tipo de vaso
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[ENTREGABLE 5] Análisis de fallos por tipo de vaso...")
    all_vessel_metrics = {"thin": [], "medium": [], "thick": []}

    with torch.no_grad():
        for batch in drive_test:
            img   = batch["image"].to(device)
            gt    = batch["mask"].numpy().squeeze()
            logit = model(img)
            prob  = torch.sigmoid(logit).cpu().numpy().squeeze()

            vm = compute_vessel_type_metrics(prob, gt, threshold=0.5)
            for vtype, m in vm.items():
                all_vessel_metrics[vtype].append(m["f1"])

    print("\nF1 POR TIPO DE VASO (DRIVE):")
    print(f"  {'Tipo':>12} | {'F1 medio':>10} | {'F1 std':>8}")
    print("  " + "-"*38)
    for vtype, f1s in all_vessel_metrics.items():
        if f1s:
            print(f"  {vtype:>12} | {np.mean(f1s):>10.4f} | {np.std(f1s):>8.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # ENTREGABLE 6: Estrategia de adaptación de dominio
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[ENTREGABLE 6] Comparación de estrategias de adaptación...")
    adapt_results = domain_adaptation_experiment(
        model, args.drive_root, args.chase_root, device, img_size=512,
    )
    with open(results_dir / "adaptation_results.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, float)}
                   for k, v in adapt_results.items()}, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # ENTREGABLE 2: Estudio de ablación (versión rápida)
    # ──────────────────────────────────────────────────────────────────────────
    if args.run_ablation:
        print("\n[ENTREGABLE 2] Estudio de ablación...")
        ablation_df = run_ablation(
            drive_root=args.drive_root,
            chase_root=args.chase_root,
            experiments=["loss_function", "architecture"],
            epochs=min(args.epochs, 30),
            save_dir=str(results_dir / "ablation"),
        )
        ablation_df.to_csv(results_dir / "ablation_results.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Experimento completo. Resultados en: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examen Parcial — Segmentación de vasos retinianos")
    parser.add_argument("--drive_root",   type=str, required=True,  help="Ruta a DRIVE/")
    parser.add_argument("--chase_root",   type=str, required=True,  help="Ruta a CHASE_DB1/")
    parser.add_argument("--stare_root",   type=str, default="",     help="Ruta a STARE/ (opcional)")
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--results_dir",  type=str, default="results")
    parser.add_argument("--run_ablation", action="store_true",
                        help="Ejecutar también el estudio de ablación (más lento)")
    args = parser.parse_args()
    main(args)

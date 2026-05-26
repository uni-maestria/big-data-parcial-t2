"""
train.py — Loop de entrenamiento para segmentación de vasos retinianos.
Soporta: early stopping, LR scheduler, logging, checkpointing.
"""

import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from config import Config
from model import build_model
from dataset import load_drive, compute_class_weights
from losses import CombinedLoss, build_loss
from evaluate import evaluate_loader


# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento por época
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    total_loss = 0.0

    for batch in loader:
        imgs   = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)
        fov    = batch.get("fov")
        if fov is not None:
            fov = fov.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            preds = model(imgs)
            # Soporte para criterios que aceptan fov_mask
            try:
                loss = criterion(preds, masks, fov)
            except TypeError:
                loss = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Devuelve True si el modelo mejoró."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Datos ────────────────────────────────────────────────────────────────
    train_loader = load_drive(
        cfg.drive_root, split="train", augment=True,
        use_clahe=cfg.use_clahe, output_size=cfg.img_size,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )
    val_loader = load_drive(
        cfg.drive_root, split="test", augment=False,
        use_clahe=cfg.use_clahe, output_size=cfg.img_size,
        batch_size=1, num_workers=cfg.num_workers,
    )

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = build_model(
        arch=cfg.arch,
        in_channels=3,
        out_channels=1,
        base_filters=cfg.base_filters,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {cfg.arch} | Parámetros: {n_params/1e6:.2f}M")

    # ── Pérdida ──────────────────────────────────────────────────────────────
    if cfg.loss == "combined":
        # Calculamos pos_weight para BCE balanceada
        print("Calculando pesos de clase...")
        pos_weight = compute_class_weights(train_loader)
        criterion  = CombinedLoss(alpha=cfg.loss_alpha, pos_weight=pos_weight)
    else:
        criterion = build_loss(cfg.loss)

    # ── Optimizador y scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-3
    )
    scaler    = GradScaler(enabled=cfg.amp)
    stopper   = EarlyStopping(patience=cfg.patience)

    # ── Directorio de salida ──────────────────────────────────────────────────
    save_dir = Path(cfg.save_dir) / cfg.arch / cfg.loss
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pth"

    # ── Log ──────────────────────────────────────────────────────────────────
    log_path = save_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_f1,val_auc,lr\n")

    print(f"\n{'Época':>6} | {'Loss':>8} | {'F1':>7} | {'AUC':>7} | {'LR':>9}")
    print("-" * 50)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Entrenamiento
        train_loss = train_one_epoch(model, train_loader, criterion,
                                      optimizer, scaler, device, cfg)

        # Validación
        metrics = evaluate_loader(model, val_loader, device, threshold=cfg.threshold)
        f1  = metrics["f1"]
        auc = metrics["auc_roc"]
        lr  = optimizer.param_groups[0]["lr"]

        scheduler.step()

        # Log
        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>8.4f} | {f1:>7.4f} | {auc:>7.4f} | {lr:>9.2e}  [{elapsed:.0f}s]")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.5f},{f1:.5f},{auc:.5f},{lr:.2e}\n")

        # Checkpoint si mejoró
        improved = stopper.step(f1)
        if improved:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "config":     vars(cfg),
                "val_f1":     f1,
                "val_auc":    auc,
            }, best_path)
            print(f"  ✓ Mejor modelo guardado (F1={f1:.4f})")

        if stopper.should_stop:
            print(f"\nEarly stopping en época {epoch} (patience={cfg.patience})")
            break

    print(f"\nEntrenamiento completo. Mejor F1: {stopper.best_score:.4f}")
    print(f"Checkpoint guardado en: {best_path}")
    return best_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar U-Net para vasos retinianos")
    parser.add_argument("--arch",        type=str,   default="unet",      choices=["unet","att_unet","unetpp"])
    parser.add_argument("--loss",        type=str,   default="combined",  choices=["bce","dice","combined","focal","tversky"])
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--base_filters",type=int,   default=64)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--img_size",    type=int,   default=512)
    parser.add_argument("--drive_root",  type=str,   default="data/DRIVE")
    parser.add_argument("--use_clahe",   action="store_true", default=True)
    parser.add_argument("--no_clahe",    dest="use_clahe", action="store_false")
    parser.add_argument("--save_dir",    type=str,   default="checkpoints")
    args = parser.parse_args()

    cfg = Config(**vars(args))
    train(cfg)

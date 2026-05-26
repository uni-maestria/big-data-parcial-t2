# Segmentación de Vasos Retinianos para Detección de Retinopatía Diabética

**Examen Parcial — Redes Neuronales y Aprendizaje Profundo**
Ph.D. Aldo Camargo

## Descripción

Implementación desde cero en PyTorch de U-Net y variantes (Attention U-Net, U-Net++) para segmentación binaria píxel a píxel de vasos sanguíneos en imágenes de fondo de ojo (fundus). El proyecto incluye un estudio de ablación, análisis de domain shift y estrategias de adaptación de dominio.

## Estructura del proyecto

```
retinal-unet/
├── model.py          # U-Net, Attention U-Net, U-Net++ (implementación desde cero)
├── dataset.py        # DataLoaders para DRIVE, STARE y CHASE_DB1
├── preprocessing.py  # CLAHE, normalización, extracción de FOV
├── losses.py         # BCE, Dice, Combinada, Focal, Tversky
├── train.py          # Loop de entrenamiento con early stopping
├── evaluate.py       # Métricas, TTA y experimento de domain shift
├── ablation.py       # Estudio de ablación sistemático
├── visualize.py      # Análisis cualitativo de fallos
├── config.py         # Hiperparámetros centralizados
├── run_experiment.py # Script principal (todos los entregables)
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Datos

Descargar y colocar en la carpeta `data/`:

| Dataset   | Imágenes | Uso               | URL                                              |
|-----------|----------|-------------------|--------------------------------------------------|
| DRIVE     | 40       | Train + evaluación | https://drive.grand-challenge.org/               |
| STARE     | 20       | Concordancia entre anotadores | https://cecas.clemson.edu/~ahoover/stare/ |
| CHASE_DB1 | 28       | Generalización entre datasets | https://blogs.kingston.ac.uk/retinal/chasedb1/ |

### Estructura esperada DRIVE

```
data/DRIVE/
  training/
    images/       *.tif  (20 imágenes)
    1st_manual/   *.gif  (20 máscaras manuales)
    mask/         *.gif  (20 máscaras de FOV)
  test/
    images/       *.tif  (20 imágenes)
    1st_manual/   *.gif
    mask/         *.gif
```

## Uso rápido

### Entrenamiento

```bash
python train.py \
  --arch att_unet \
  --loss combined \
  --epochs 100 \
  --batch_size 4 \
  --drive_root data/DRIVE \
  --use_clahe
```

### Experimento completo (todos los entregables)

```bash
python run_experiment.py \
  --drive_root data/DRIVE \
  --chase_root data/CHASE_DB1 \
  --epochs 100 \
  --results_dir results/ \
  --run_ablation
```

### Solo estudio de ablación

```bash
python ablation.py \
  --drive_root data/DRIVE \
  --chase_root data/CHASE_DB1 \
  --experiments loss_function architecture \
  --epochs 50
```

## Entregables implementados

### 1. Implementación de U-Net (`model.py`)
- **U-Net estándar**: profundidad configurable, BN, dropout
- **Attention U-Net**: puertas de atención en cada skip connection
- **U-Net++**: conexiones de salto densas anidadas con deep supervision

### 2. Estudio de ablación (`ablation.py`)
Compara:
- Función de pérdida: BCE vs Dice vs Combinada vs Focal
- Arquitectura: U-Net vs Attention U-Net vs U-Net++
- Capacidad del modelo (base_filters: 32/64/128)
- Preprocesamiento: con/sin CLAHE

### 3. Evaluación in-distribution DRIVE (`evaluate.py`)
Métricas: Sensibilidad, Especificidad, F1, AUC-ROC, Accuracy (con FOV mask)

### 4. Experimento de domain shift (`evaluate.py`)
Entrena en DRIVE, evalúa en CHASE_DB1. Reporta la brecha de rendimiento.

### 5. Análisis cualitativo de fallos (`visualize.py`)
Clasifica vasos por diámetro (fino/medio/grueso) y calcula F1 por categoría.

### 6. Estrategia de adaptación de dominio (`ablation.py`)
Compara: baseline sin adaptación vs CLAHE vs Test-Time Augmentation (8 aumentaciones).

## Resultados esperados (referencia)

| Modelo        | DRIVE F1 | CHASE F1 | Brecha |
|---------------|----------|----------|--------|
| U-Net         | ~0.820   | ~0.780   | ~0.040 |
| Attention U-Net | ~0.828 | ~0.791   | ~0.037 |
| U-Net++ (DS)  | ~0.831   | ~0.793   | ~0.038 |

*Resultados con CLAHE + TTA. Varían según semilla y hardware.*

## Decisiones de diseño clave

### Función de pérdida
La pérdida combinada (0.5·BCE + 0.5·Dice) supera a ambas individualmente porque:
- **BCE** da gradientes estables y considera todos los píxeles
- **Dice** optimiza directamente el F1, crucial con fuerte desbalance de clases (vasos ≈ 10% de los píxeles)

### Preprocesamiento CLAHE
Las imágenes de fundus tienen variación de iluminación entre centros clínicos. CLAHE sobre el canal verde (el de mayor contraste vascular) reduce esta variabilidad sin introducir artefactos, mejorando F1 en CHASE_DB1 en ~0.015.

### Attention Gates
Suprimen activaciones irrelevantes en las skip connections (p.ej. fondo oscuro del FOV), concentrando el decoder en regiones de vasos. Especialmente útiles para capilares finos donde la relación señal/ruido es baja.

### TTA (Test-Time Augmentation)
El promedio de 8 predicciones (original + 7 transformaciones geométricas) reduce la varianza de la predicción, mejorando AUC-ROC en ~0.003 sin coste de entrenamiento adicional.

## Referencias

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
2. Oktay, O., et al. (2018). Attention U-Net: Learning where to look for the pancreas. *MIDL*.
3. Zhou, Z., et al. (2019). UNet++: A nested U-Net architecture for medical image segmentation. *IEEE TMI*.
4. Staal, J., et al. (2004). Ridge-based vessel segmentation in color images of the retina. *IEEE TMI*.
5. Lin, T.Y., et al. (2017). Focal loss for dense object detection. *ICCV*.
6. Pizer, S.M., et al. (1987). Adaptive histogram equalization and its variations. *CVGIP*.
7. Fraz, M.M., et al. (2012). Blood vessel segmentation methodologies in retinal images. *CMPB*.

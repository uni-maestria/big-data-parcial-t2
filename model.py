"""
model.py — Implementación desde cero de U-Net y variantes para segmentación de vasos retinianos.
Implementa: U-Net, Attention U-Net, U-Net++
Referencia: Ronneberger et al. (2015), Oktay et al. (2018), Zhou et al. (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Bloques fundamentales
# ──────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Bloque convolucional doble: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, dropout: float = 0.0):
        super().__init__()
        mid_ch = mid_ch or out_ch
        layers = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool 2×2 seguido de DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample bilineal + concatenación de skip + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Padding para alinear dimensiones si hay discrepancia
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# U-Net estándar
# ──────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net original (Ronneberger et al., 2015) con BN y dropout opcional.

    Args:
        in_channels: canales de entrada (3 para RGB, 1 para escala de grises).
        out_channels: 1 para segmentación binaria.
        base_filters: número de filtros en el primer nivel (duplica por nivel).
        depth: número de niveles de encoder (4 en el paper original).
        bilinear: True = upsample bilineal; False = ConvTranspose2d.
        dropout: tasa de dropout en bloques convolucionales.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, base_filters, dropout=dropout)
        self.downs = nn.ModuleList()
        ch = base_filters
        for _ in range(depth - 1):
            self.downs.append(DownBlock(ch, ch * 2, dropout=dropout))
            ch *= 2

        # Bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = DownBlock(ch, ch * 2 // factor, dropout=dropout)
        ch = ch * 2 // factor

        # Decoder
        self.ups = nn.ModuleList()
        skip_chs = [base_filters * (2 ** i) for i in range(depth)]
        for i in range(depth):
            skip_ch = skip_chs[depth - 1 - i]
            self.ups.append(UpBlock(ch + skip_ch, skip_ch // (factor if i < depth - 1 else 1),
                                     bilinear=bilinear, dropout=dropout))
            ch = skip_ch // (factor if i < depth - 1 else 1)

        self.outc = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = self.bottleneck(skips[-1])
        for i, up in enumerate(self.ups):
            x = up(x, skips[self.depth - 1 - i])
        return self.outc(x)


# ──────────────────────────────────────────────────────────────────────────────
# Attention Gate (para Attention U-Net)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Attention Gate de Oktay et al. (2018).
    Pondera las conexiones de salto usando la señal del decoder como guía.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Alinear dimensiones
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUpBlock(nn.Module):
    """Up block con attention gate en la skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reduce = nn.Conv2d(in_ch, skip_ch, kernel_size=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        self.attn = AttentionGate(F_g=skip_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = DoubleConv(skip_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        skip_att = self.attn(g=x, x=skip)
        return self.conv(torch.cat([skip_att, x], dim=1))


class AttentionUNet(nn.Module):
    """
    Attention U-Net (Oktay et al., 2018).
    Agrega puertas de atención en cada conexión de salto.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        f = base_filters
        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(in_channels, f, dropout=dropout)
        self.down1 = DownBlock(f,     f * 2,          dropout=dropout)
        self.down2 = DownBlock(f * 2, f * 4,          dropout=dropout)
        self.down3 = DownBlock(f * 4, f * 8,          dropout=dropout)
        self.down4 = DownBlock(f * 8, f * 16 // factor, dropout=dropout)

        # Decoder con attention
        # in_ch = canales del decoder (ya reducido por bilinear), skip_ch = canales del encoder
        d4 = f * 16 // factor          # canales del bottleneck
        self.up1 = AttentionUpBlock(d4,         f * 8, f * 8 // factor, bilinear, dropout)
        self.up2 = AttentionUpBlock(f * 8 // factor, f * 4, f * 4 // factor, bilinear, dropout)
        self.up3 = AttentionUpBlock(f * 4 // factor, f * 2, f * 2 // factor, bilinear, dropout)
        self.up4 = AttentionUpBlock(f * 2 // factor, f,     f,               bilinear, dropout)

        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        x  = self.down4(s4)
        x  = self.up1(x, s4)
        x  = self.up2(x, s3)
        x  = self.up3(x, s2)
        x  = self.up4(x, s1)
        return self.outc(x)


# ──────────────────────────────────────────────────────────────────────────────
# U-Net++ (nested skip connections)
# ──────────────────────────────────────────────────────────────────────────────

class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Zhou et al., 2019) — conexiones de salto densas anidadas.
    Implementa el esquema completo de 4 niveles con deep supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 32,
        deep_supervision: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [base_filters * (2 ** i) for i in range(5)]

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Nodos de la malla anidada X_{i,j}
        # Eje i = profundidad del encoder, eje j = paso del decoder
        self.conv0_0 = DoubleConv(in_channels,   nb_filter[0], dropout=dropout)
        self.conv1_0 = DoubleConv(nb_filter[0],  nb_filter[1], dropout=dropout)
        self.conv2_0 = DoubleConv(nb_filter[1],  nb_filter[2], dropout=dropout)
        self.conv3_0 = DoubleConv(nb_filter[2],  nb_filter[3], dropout=dropout)
        self.conv4_0 = DoubleConv(nb_filter[3],  nb_filter[4], dropout=dropout)

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1], dropout=dropout)
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2], dropout=dropout)
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3], dropout=dropout)

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], dropout=dropout)
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], dropout=dropout)

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], dropout=dropout)

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], dropout=dropout)

        # Cabezas de salida (deep supervision)
        self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def _up_cat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([skip, x], dim=1)

    def forward(self, x: torch.Tensor):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self._up_cat(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self._up_cat(x2_0, x1_0))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)[:, :, :x0_0.shape[2], :x0_0.shape[3]]], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self._up_cat(x3_0, x2_0))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)[:, :, :x1_0.shape[2], :x1_0.shape[3]]], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)[:, :, :x0_0.shape[2], :x0_0.shape[3]]], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self._up_cat(x4_0, x3_0))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)[:, :, :x2_0.shape[2], :x2_0.shape[3]]], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)[:, :, :x1_0.shape[2], :x1_0.shape[3]]], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)[:, :, :x0_0.shape[2], :x0_0.shape[3]]], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return (out1 + out2 + out3 + out4) / 4  # promedio en entrenamiento
        else:
            return self.final4(x0_4)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(arch: str = "unet", **kwargs) -> nn.Module:
    """Construye el modelo por nombre."""
    registry = {
        "unet":     UNet,
        "att_unet": AttentionUNet,
        "unetpp":   UNetPlusPlus,
    }
    if arch not in registry:
        raise ValueError(f"Arquitectura '{arch}' no reconocida. Opciones: {list(registry)}")
    return registry[arch](**kwargs)


if __name__ == "__main__":
    # Prueba rápida de dimensiones
    x = torch.randn(2, 3, 512, 512)
    for arch in ["unet", "att_unet", "unetpp"]:
        model = build_model(arch, in_channels=3, out_channels=1, base_filters=32)
        y = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{arch:10s} | salida: {tuple(y.shape)} | params: {params/1e6:.2f}M")

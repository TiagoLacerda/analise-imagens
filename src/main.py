"""
Analyse an image step by step: source image, color spaces, histogram and frequency spectrum.

Usage:
    python3 src/main.py PATH

Each step opens its figures in a window. Close them to move on to the next step.

Example:
    python3 src/main.py assets/images/chicken.jpg
"""

import argparse
import kornia.color as kornia_colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image


def parse_args() -> str:
    """
    Parse command line arguments.

    Returns the path to the image.
    """
    parser = argparse.ArgumentParser("main")
    parser.add_argument("path", help="Caminho de um arquivo de imagem")

    return parser.parse_args().path


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """
    Converts a (H, W, 3) RGB tensor to an (H, W, 3) CIELAB tensor.
    """
    rgb = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lab = kornia_colors.rgb_to_lab(rgb)
    
    return lab.squeeze(0).permute(1, 2, 0)


def rgb_to_hsl(image: torch.Tensor) -> torch.Tensor:
    """
    Converts a (H, W, 3) RGB tensor to an (H, W, 3) HSL tensor.
    """
    rgb = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    hls = kornia_colors.rgb_to_hls(rgb)
    hsl = hls[:, [0, 2, 1], ...]

    return hsl.squeeze(0).permute(1, 2, 0)


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    """
    Converts a (H, W, 3) RGB tensor to an (H, W, 3) XYZ tensor.
    """
    rgb = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    xyz = kornia_colors.rgb_to_xyz(rgb)

    return xyz.squeeze(0).permute(1, 2, 0)


def main():
    plt.style.use('dark_background')
    
    path = parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    image = np.array(Image.open(path).convert("RGB"))

    # 1. Source image

    tensor = torch.from_numpy(image).to(device)

    # Image

    fig, ax = plt.subplots()

    ax.imshow(tensor.cpu().numpy())

    ax.axis("off")

    # Histograms

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    
    colors = ("Red", "Green", "Blue")

    for i in range(3):
        axes[i].hist(tensor[..., i].ravel(), bins=256, range=(0, 255), color=colors[i])
        axes[i].set_title(colors[i])
        axes[i].set_xlim(0, 255)

    plt.tight_layout()
    plt.show(block=True)

    # 2. RGB -> CIELAB, HSL and XYZ

    lab = rgb_to_lab(tensor)
    hsl = rgb_to_hsl(tensor)
    xyz = rgb_to_xyz(tensor)

    channels = (
        (lab[..., 0], "CIELAB L*", 0, 100),
        (hsl[..., 2], "HSL l", 0, 1),
        (xyz[..., 1], "XYZ Y", 0, 1),
    )

    # Images

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i, (ax, (channel, title, lo, hi)) in enumerate(zip(axes.flat, channels)):
        ax.imshow(channel.cpu().numpy(), cmap="gray", vmin=lo, vmax=hi)
        ax.axis("off")

        if i < 2:
            ax.set_title(title)
        else:
            ax.set_title(title, y=0, pad=-6, verticalalignment="top")

    axes[1, 1].axis("off")

    fig.tight_layout()

    # Histograms

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    for i, (channel, title, lo, hi) in enumerate(channels):
        axes[i].hist(channel.cpu().numpy().ravel(), bins=256, color="gray")
        axes[i].set_title(title)
        axes[i].set_xlim(lo, hi)

    plt.tight_layout()
    plt.show(block=True)

    # 3. Histogram analysis

    tensor = (xyz[..., 1] * 255).clamp(0, 255)

    lo = 0
    hi = 215

    tensor_normalized = ((tensor.float() - lo) * (255.0 / (hi - lo))).clamp(0, 255).round().to(torch.uint8)

    # Images

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    axes[0].imshow(tensor.cpu().numpy(), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Y")
    axes[0].axis("off")

    axes[1].imshow(tensor_normalized.cpu().numpy(), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Y Normalizado")
    axes[1].axis("off")

    plt.tight_layout()

    # Histograms

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        tensor.cpu().numpy().ravel(),
        bins=256,
        range=(0, 255),
        alpha=0.4,
        label="Y",
    )

    ax.hist(
        tensor_normalized.cpu().numpy().ravel(),
        bins=256,
        range=(0, 255),
        histtype="step",
        linewidth=1.5,
        label="Y Normalizado",
    )

    ax.set_title("Distribuição da Luminância")
    ax.set_xlabel("Luminância")
    ax.set_ylabel("Pixels")
    ax.set_xlim(0, 255)
    ax.legend()

    plt.tight_layout()
    plt.show(block=True)

    # 4. Frequency analysis

    spectrum = torch.abs(torch.fft.fftshift(torch.fft.fft2(tensor.float())))
    magnitude = torch.log1p(spectrum)

    spectrum_normalized = torch.abs(torch.fft.fftshift(torch.fft.fft2(tensor_normalized.float())))
    magnitude_normalized = torch.log1p(spectrum_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(6, 10))

    axes[0].imshow(magnitude.cpu().numpy(), cmap="gray")
    axes[0].set_title("Transformada de Fourier 2D — Y")
    axes[0].axis("off")

    axes[1].imshow(magnitude_normalized.cpu().numpy(), cmap="gray")
    axes[1].set_title("Transformada de Fourier 2D — Y Normalizado")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show(block=True)



if __name__ == "__main__":
    main()

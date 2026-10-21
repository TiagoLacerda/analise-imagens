import argparse
import kornia.color as kornia_colors
import kornia.filters as kornia_filters
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image


def parse_args() -> str:
    """
    Parse command line arguments.

    Returns a string containing a path to an image.
    """
    parser = argparse.ArgumentParser("foo")
    parser.add_argument("path", help="Caminho de um arquivo de imagem", type=str)

    args = parser.parse_args()

    print(args)

    return args.path


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """
    Converts a (H, W, 3) RGB tensor to an (H, W, 3) CIELAB tensor.
    """
    rgb = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lab = kornia_colors.rgb_to_lab(rgb)
    
    return lab.squeeze(0).permute(1, 2, 0)


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

    # 2. RGB -> CIELAB

    tensor = rgb_to_lab(tensor)

    tensor = tensor[..., 0]

    tensor = (tensor * 2.55).clamp(0, 255)

    # Image

    fig, ax = plt.subplots()

    ax.imshow(tensor.cpu().numpy(), cmap="gray")

    ax.axis("off")

    # Histograms

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    channels = ("L*", "a*", "b*")
    colors = ("gray", "red", "blue")

    for i in range(3):
        axes[i].hist(tensor[..., i].ravel().cpu().numpy(), bins=256, color=colors[i])
        axes[i].set_title(channels[i])

    plt.tight_layout()
    plt.show(block=True)

    # 3. Histogram analysis

    lo = 0
    hi = 215

    tensor_normalized = ((tensor.float() - lo) * (255.0 / (hi - lo))).clamp(0, 255).round().to(torch.uint8)

    # Images

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    axes[0].imshow(tensor.cpu().numpy(), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(tensor_normalized.cpu().numpy(), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Normalizada")
    axes[1].axis("off")

    plt.tight_layout()

    # Histograms

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        tensor.cpu().numpy().ravel(),
        bins=256,
        range=(0, 255),
        alpha=0.4,
        label="Original",
    )

    ax.hist(
        tensor_normalized.cpu().numpy().ravel(),
        bins=256,
        range=(0, 255),
        histtype="step",
        linewidth=1.5,
        label="Normalizado",
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

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    axes[0].imshow(magnitude.cpu().numpy(), cmap="gray")
    axes[0].set_title("Transformada de Fourier 2D — Original")
    axes[0].axis("off")

    axes[1].imshow(magnitude_normalized.cpu().numpy(), cmap="gray")
    axes[1].set_title("Transformada de Fourier 2D — Normalizada")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show(block=True)



if __name__ == "__main__":
    main()
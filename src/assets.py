"""
Save the images of an input image used in the presentation.

Usage:
    python3 src/assets.py PATH

Example:
    python3 src/assets.py assets/images/chicken.jpg

The images are saved to the output/<name>/ folder in the project root, and are
meant to be displayed over a white background. For an input image
<name>.<extension>:

    <name>.<extension>               Input image
    <name>_blurred_2.png             Gaussian blur, sigma = 2
    <name>_blurred_3.png             Gaussian blur, sigma = 3
    <name>_dog_9_2_3.png             DoG of the XYZ Y channel, radius = 9, sigmas = 2 and 3
    <name>_sobel.png                 Sobel filter of the XYZ Y channel
    <name>_sobel_x.png               Sobel gradient Gx of the XYZ Y channel, zero is mid-gray
    <name>_sobel_y.png               Sobel gradient Gy of the XYZ Y channel, zero is mid-gray
    <name>_sobel_xy.png              Absolute Gx in the red channel, absolute Gy in the green channel
    <name>_eigenvector_major.png     Major structure tensor eigenvector, x in the red channel,
                                     y in the green channel, zero is 0.5
    <name>_eigenvector_minor.png     Minor structure tensor eigenvector, same channels
    <name>_canny.png                 Canny edges of the XYZ Y channel
    <name>_luminance_lab.png         CIELAB L* channel
    <name>_luminance_hsl.png         HSL l channel
    <name>_luminance_xyz.png         XYZ Y channel
    <name>_luminance.png             The three luminance channels in a 2x2 grid
    <name>_histograms_rgb.png        Histograms of the RGB channels
    <name>_histograms_luminance.png  Histograms of the three luminance channels
    <name>_contrast_balancing.png    XYZ Y channel after histogram equalization
    <name>_contrast_balancing_histograms.png
                                     Histograms of the XYZ Y channel, before and after
                                     histogram equalization
    <name>_spectrum.png              Fourier transform of the XYZ Y channel
"""

import argparse
import shutil
from pathlib import Path

import kornia.enhance as kornia_enhance
import kornia.filters as kornia_filters
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

from assets_static import style_axes
from util import apply_filter, rgb_to_hsl, rgb_to_lab, rgb_to_xyz

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

BLUR_RADIUS = 9
BLUR_SIGMAS = (2, 3)

STRUCTURE_TENSOR_SIGMA = 2
COHERENCE_EPSILON = 1e-3


def parse_args() -> str:
    """
    Parse command line arguments.

    Returns the path to the image.
    """
    parser = argparse.ArgumentParser("assets")
    parser.add_argument("path", help="Caminho de um arquivo de imagem")

    return parser.parse_args().path


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure and release its memory."""
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_blurred(path: str, output_dir: Path, name: str) -> None:
    """Save the input image blurred with each of the gaussian sigmas."""
    for sigma in BLUR_SIGMAS:
        blurred = apply_filter(path, "gaussian", BLUR_RADIUS, sigma)

        plt.imsave(
            output_dir / f"{name}_blurred_{sigma}.png",
            blurred.clamp(0, 1).cpu().numpy(),
        )


def save_dog(path: str, output_dir: Path, name: str) -> None:
    """Save the difference of gaussians of the XYZ Y channel."""
    small_sigma, large_sigma = sorted(BLUR_SIGMAS)
    dog = apply_filter(path, "dog", BLUR_RADIUS, small_sigma, large_sigma, luminance=True)

    plt.imsave(
        output_dir / f"{name}_dog_{BLUR_RADIUS}_{small_sigma}_{large_sigma}.png",
        dog.cpu().numpy(),
        cmap="gray",
    )


def save_edges(path: str, output_dir: Path, name: str) -> None:
    """Save the Sobel and Canny filters of the XYZ Y channel."""
    for filter_name in ("sobel", "canny"):
        edges = apply_filter(path, filter_name, luminance=True)

        plt.imsave(
            output_dir / f"{name}_{filter_name}.png",
            edges.cpu().numpy(),
            cmap="gray",
        )


def sobel_gradients(luminance: torch.Tensor) -> torch.Tensor:
    """Returns a (2, H, W) tensor with the signed Sobel gradients Gx and Gy of a luminance channel."""
    gradients = kornia_filters.spatial_gradient(luminance.unsqueeze(0).unsqueeze(0))

    return gradients.squeeze(0).squeeze(0)


def save_sobel_gradients(luminance: torch.Tensor, output_dir: Path, name: str) -> None:
    """Save the Sobel gradients of a luminance channel, on a shared scale."""
    gradients = sobel_gradients(luminance)
    gx, gy = gradients / gradients.abs().max()

    # Signed gradients: zero is mid-gray, negative is darker and positive is lighter
    for axis, gradient in (("x", gx), ("y", gy)):
        plt.imsave(
            output_dir / f"{name}_sobel_{axis}.png",
            gradient.cpu().numpy(),
            cmap="gray",
            vmin=-1,
            vmax=1,
        )

    plt.imsave(
        output_dir / f"{name}_sobel_xy.png",
        torch.stack((gx.abs(), gy.abs(), torch.zeros_like(gx)), dim=-1).cpu().numpy(),
    )


def gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Returns a (C, H, W) tensor blurred with a gaussian of the given sigma."""
    kernel_size = 2 * BLUR_RADIUS + 1
    blurred = kornia_filters.gaussian_blur2d(
        tensor.unsqueeze(0),
        (kernel_size, kernel_size),
        (sigma, sigma),
    )

    return blurred.squeeze(0)


def structure_tensor(luminance: torch.Tensor) -> torch.Tensor:
    """Returns a (3, H, W) tensor with the smoothed structure tensor components E, F and G."""
    gx, gy = sobel_gradients(luminance)
    tensor = torch.stack((gx * gx, gx * gy, gy * gy))

    return gaussian_blur(tensor, STRUCTURE_TENSOR_SIGMA)


def eigenvectors(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the (2, H, W) unit major and minor eigenvectors of a structure tensor."""
    # Dividing by the trace makes the result independent of brightness, and the
    # difference of the eigenvalues becomes the coherence: (l1 - l2) / (l1 + l2)
    trace = (e + g).clamp(min=torch.finfo(e.dtype).tiny)
    e, f, g = e / trace, f / trace, g / trace

    coherence = torch.sqrt((e - g) ** 2 + 4 * f ** 2)
    major_eigenvalue = 0.5 * (e + g + coherence)

    # Both are eigenvectors of the major eigenvalue, but each one collapses to zero
    # on some axis-aligned edges. The longer one is always usable.
    a = torch.stack((f, major_eigenvalue - e))
    b = torch.stack((major_eigenvalue - g, f))
    major = torch.where((a * a).sum(0) > (b * b).sum(0), a, b)

    # Isotropic regions have no orientation, and are left as zero vectors
    v1 = torch.where(
        coherence > COHERENCE_EPSILON,
        major * torch.rsqrt((major * major).sum(0)),
        torch.zeros_like(major),
    )
    v2 = torch.stack((v1[1], -v1[0]))

    return v1, v2


def save_eigenvectors(luminance: torch.Tensor, output_dir: Path, name: str) -> None:
    """Save the major and minor eigenvectors, with x as the red channel and y as the green channel."""
    v1, v2 = eigenvectors(*structure_tensor(luminance))

    for label, vector in (("major", v1), ("minor", v2)):
        # Signed values: zero is 0.5
        red, green = 0.5 + 0.5 * vector

        plt.imsave(
            output_dir / f"{name}_eigenvector_{label}.png",
            torch.stack((red, green, torch.zeros_like(red)), dim=-1).cpu().numpy(),
        )


def save_luminance(channels: tuple, output_dir: Path, name: str) -> None:
    """Save each luminance channel as a grayscale image."""
    for space, _, channel, lo, hi in channels:
        plt.imsave(
            output_dir / f"{name}_luminance_{space}.png",
            channel.cpu().numpy(),
            cmap="gray",
            vmin=lo,
            vmax=hi,
        )


def save_luminance_grid(channels: tuple, path: Path) -> None:
    """Save the luminance channels in a 2x2 grid, with an empty bottom right quadrant."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), facecolor="white")

    for i, (ax, (_, title, channel, lo, hi)) in enumerate(zip(axes.flat, channels)):
        ax.imshow(channel.cpu().numpy(), cmap="gray", vmin=lo, vmax=hi)
        ax.axis("off")

        if i < 2:
            ax.set_title(title, fontfamily="serif")
        else:
            ax.set_title(
                title,
                fontfamily="serif",
                y=0,
                pad=-6,
                verticalalignment="top",
            )

    axes[1, 1].axis("off")

    save_figure(fig, path)


def save_histograms(histograms: tuple, path: Path) -> None:
    """Save a column of histograms, given as (values, title, color, lo, hi) tuples."""
    fig, axes = plt.subplots(len(histograms), 1, figsize=(8, 6), facecolor="white")

    for ax, (values, title, color, lo, hi) in zip(axes, histograms):
        style_axes(ax)
        ax.set_axisbelow(True)
        ax.hist(values.cpu().numpy().ravel(), bins=256, range=(lo, hi), color=color)
        ax.set_title(title, fontfamily="serif")
        ax.set_xlim(lo, hi)

    save_figure(fig, path)


def save_contrast_balancing(luminance: torch.Tensor, output_dir: Path, name: str) -> None:
    """Save a luminance channel after histogram equalization, and its histograms before and after."""
    equalized = kornia_enhance.equalize(luminance.clamp(0, 1).unsqueeze(0)).squeeze(0)

    plt.imsave(
        output_dir / f"{name}_contrast_balancing.png",
        equalized.cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    style_axes(ax)
    ax.set_axisbelow(True)

    ax.hist(
        luminance.cpu().numpy().ravel(),
        bins=256,
        range=(0, 1),
        alpha=0.4,
        label="Y",
    )

    ax.hist(
        equalized.cpu().numpy().ravel(),
        bins=256,
        range=(0, 1),
        histtype="step",
        linewidth=1.5,
        label="Y Equalizado",
    )

    ax.set_title("Distribuição da Luminância", fontfamily="serif")
    ax.set_xlabel("Luminância", fontfamily="serif")
    ax.set_ylabel("Pixels", fontfamily="serif")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center")

    save_figure(fig, output_dir / f"{name}_contrast_balancing_histograms.png")


def save_spectrum(luminance: torch.Tensor, path: Path) -> None:
    """Save the log magnitude of the Fourier transform of a luminance channel."""
    spectrum = torch.abs(torch.fft.fftshift(torch.fft.fft2(luminance.float())))
    magnitude = torch.log1p(spectrum)

    plt.imsave(path, magnitude.cpu().numpy(), cmap="gray")


def main():
    path = parse_args()
    name = Path(path).stem
    output_dir = OUTPUT_DIR / name

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    image = np.array(Image.open(path).convert("RGB"))
    tensor = torch.from_numpy(image).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source image

    shutil.copy(path, output_dir / Path(path).name)

    colors = ("Red", "Green", "Blue")

    save_histograms(
        [(tensor[..., i], color, color, 0, 255) for i, color in enumerate(colors)],
        output_dir / f"{name}_histograms_rgb.png",
    )

    # 2. Filters

    save_blurred(path, output_dir, name)
    save_dog(path, output_dir, name)
    save_edges(path, output_dir, name)

    # 3. RGB -> CIELAB, HSL and XYZ

    lab = rgb_to_lab(tensor)
    hsl = rgb_to_hsl(tensor)
    xyz = rgb_to_xyz(tensor)

    channels = (
        ("lab", "CIELAB L*", lab[..., 0], 0, 100),
        ("hsl", "HSL l", hsl[..., 2], 0, 1),
        ("xyz", "XYZ Y", xyz[..., 1], 0, 1),
    )

    save_luminance(channels, output_dir, name)
    save_luminance_grid(channels, output_dir / f"{name}_luminance.png")

    save_histograms(
        [(channel, title, "gray", lo, hi) for _, title, channel, lo, hi in channels],
        output_dir / f"{name}_histograms_luminance.png",
    )

    # 4. Edge analysis

    save_sobel_gradients(xyz[..., 1], output_dir, name)
    save_eigenvectors(xyz[..., 1], output_dir, name)

    # 5. Histogram analysis

    save_contrast_balancing(xyz[..., 1], output_dir, name)

    # 6. Frequency analysis

    save_spectrum((xyz[..., 1] * 255).clamp(0, 255), output_dir / f"{name}_spectrum.png")


if __name__ == "__main__":
    main()

"""
Save the images of an input image used in the presentation.

Usage:
    python3 src/assets.py PATH

Example:
    python3 src/assets.py latex/assets/images/chicken.jpg

The images are saved to the latex/assets/images/ folder in the project root, where
the presentation reads them, and are meant to be displayed over a white background.
For an input image <name>.<extension>:

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
    <name>_flow.png                  Minor eigenvectors, the direction of least change, as line
                                     segments over the XYZ Y channel, with the crop outlined
    <name>_flow_crop.png             The same segments, denser, over a crop of the image center,
                                     with the pixel crop outlined
    <name>_flow_pixels.png           One segment per pixel, over a smaller crop of the image center
    <name>_transversal_blur.png      XYZ Y channel blurred along a straight line across the edge,
                                     in the direction of greatest change of each pixel
    <name>_lic_blur.png              XYZ Y channel blurred along the curve that follows the
                                     direction of least change (line integral convolution)
    <name>_transversal_pixels.png    Close-up of the points the transversal blur reads for the
                                     center pixel, sized by their weight
    <name>_lic_pixels.png            Close-up of the points the LIC blur reads for the center pixel
    <name>_transversal_dog.png       Flow-based XDoG, step 1: DoG of the XYZ Y channel across the
                                     edges, clipped to [0, 1]
    <name>_fdog_lic.png              Step 2: that response blurred along the edges with LIC,
                                     clipped to [0, 1]
    <name>_fdog.png                  Step 3: that response after the soft threshold
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
import torch.nn.functional as F

from PIL import Image

from assets_static import style_axes
from util import apply_filter, rgb_to_hsl, rgb_to_lab, rgb_to_xyz

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "latex" / "assets" / "images"

BLUR_RADIUS = 9
BLUR_SIGMAS = (2, 3)

STRUCTURE_TENSOR_SIGMA = 2
COHERENCE_EPSILON = 1e-3

FLOW_SPACING = 12
FLOW_CROP_SIZE = 128
FLOW_CROP_SPACING = 6
FLOW_PIXELS_SIZE = 24

FLOW_BLUR_SIGMA = 5
FLOW_BLUR_RADIUS = 15

XDOG_SIGMA = 1.0
XDOG_K = 1.6
XDOG_P = 20
XDOG_SIGMA_M = 3.0
# Linear Y is mostly below 0.3, so the threshold is lower than the usual 0.5
XDOG_EPSILON = 0.1
XDOG_PHI = 50


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


def plot_flow(
    luminance: torch.Tensor,
    flow: torch.Tensor,
    region: tuple[int, int, int, int],
    spacing: int,
    path: Path,
    outline: tuple[int, int, int, int] | None = None,
    pixel_grid: bool = False,
    samples: list[tuple[int, float, float]] | None = None,
) -> None:
    """
    Plot a (2, H, W) direction field as line segments over a region of a luminance channel.

    Regions are (left, top, width, height) tuples. One segment is drawn every spacing
    pixels, the outline region, if given, is drawn as a rectangle, and pixel_grid draws
    the pixel borders. Samples, if given, are the (step, x, y) points a blur reads for one
    pixel, drawn in order with their size given by their gaussian weight.
    """
    left, top, width, height = region

    fig, ax = plt.subplots(figsize=(8, 8 * height / width), facecolor="white")

    # Faded so the segments stand out
    ax.imshow(
        luminance[top:top + height, left:left + width].cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=0.5,
    )

    ys, xs = np.mgrid[spacing // 2:height:spacing, spacing // 2:width:spacing]
    u, v = flow[:, top + ys, left + xs].cpu().numpy()

    # Isotropic regions have no direction to draw
    drawn = (u != 0) | (v != 0)

    # The sign of an eigenvector is arbitrary, so segments have no arrowheads and are
    # centered on their pixel. Data coordinates keep the y axis pointing down, like the image
    ax.quiver(
        xs[drawn],
        ys[drawn],
        u[drawn],
        v[drawn],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1 / (0.8 * spacing),
        pivot="middle",
        width=0.003,
        headwidth=0,
        headlength=0,
        headaxislength=0,
    )

    if outline is not None:
        outline_left, outline_top, outline_width, outline_height = outline
        ax.add_patch(plt.Rectangle(
            (outline_left - left - 0.5, outline_top - top - 0.5),
            outline_width,
            outline_height,
            fill=False,
            edgecolor="blue",
            linewidth=1.5,
        ))

    if pixel_grid:
        borders = np.arange(-0.5, max(width, height))
        ax.vlines(borders[:width + 1], -0.5, height - 0.5, color="white", linewidth=0.5)
        ax.hlines(borders[:height + 1], -0.5, width - 0.5, color="white", linewidth=0.5)

    if samples is not None:
        steps, xs, ys = (np.array(values) for values in zip(*samples))
        xs, ys = xs - left, ys - top

        ax.plot(xs, ys, color="blue", linewidth=1)
        ax.scatter(xs, ys, s=80 * gaussian_weight(steps), color="blue", zorder=3)

        # The pixel being blurred
        center = steps == 0
        ax.add_patch(plt.Rectangle(
            (xs[center][0] - 0.5, ys[center][0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        ))

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")

    save_figure(fig, path)


def centered(size: int, height: int, width: int) -> tuple[int, int, int, int]:
    """Returns the (left, top, width, height) square region of the given size at the center of an image."""
    size = min(size, height, width)

    return (width - size) // 2, (height - size) // 2, size, size


def save_flow(luminance: torch.Tensor, output_dir: Path, name: str) -> None:
    """Save the minor eigenvectors, the direction of least change, over the whole image and two centered crops."""
    _, v2 = eigenvectors(*structure_tensor(luminance))

    height, width = luminance.shape
    crop = centered(FLOW_CROP_SIZE, height, width)
    pixels = centered(FLOW_PIXELS_SIZE, height, width)

    plot_flow(luminance, v2, (0, 0, width, height), FLOW_SPACING, output_dir / f"{name}_flow.png", outline=crop)
    plot_flow(luminance, v2, crop, FLOW_CROP_SPACING, output_dir / f"{name}_flow_crop.png", outline=pixels)
    plot_flow(luminance, v2, pixels, 1, output_dir / f"{name}_flow_pixels.png", pixel_grid=True)


def sample(tensor: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, mode: str) -> torch.Tensor:
    """Returns a (C, H, W) tensor sampled at the (H, W) pixel coordinates xs and ys, clamped to the border."""
    _, height, width = tensor.shape
    grid = torch.stack((2 * xs / (width - 1) - 1, 2 * ys / (height - 1) - 1), dim=-1)

    sampled = F.grid_sample(
        tensor.unsqueeze(0),
        grid.unsqueeze(0),
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )

    return sampled.squeeze(0)


def gaussian_weight(step: int | np.ndarray, sigma: float = FLOW_BLUR_SIGMA) -> float | np.ndarray:
    """Returns the unnormalized 1D gaussian weight at a distance of step pixels."""
    return np.exp(-step ** 2 / (2 * sigma ** 2))


def pixel_coordinates(luminance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the (H, W) x and y coordinates of every pixel of a luminance channel."""
    height, width = luminance.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=luminance.dtype, device=luminance.device),
        torch.arange(width, dtype=luminance.dtype, device=luminance.device),
        indexing="ij",
    )

    return xs, ys


def line_path(
    flow: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    radius: int = FLOW_BLUR_RADIUS,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Returns the (step, x, y) points along a straight line in the flow direction of each start point.

    The points are ordered from step -radius to radius, and xs and ys can have any (H, W) shape.
    """
    tx, ty = sample(flow, xs, ys, "nearest")

    return [(step, xs + step * tx, ys + step * ty) for step in range(-radius, radius + 1)]


def lic_path(
    flow: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    radius: int = FLOW_BLUR_RADIUS,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Returns the (step, x, y) points along the curve that follows the flow from each start point.

    The points are ordered from step -radius to radius, and xs and ys can have any (H, W) shape.
    """
    halves = []

    # Walk the curve forwards and backwards from each start point
    for direction in (1, -1):
        x, y = xs, ys
        tx, ty = sample(flow, x, y, "nearest")
        dx, dy = direction * tx, direction * ty
        half = []

        for step in range(1, radius + 1):
            x, y = x + dx, y + dy
            half.append((direction * step, x, y))

            # Interpolating v and -v would cancel out, so the flow is read from the nearest
            # pixel. Its sign is arbitrary, so it is flipped to keep walking the same way
            tx, ty = sample(flow, x, y, "nearest")
            sign = torch.where(tx * dx + ty * dy < 0, -1.0, 1.0)
            dx, dy = sign * tx, sign * ty

        halves.append(half)

    forwards, backwards = halves

    return backwards[::-1] + [(0, xs, ys)] + forwards


def path_blur(
    luminance: torch.Tensor,
    path: list[tuple[int, torch.Tensor, torch.Tensor]],
    sigma: float = FLOW_BLUR_SIGMA,
) -> torch.Tensor:
    """Returns a luminance channel blurred with a 1D gaussian over the points of a path."""
    total = torch.zeros_like(luminance)
    weights = 0.0

    for step, x, y in path:
        weight = gaussian_weight(step, sigma)
        total += weight * sample(luminance.unsqueeze(0), x, y, "bilinear")[0]
        weights += weight

    return total / weights


def save_flow_blur(luminance: torch.Tensor, output_dir: Path, name: str) -> None:
    """
    Save a luminance channel blurred across the edges, along a straight line, and along them,
    following the curve. Also save a close-up of the points each blur reads for the center pixel.
    """
    v1, v2 = eigenvectors(*structure_tensor(luminance))

    height, width = luminance.shape
    xs, ys = pixel_coordinates(luminance)

    # Large enough to hold every point read for the center pixel
    pixels = centered(2 * FLOW_BLUR_RADIUS + 6, height, width)
    center_x = torch.full((1, 1), float(width // 2), dtype=luminance.dtype, device=luminance.device)
    center_y = torch.full((1, 1), float(height // 2), dtype=luminance.dtype, device=luminance.device)

    for label, walk, flow in (("transversal", line_path, v1), ("lic", lic_path, v2)):
        plt.imsave(
            output_dir / f"{name}_{label}_blur.png",
            path_blur(luminance, walk(flow, xs, ys)).cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        samples = [(step, x.item(), y.item()) for step, x, y in walk(flow, center_x, center_y)]

        # The segments always show the direction of least change, which the transversal
        # blur crosses and the LIC blur follows
        plot_flow(
            luminance,
            v2,
            pixels,
            1,
            output_dir / f"{name}_{label}_pixels.png",
            pixel_grid=True,
            samples=samples,
        )


def save_fdog(luminance: torch.Tensor, output_dir: Path, name: str) -> None:
    """
    Save the steps of the flow-based XDoG of a luminance channel: the DoG across the edges,
    that response blurred along the edges, and the thresholded result.
    """
    v1, v2 = eigenvectors(*structure_tensor(luminance))
    xs, ys = pixel_coordinates(luminance)

    # 1. Across the edges: both gaussians read the same straight line, long enough for the wider one
    across = line_path(v1, xs, ys, int(np.ceil(3 * XDOG_K * XDOG_SIGMA)))
    narrow = path_blur(luminance, across, XDOG_SIGMA)
    wide = path_blur(luminance, across, XDOG_K * XDOG_SIGMA)
    dog = (1 + XDOG_P) * narrow - XDOG_P * wide

    # 2. Along the edges: joins the response into lines and removes noise
    along = lic_path(v2, xs, ys, int(np.ceil(3 * XDOG_SIGMA_M)))
    smoothed = path_blur(dog, along, XDOG_SIGMA_M)

    # 3. Soft threshold: white above epsilon, falling smoothly to black below it
    thresholded = torch.where(
        smoothed >= XDOG_EPSILON,
        torch.ones_like(smoothed),
        1 + torch.tanh(XDOG_PHI * (smoothed - XDOG_EPSILON)),
    )

    # The responses are unbounded, and are clipped to the displayable range
    for label, image in (("transversal_dog", dog), ("fdog_lic", smoothed), ("fdog", thresholded)):
        plt.imsave(
            output_dir / f"{name}_{label}.png",
            image.clamp(0, 1).cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
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
    output_dir = OUTPUT_DIR

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    image = np.array(Image.open(path).convert("RGB"))
    tensor = torch.from_numpy(image).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source image

    # The input image may already be in the output folder
    source = output_dir / Path(path).name
    if Path(path).resolve() != source.resolve():
        shutil.copy(path, source)

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
    save_flow(xyz[..., 1], output_dir, name)
    save_flow_blur(xyz[..., 1], output_dir, name)
    save_fdog(xyz[..., 1], output_dir, name)

    # 5. Histogram analysis

    save_contrast_balancing(xyz[..., 1], output_dir, name)

    # 6. Frequency analysis

    save_spectrum((xyz[..., 1] * 255).clamp(0, 255), output_dir / f"{name}_spectrum.png")


if __name__ == "__main__":
    main()

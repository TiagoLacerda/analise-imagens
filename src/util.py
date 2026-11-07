from math import isfinite

import kornia.color as kornia_colors
import kornia.filters as kornia_filters
import numpy as np
import torch

from PIL import Image


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
    Converts a (H, W, 3) sRGB tensor to an (H, W, 3) XYZ tensor.
    """
    rgb = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    xyz = kornia_colors.rgb_to_xyz(kornia_colors.rgb_to_linear_rgb(rgb))

    return xyz.squeeze(0).permute(1, 2, 0)


def apply_filter(
    path: str,
    filter_name: str,
    radius: int = 0,
    sigma: float = 0.0,
    sigma_2: float = 0.0,
    luminance: bool = False,
) -> torch.Tensor:
    """
    Applies a sobel, canny, gaussian or dog filter to the image in path.

    Filters the RGB channels, or only the XYZ Y channel when luminance is set.
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    image = np.array(Image.open(path).convert("RGB"))
    tensor = torch.from_numpy(image).to(device)
    filter_input = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0

    if luminance:
        filter_input = rgb_to_xyz(tensor)[..., 1].unsqueeze(0).unsqueeze(0)

    if filter_name == "sobel":
        result = kornia_filters.sobel(filter_input)
    elif filter_name == "canny":
        _, result = kornia_filters.canny(filter_input)
    elif filter_name == "gaussian":
        if radius <= 0 or not isfinite(sigma) or sigma <= 0:
            raise ValueError("radius e sigma devem ser maiores que zero")

        kernel_size = 2 * radius + 1
        result = kornia_filters.gaussian_blur2d(
            filter_input,
            (kernel_size, kernel_size),
            (sigma, sigma),
        )
    elif filter_name == "dog":
        if radius <= 0 or any(
            not isfinite(value) or value <= 0 for value in (sigma, sigma_2)
        ):
            raise ValueError("radius e os dois sigmas devem ser maiores que zero")

        sigma_1, sigma_2 = sorted((sigma, sigma_2))
        kernel_size = 2 * radius + 1
        blur_1 = kornia_filters.gaussian_blur2d(
            filter_input,
            (kernel_size, kernel_size),
            (sigma_1, sigma_1),
        )
        blur_2 = kornia_filters.gaussian_blur2d(
            filter_input,
            (kernel_size, kernel_size),
            (sigma_2, sigma_2),
        )
        result = (blur_1 - blur_2).clamp(0, 255)
    else:
        raise ValueError("Filtro inválido")

    if luminance:
        return result.squeeze(0).squeeze(0)

    return result.squeeze(0).permute(1, 2, 0)

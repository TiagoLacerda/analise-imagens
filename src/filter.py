import argparse
from math import isfinite
from pathlib import Path

import kornia.color as kornia_colors
import kornia.filters as kornia_filters
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image


def parse_args() -> tuple[str, str, int, float, float, bool, str]:
    parser = argparse.ArgumentParser("filter")
    parser.add_argument("path", help="Caminho de um arquivo de imagem", type=str)
    parser.add_argument(
        "filter",
        choices=("sobel", "canny", "gaussian", "dog"),
        help="Filtro a ser aplicado",
    )
    parser.add_argument(
        "arguments",
        nargs="*",
        help="Parâmetros do filtro e caminho de saída",
    )
    parser.add_argument(
        "-L",
        "--luminance",
        action="store_true",
        help="Aplica o filtro somente ao canal Y",
    )

    args = parser.parse_intermixed_args()

    print(args)

    parameter_count = {"gaussian": 2, "dog": 3}.get(args.filter, 0)
    if len(args.arguments) not in (parameter_count, parameter_count + 1):
        parser.error("parâmetros inválidos ou caminho de saída extra")

    radius = 0
    sigma_1 = 0.0
    sigma_2 = 0.0
    save_path = (
        args.arguments[parameter_count]
        if len(args.arguments) > parameter_count
        else ""
    )

    if args.filter in ("gaussian", "dog"):
        try:
            parameters = [float(value) for value in args.arguments[:parameter_count]]
        except ValueError:
            parser.error("radius e sigma devem ser números")

        radius_value = parameters[0]
        if (
            not isfinite(radius_value)
            or not radius_value.is_integer()
            or radius_value <= 0
        ):
            parser.error("radius deve ser um inteiro maior que zero")
        radius = int(radius_value)

        sigmas = parameters[1:]
        if any(not isfinite(sigma) or sigma <= 0 for sigma in sigmas):
            parser.error("sigma deve ser um número finito maior que zero")
        sigma_1 = sigmas[0]
        if args.filter == "dog":
            sigma_1, sigma_2 = sorted(sigmas)

    return (
        args.path,
        args.filter,
        radius,
        sigma_1,
        sigma_2,
        args.luminance,
        save_path,
    )


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    rgb = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    xyz = kornia_colors.rgb_to_xyz(rgb)

    return xyz.squeeze(0).permute(1, 2, 0)


def apply_filter(
    path: str,
    filter_name: str,
    radius: int = 0,
    sigma: float = 0.0,
    sigma_2: float = 0.0,
    luminance: bool = False,
) -> torch.Tensor:
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


def main():
    plt.style.use("dark_background")

    path, filter_name, radius, sigma, sigma_2, luminance, save_path = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    image = np.array(Image.open(path).convert("RGB"))
    tensor = torch.from_numpy(image).to(device)

    fig, ax = plt.subplots()

    ax.imshow(tensor.cpu().numpy())
    ax.axis("off")

    result = apply_filter(path, filter_name, radius, sigma, sigma_2, luminance)

    fig, ax = plt.subplots()

    if luminance:
        ax.imshow(result.cpu().numpy(), cmap="gray")
    else:
        ax.imshow(result.cpu().numpy())
    ax.axis("off")

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            output_path,
            result.cpu().numpy(),
            cmap="gray" if luminance else None,
        )

    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()

import argparse

import kornia.color as kornia_colors
import kornia.filters as kornia_filters
import numpy as np
import torch

from PIL import Image


def parse_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser("filter")
    parser.add_argument("path", help="Caminho de um arquivo de imagem", type=str)
    parser.add_argument(
        "filter",
        choices=("sobel", "canny", "dog"),
        help="Filtro a ser aplicado",
    )

    args = parser.parse_args()

    return args.path, args.filter


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    rgb = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    xyz = kornia_colors.rgb_to_xyz(rgb)

    return xyz.squeeze(0).permute(1, 2, 0)


def apply_filter(path: str, filter_name: str) -> torch.Tensor:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    image = np.array(Image.open(path).convert("RGB"))
    tensor = torch.from_numpy(image).to(device)
    y = rgb_to_xyz(tensor)[..., 1].unsqueeze(0).unsqueeze(0)

    if filter_name == "sobel":
        result = kornia_filters.sobel(y)
    elif filter_name == "canny":
        _, result = kornia_filters.canny(y)
    elif filter_name == "dog":
        blur_1 = kornia_filters.gaussian_blur2d(y, (5, 5), (1.0, 1.0))
        blur_2 = kornia_filters.gaussian_blur2d(y, (11, 11), (2.0, 2.0))
        result = blur_1 - blur_2
    else:
        raise ValueError("Filtro inválido")

    return result.squeeze(0).squeeze(0)


def main():
    path, filter_name = parse_args()
    return apply_filter(path, filter_name)


if __name__ == "__main__":
    main()

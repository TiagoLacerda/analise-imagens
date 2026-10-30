import argparse
from math import isfinite

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> tuple[str, list[float], str]:
    """
    Parse command line arguments.

    Returns the figure name, sigmas, and path to save the figure.
    """
    parser = argparse.ArgumentParser("figures")
    parser.add_argument("figure", help="Figura a ser gerada", type=str)
    parser.add_argument(
        "sigmas",
        nargs="*",
        help="Um ou dois valores de sigma para a gaussiana",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="gaussian.png",
        help="Caminho para salvar a figura",
    )

    args = parser.parse_args()

    print(args)

    if args.figure != "gaussian":
        return args.figure, [], args.output

    if len(args.sigmas) > 2:
        parser.error("informe no máximo dois valores de sigma")

    try:
        sigmas = [float(sigma) for sigma in args.sigmas] or [1.0]
    except ValueError:
        parser.error("sigma deve ser um número")

    if any(not isfinite(sigma) or sigma <= 0 for sigma in sigmas):
        parser.error("sigma deve ser um número finito maior que zero")

    return args.figure, sigmas, args.output


def main():
    figure, sigmas, path = parse_args()

    if figure != "gaussian":
        return

    x = np.linspace(-10, 10, 10000)

    fig, ax = plt.subplots(facecolor="white")
    ax.set_facecolor("white")

    colors = ("blue", "blue")
    curves = []

    for sigma, color in zip(sigmas, colors):
        y = np.exp(-(x ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        curves.append((sigma, y))
        ax.plot(x, y, color=color, label=f"σ = {sigma}")

    if len(curves) == 2:
        smaller_sigma, smaller_curve = min(curves, key=lambda curve: curve[0])
        larger_sigma, larger_curve = max(curves, key=lambda curve: curve[0])
        ax.plot(
            x,
            smaller_curve - larger_curve,
            color="purple",
            linestyle="--",
            label=f"σ = {smaller_sigma} − σ = {larger_sigma}",
        )

    ax.set_xlabel("x", fontfamily="serif")
    ax.set_ylabel("f(x)", fontfamily="serif")
    ax.set_ylim(top=0.25, bottom=-0.049)
    ax.set_xlim(left=-10, right=10)
    ax.grid(True, linewidth=0.5, color="lightgray")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend()

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()

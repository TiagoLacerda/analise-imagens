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
        default=None,
        help="Caminho para salvar a figura",
    )

    args = parser.parse_args()

    print(args)

    path = args.output or f"{args.figure}.png"

    if args.figure != "gaussian":
        return args.figure, [], path

    if len(args.sigmas) > 2:
        parser.error("informe no máximo dois valores de sigma")

    try:
        sigmas = [float(sigma) for sigma in args.sigmas] or [1.0]
    except ValueError:
        parser.error("sigma deve ser um número")

    if any(not isfinite(sigma) or sigma <= 0 for sigma in sigmas):
        parser.error("sigma deve ser um número finito maior que zero")

    return args.figure, sigmas, path


def main():
    figure, sigmas, path = parse_args()

    if figure == "hystheresis":
        x = np.linspace(0, 10, 1000)
        response = (
            0.95 * np.exp(-((x - 2.8) ** 2) / 0.18)
            + 0.55 * np.exp(-((x - 4.3) ** 2) / 0.5)
            + 0.42 * np.exp(-((x - 7.4) ** 2) / 0.3)
        )

        low = 0.3
        high = 0.7
        strong = response >= high
        weak = (response >= low) & ~strong
        connected = weak & (x >= 3.4) & (x <= 5.2)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.patch.set_facecolor("white")

        for ax in axes:
            ax.set_facecolor("white")
            ax.grid(True, linewidth=0.5, color="lightgray")
            ax.set_ylim(bottom=0, top=1.05)

            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[0].plot(x, response, color="black", label="Resposta do gradiente")
        axes[0].axhline(low, color="orange", linestyle="--", label="Limiar baixo")
        axes[0].axhline(high, color="red", linestyle="--", label="Limiar alto")
        axes[0].fill_between(x, 0, response, where=strong, color="red", alpha=0.25)
        axes[0].fill_between(x, 0, response, where=weak, color="orange", alpha=0.25)
        axes[0].set_ylabel("Magnitude", fontfamily="serif")
        axes[0].set_title("Canny: limiares de histerese", fontfamily="serif")
        axes[0].legend()

        axes[1].plot(x, strong, color="red", label="Borda forte")
        axes[1].plot(x, connected, color="blue", label="Borda fraca conectada")
        axes[1].plot(x, weak & ~connected, color="gray", label="Borda fraca descartada")
        axes[1].set_xlabel("Posição", fontfamily="serif")
        axes[1].set_ylabel("Classificação", fontfamily="serif")
        axes[1].set_yticks((0, 1), ("Descartado", "Aceito"))
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(path)
        return

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

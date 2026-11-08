"""
Save the images used in the presentation that do not depend on an input image.

Usage:
    python3 src/assets_static.py

The images are saved to the latex/assets/images/ folder in the project root, where
the presentation reads them:

    gaussian_2.png    Gaussian curve, sigma = 2
    gaussian_3.png    Gaussian curve, sigma = 3
    dog.png           Both gaussian curves and their difference
    hystheresis.png   Canny's hysteresis thresholds
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "latex" / "assets" / "images"

SIGMA_COLORS = {2: "red", 3: "blue"}


def style_axes(ax: plt.Axes) -> None:
    """Apply the style shared by all figures: white background, light grid, no spines."""
    ax.set_facecolor("white")
    ax.grid(True, linewidth=0.5, color="lightgray")

    for spine in ax.spines.values():
        spine.set_visible(False)


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate the zero-mean normal distribution with standard deviation sigma at x."""
    return np.exp(-(x ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def plot_hysteresis(path: Path) -> None:
    """Plot a synthetic gradient response and how Canny's hysteresis classifies it."""
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

    fig, (response_ax, class_ax) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, facecolor="white"
    )

    for ax in (response_ax, class_ax):
        style_axes(ax)
        ax.set_ylim(bottom=0, top=1.05)

    response_ax.plot(x, response, color="black", label="Resposta do gradiente")
    response_ax.axhline(low, color="orange", linestyle="--", label="Limiar baixo")
    response_ax.axhline(high, color="red", linestyle="--", label="Limiar alto")
    response_ax.fill_between(x, 0, response, where=strong, color="red", alpha=0.25)
    response_ax.fill_between(x, 0, response, where=weak, color="orange", alpha=0.25)
    response_ax.set_ylabel("Magnitude", fontfamily="serif")
    response_ax.set_title("Canny: limiares de histerese", fontfamily="serif")
    response_ax.legend()

    class_ax.plot(x, strong, color="red", label="Borda forte")
    class_ax.plot(x, connected, color="blue", label="Borda fraca conectada")
    class_ax.plot(x, weak & ~connected, color="gray", label="Borda fraca descartada")
    class_ax.set_xlabel("Posição", fontfamily="serif")
    class_ax.set_ylabel("Classificação", fontfamily="serif")
    class_ax.set_yticks((0, 1), ("Descartado", "Aceito"))
    class_ax.legend()

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_gaussian(sigmas: tuple[int, ...], path: Path) -> None:
    """Plot one or two gaussians. With two, also plot their difference (DoG)."""
    x = np.linspace(-10, 10, 10000)

    fig, ax = plt.subplots(facecolor="white")
    style_axes(ax)

    for sigma in sigmas:
        ax.plot(x, gaussian(x, sigma), color=SIGMA_COLORS[sigma], label=f"σ = {sigma}")

    if len(sigmas) == 2:
        smaller, larger = sorted(sigmas)
        ax.plot(
            x,
            gaussian(x, smaller) - gaussian(x, larger),
            color="purple",
            linestyle="--",
            label=f"σ = {smaller} − σ = {larger}",
        )

    ax.set_xlabel("x", fontfamily="serif")
    ax.set_ylabel("f(x)", fontfamily="serif")
    ax.set_ylim(top=0.25, bottom=-0.049)
    ax.set_xlim(left=-10, right=10)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend()

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_gaussian((2,), OUTPUT_DIR / "gaussian_2.png")
    plot_gaussian((3,), OUTPUT_DIR / "gaussian_3.png")
    plot_gaussian((2, 3), OUTPUT_DIR / "dog.png")
    plot_hysteresis(OUTPUT_DIR / "hystheresis.png")


if __name__ == "__main__":
    main()

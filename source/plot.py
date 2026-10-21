import torch
import matplotlib.pyplot as plt


def plot_rgb(image: torch.Tensor, block=True):
    """
    Plot the source image.
    """
    fig, ax = plt.subplots()

    ax.imshow(image.cpu().numpy())

    ax.axis("off")

    plt.show(block=block)

    
def plot_luminance(image: torch.Tensor, block=True):
    """
    Plot the source image.
    """
    fig, ax = plt.subplots()

    ax.imshow(image.cpu().numpy(), cmap="gray")

    ax.axis("off")

    plt.show(block=block)


def plot_rgb_histograms(image: torch.Tensor, block=True):
    """
    Plot histograms for each channel of an RGB image.
    Assumes image has shape (H, W, 3), with channels R, G, B.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    colors = ("Red", "Green", "Blue")

    for i, channel in enumerate(("R", "G", "B")):
        axes[i].hist(image[..., i].ravel(), bins=256, range=(0, 255), color=colors[i])
        axes[i].set_title(colors[i])
        axes[i].set_xlim(0, 255)

    plt.tight_layout()
    plt.show(block=block)


def plot_lab_histograms(image: torch.Tensor, block=True):
    """
    Plot histograms for each channel of a CIELAB image.
    Assumes image has shape (H, W, 3), with channels L*, a*, b*.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    channels = ("L*", "a*", "b*")
    colors = ("gray", "red", "blue")

    for i, channel in enumerate(channels):
        axes[i].hist(image[..., i].ravel().cpu().numpy(), bins=256, color=colors[i])
        axes[i].set_title(channel)

    plt.tight_layout()
    plt.show(block=block)


def plot_luminance_histogram(image: torch.Tensor, block=True):
    """
    Plot histogram for the luminance channel of a CIELAB image.
    Assumes image has shape (H, W, 1).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.hist(image[..., 0].ravel().cpu().numpy(), bins=256, color="gray")
    ax.set_title("L")

    plt.tight_layout()
    plt.show(block=block)


def plot_fft(image: torch.Tensor, block=True):
    spectrum = torch.fft.fft2(image.float())

    shifted = torch.fft.fftshift(spectrum)

    magnitude = torch.log1p(torch.abs(shifted))

    fig, ax = plt.subplots()

    ax.imshow(magnitude.cpu().numpy(), cmap="gray")
    ax.set_title("Transformada de Fourier 2D")
    ax.axis("off")

    plt.show(block=block)
import matplotlib.pyplot as plt

from math import pi
from util import *

def main():
    # image = sinusoidal_image(0.25 * pi, frequency=8, dimension=256)
    image = sinusoidal_image(0.0, frequency=8, dimension=256)

    spectrum = torch.abs(torch.fft.fftshift(torch.fft.fft2(image.float())))
    magnitude = torch.log1p(spectrum)

    fig, ax = plt.subplots()

    ax.imshow(image.cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")

    fig, ax = plt.subplots()
    
    ax.imshow(magnitude.cpu().numpy(), cmap="gray")
    ax.set_title("Transformada de Fourier 2D")
    ax.axis("off")

    plt.show(block=True)


if __name__ == "__main__":
    main()
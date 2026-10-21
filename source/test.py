from source.util import *
from source.plot import *

from math import pi

def main():
    # image = sinusoidal_image(0.25 * pi, frequency=8, dimension=256)
    image = sinusoidal_image(0.0, frequency=8, dimension=256)

    fig, ax = plt.subplots()

    ax.imshow(image.cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")

    plt.show(block=False)

    plot_fft(image)


if __name__ == "__main__":
    main()
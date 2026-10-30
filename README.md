# dog

A difference-of-gaussians pipeline pre-processing script.

## Getting started

```sh
py main.py [filename with extension]
```

Save the canonical Gaussian curve:

```sh
python3 src/figures.py gaussian [sigma1] [sigma2]
```

Sigmas default to 1. With two values, the figure also shows their difference
(larger sigma minus smaller sigma). The figure is saved as `gaussian.png` in
the current directory; use `-o path/to/file.png` to choose another location.

Apply a filter to an image:

```sh
py src/filter.py [filename with extension] gaussian [radius] [sigma]
py src/filter.py [filename with extension] dog [radius] [sigma1] [sigma2]
```

Append an output path to save only the filtered image, relative to the current
working directory:

```sh
py src/filter.py image.png gaussian 3 1.5 -L output/images/foo.png
```

By default, all filters process the RGB channels. Add `-L` to apply a filter
only to the Y (luminance) channel. For example:

```sh
py src/filter.py image.png gaussian 3 1.5 -L
```

Available filters are `sobel`, `canny`, `gaussian`, and `dog`. `gaussian`
takes a radius and sigma; `dog` takes a radius and two sigmas. The DoG sigmas
are automatically ordered from smaller to larger.

## Links

https://www.scenestill.com
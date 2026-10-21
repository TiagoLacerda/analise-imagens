import math
import torch

def sinusoidal_image(
    direction: float,
    frequency: float,
    dimension: int,
) -> torch.Tensor:    
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    dx = math.cos(direction)
    dy = math.sin(direction)

    y, x = torch.meshgrid(
        torch.arange(dimension, device=device, dtype=torch.float32),
        torch.arange(dimension, device=device, dtype=torch.float32),
        indexing="ij",
    )

    phase = 2 * math.pi * frequency * (dx * x + dy * y) / dimension

    return torch.sin(phase)



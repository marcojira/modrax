"""Utility functions."""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def fig_to_rgb_array(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Convert matplotlib figure to RGB array.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB array of shape (H, W, 3) with dtype uint8
    """
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()  # type: ignore
    width, height = fig.canvas.get_width_height()
    rgb_array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    plt.close(fig)
    return rgb_array

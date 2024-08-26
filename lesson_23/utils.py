import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

from kornia.utils import draw_line

def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=6, pad=0.5):
    """Рисует батч изображений горизонтально.
    Args:
        imgs: батч NumPy или PyTorch изображений, в RGB (H, W, 3) или grayscale (H, W) формате.
        titles: список названий для каждого изображения.
        cmaps: colormaps для grayscale изображений.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n
    figsize = (size * n, size * 3 / 4) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)
    
def plot_lines(lines, line_colors="orange", point_colors="cyan", ps=4, lw=2, indices=(0, 1)):
    """Рисует линии и конечные точки для заданных изображений.
    Args:
        lines: список ndarrays размера (N, 2, 2).
        colors: строка, или list of list of tuples (один для каждой ключевой точки).
        ps: размер ключевых точек в пикселях (float).
        lw: ширина линий в пикселях (float).
        indices: индексы изображения на которых нужно отрисовать совпадения.
    """
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(lines)
    if not isinstance(point_colors, list):
        point_colors = [point_colors] * len(lines)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, lc, pc in zip(axes, lines, line_colors, point_colors):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 1, 1], l[i, 0, 1]),
                (l[i, 1, 0], l[i, 0, 0]),
                zorder=1,
                c=lc,
                linewidth=lw,
            )
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 1], pts[:, 0], c=pc, s=ps, linewidths=0, zorder=2)


def plot_color_line_matches(lines, lw=2, indices=(0, 1)):
    """Рисует совпадение линий для заданных изображений и спользованием разных цветов.
    Args:
        lines: список ndarrays размера (N, 2, 2).
        lw: ширина линий в пикселях (float).
        indices: индексы изображения на которых нужно отрисовать совпадения.
    """
    n_lines = len(lines[0])

    cmap = plt.get_cmap("nipy_spectral", lut=n_lines)
    colors = np.array([mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)])

    np.random.shuffle(colors)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, l in zip(axes, lines):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 1, 1], l[i, 0, 1]),
                (l[i, 1, 0], l[i, 0, 0]),
                zorder=1,
                c=colors[i],
                linewidth=lw,
            )
            a.add_line(line)

def draw_polyline(img, points, color=torch.tensor([0, 1, 0])):
    out = img.clone()
    for i in range(len(points)):
        p1 = points[i]
        j = i + 1 if i < len(points) - 1 else 0
        p2 = points[j]
        out = draw_line(out, p1, p2, color)
    return out
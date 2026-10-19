"""Animate x_j = sum_i y_i w_ji and save a GIF beside this guide.

Run from the project root:
    python notes/01_1986_Backpropagation/Aimation-code/animate_weighted_sum.py
Requires matplotlib and Pillow. No GPU or external animation encoder needed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to a file without opening a GUI window.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch


def main():
    signals = [0.8, 0.4, 0.6]
    weights = [0.5, -0.5, 0.75]
    products = [y * w for y, w in zip(signals, weights)]
    totals = [0.0]
    for product in products:
        totals.append(totals[-1] + product)

    blue, red, ink = "#2463c4", "#c44332", "#182b49"
    fig = plt.figure(figsize=(10, 7), facecolor="#f5f8fc")
    fig.suptitle("How a neuron adds its weighted inputs", fontsize=21,
                 fontweight="bold", color=ink, y=0.97)
    fig.text(0.5, 0.85, r"$x_j = \sum_i y_i w_{ji}$", ha="center",
             fontsize=22, color=ink)
    ax = fig.add_axes([0.04, 0.36, 0.92, 0.46])
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    positions = [4, 2.5, 1]
    arrows, contribution_labels = [], []
    for i, (y, w, row) in enumerate(zip(signals, weights, positions)):
        color = red if w < 0 else blue
        arrow = FancyArrowPatch((2.1, row), (7.7, 2.5), arrowstyle="-|>",
                                mutation_scale=19, linewidth=2, color=color,
                                alpha=0.3)
        ax.add_patch(arrow)
        arrows.append(arrow)
        ax.add_patch(Circle((1.25, row), 0.72, facecolor="white",
                            edgecolor=blue, linewidth=2))
        ax.text(1.25, row, rf"$y_{i+1}={y:.1f}$", ha="center", va="center",
                fontsize=18, color=ink)
        label_y = [4.3, 2.92, 0.65][i]
        ax.text(4.6, label_y, rf"$w_{{j{i+1}}}={w:+.2f}$", ha="center",
                fontsize=17, color=color)
        label = ax.text(4.6, label_y - 0.4, "", ha="center", fontsize=15,
                        color=color, bbox=dict(facecolor="#f5f8fc", edgecolor="none"))
        contribution_labels.append(label)

    ax.add_patch(Circle((8.5, 2.5), 0.8, facecolor="white", edgecolor=ink,
                        linewidth=2))
    ax.text(8.5, 2.75, "unit j", ha="center", fontsize=18, color=ink)
    node_total = ax.text(8.5, 2.24, "", ha="center", fontsize=17, color=ink)
    packet, = ax.plot([], [], "o", markersize=12, zorder=10)
    status = fig.text(0.5, 0.32, "", ha="center", fontsize=18, color=ink)
    total_label = fig.text(0.5, 0.24, "", ha="center", fontsize=22,
                           color=ink, fontweight="bold")
    equation = fig.text(0.5, 0.16, "", ha="center", fontsize=18, color=ink)
    fig.text(0.5, 0.065, "Positive products add; negative products subtract.\n"
             "One connection at a time for teaching; real calculations can run in parallel.",
             ha="center", fontsize=12, color=ink)

    # Each phase moves a product along its connection, then holds the new sum.
    intro, phase, travel, ending = 15, 30, 19, 30
    frames = intro + 3 * phase + ending

    def update(frame):
        elapsed = frame - intro
        active = min(max(elapsed // phase, 0), 2)
        offset = elapsed % phase
        completed = 0 if elapsed < 0 else min(elapsed // phase, 3)
        if 0 <= elapsed < 3 * phase and offset >= travel:
            completed += 1
        total = totals[completed]
        for i, arrow in enumerate(arrows):
            arrow.set_alpha(1 if elapsed >= 0 and i <= active else 0.3)
            arrow.set_linewidth(4 if elapsed >= 0 and i == active else 2)
            contribution_labels[i].set_text(
                f"{signals[i]:.1f} x ({weights[i]:+.2f}) = {products[i]:+.2f}"
                if i < completed or (elapsed >= 0 and i == active) else "")
        packet.set_data([], [])
        if elapsed < 0:
            message = "Start with a total of zero."
        elif elapsed >= 3 * phase:
            message = "Weighted sum complete. Next comes the activation function."
        else:
            message = f"Connection {active+1}: multiply the signal by its weight."
            if offset < travel:
                t = offset / (travel - 1)
                packet.set_data([2.1 + (7.7 - 2.1) * t],
                                [positions[active] + (2.5 - positions[active]) * t])
                packet.set_color(red if products[active] < 0 else blue)
            else:
                message = f"Add {products[active]:+.2f} to the total."
        status.set_text(message)
        node_total.set_text(rf"$x_j={total:.2f}$")
        total_label.set_text(f"Running total: {total:.2f}")
        terms = " ".join(f"{p:+.2f}" for p in products[:completed])
        equation.set_text(f"{terms} = {total:.2f}" if terms else "No contributions added yet")

    animation = FuncAnimation(fig, update, frames=frames, interval=100, repeat=True)
    output = Path(__file__).resolve().parent.parent / "images" / "weighted_sum_animation.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

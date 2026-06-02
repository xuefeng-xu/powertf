import numpy as np
from scipy.special import boxcox
from scipy.stats import yeojohnson
from pathlib import Path
import matplotlib.pyplot as plt


def power_plot(x_min, x_max, power):
    if power == "BC":
        power_func = boxcox
    else:
        power_func = yeojohnson

    eps = 0.01
    x = np.arange(x_min, x_max, eps)

    if power == "BC":
        fig, ax = plt.subplots(figsize=(2.6, 3.3))
    else:
        fig, ax = plt.subplots(figsize=(3.2, 3.3))

    line_color = ["dodgerblue", "limegreen", "red", "mediumpurple", "orange"]
    for idx, lmb in enumerate([3, 2, 1, 0, -1]):
        y = power_func(x, lmb)
        ax.plot(x, y, label=rf"$\lambda$={lmb}", color=line_color[idx])

    ax.set_xlabel(r"$x$")
    if power == "BC":
        ax.set_ylabel(r"$\psi_{\text{BC}}(\lambda, x)$")
        ax.set_title("Box-Cox")
    else:
        ax.set_ylabel(r"$\psi_{\text{YJ}}(\lambda, x)$")
        ax.set_title("Yeo-Johnson")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-4, 4)

    ax.yaxis.set_ticks(list(np.arange(-4, 5, 2)))
    ax.get_yticklabels()[2].set(color="red")
    ax.xaxis.set_ticks(list(np.arange(x_min, x_max + 1)))
    if power == "BC":
        ax.get_xticklabels()[1].set(color="red")
    else:
        ax.get_xticklabels()[2].set(color="red")

    ax.axhline(0, linestyle="--", color="k")
    if power == "BC":
        ax.axvline(1, linestyle="--", color="k")
    else:
        ax.axvline(0, linestyle="--", color="k")

    ax.grid()
    ax.set_aspect(0.5)

    leg = ax.legend(loc="lower right")
    leg.get_texts()[2].set(color="red")

    PROJECT_ROOT = Path(__file__).parent.parent
    if power == "BC":
        img_path = PROJECT_ROOT / f"img/numerical/boxcox.pdf"
    else:
        img_path = PROJECT_ROOT / f"img/numerical/yeojohnson.pdf"

    fig.tight_layout()
    fig.savefig(img_path)


if __name__ == "__main__":
    power_plot(x_min=0, x_max=3, power="BC")
    power_plot(x_min=-2, x_max=2, power="YJ")
    plt.show()

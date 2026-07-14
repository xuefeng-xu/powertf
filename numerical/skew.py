import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from pathlib import Path
import matplotlib.pyplot as plt


if __name__ == "__main__":
    n = 10000
    bins = 60

    figsize = (6, 3)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    rng = np.random.RandomState(304)
    bc = PowerTransformer(method="box-cox")

    X_weibull_right = rng.weibull(a=1.5, size=n)

    ax[1, 0].hist(X_weibull_right, color="gray", bins=bins, density=True)
    ax[1, 0].set_title("Right-skewed data")

    X_weibull_right_bc, lmb_right = boxcox(X_weibull_right)
    lmb_right = round(lmb_right, 2)

    ax[1, 1].hist(X_weibull_right_bc, color="gray", bins=bins, density=True)
    ax[1, 1].set_title(rf"After Box-Cox ($\lambda$={lmb_right})")

    X_weibull_left = 2 * rng.weibull(a=20, size=n)

    ax[0, 0].hist(X_weibull_left, color="gray", bins=bins, density=True)
    ax[0, 0].set_title("Left-skewed data")

    X_weibull_left_bc, lmb_left = boxcox(X_weibull_left)
    lmb_left = round(lmb_left, 2)

    ax[0, 1].hist(X_weibull_left_bc, color="gray", bins=bins, density=True)
    ax[0, 1].set_title(rf"After Box-Cox ($\lambda$={lmb_left})")

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/numerical/skew.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)
    plt.show()

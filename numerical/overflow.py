# NOTE: Overflow warning in ExpSearch is expected,
# as this method does not address overflow issues
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from optimize import power_expsearch, boxcox_llf, boxcox_mle


def main(x, lb, ub, img_name):
    fig, ax = plt.subplots(figsize=(3, 3))

    lmbs = np.linspace(lb, ub, 400)
    nll = [-boxcox_llf(l, x) for l in lmbs]
    ax.plot(lmbs, nll, color="C0", label="NLL")

    lmb_opt = boxcox_mle(x)
    ax.axvline(lmb_opt, linestyle=":", color="k", label=r"$\lambda^*$")

    lmb_expsearch = power_expsearch("boxcox", x)
    ax.axvline(lmb_expsearch, linestyle="--", color="r", label="ExpSearch")

    ax.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\text{NLL}_\text{BC}$")

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/numerical/boxcox_expsearch_{img_name}.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)


if __name__ == "__main__":
    neg_data = [0.1, 0.1, 0.1, 0.101]
    main(neg_data, -550, -200, img_name="neg")

    pos_data = [10, 10, 10, 9.9]
    main(pos_data, 230, 500, img_name="pos")

    plt.show()

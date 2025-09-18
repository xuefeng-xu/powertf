import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from optimize.logcomp import boxcox_llf, yeojohnson_llf, boxcox_mle, yeojohnson_mle
from dataloader import load_data


def main(power, X, dataset, feature, lmbs):
    fig, ax = plt.subplots(figsize=(3, 3))

    x = X[feature]

    if power == "boxcox":
        lmb_log = boxcox_mle(x, var_comp="log")
        nll_log = [-boxcox_llf(l, x, "log") for l in lmbs]

        with np.errstate(all="ignore"):
            lmb_linear = boxcox_mle(x, var_comp="linear")
            nll_linear = [-boxcox_llf(l, x, "linear") for l in lmbs]

    elif power == "yeojohnson":
        lmb_log = yeojohnson_mle(x, var_comp="log")
        nll_log = [-yeojohnson_llf(l, x, "log") for l in lmbs]

        with np.errstate(all="ignore"):
            lmb_linear = yeojohnson_mle(x, var_comp="linear")
            nll_linear = [-yeojohnson_llf(l, x, "linear") for l in lmbs]

    for idx, nll in enumerate(nll_linear):
        if not np.isfinite(nll):
            ax.axvline(lmbs[idx], linestyle="-", color="blanchedalmond")

    ax.axvline(lmb_log, linestyle=":", color="k", label="log")
    ax.plot(lmbs, nll_log, linestyle="--", color="C0", label="log")

    ax.axvline(lmb_linear, linestyle="-.", color="C2", label="linear")
    ax.plot(lmbs, nll_linear, color="r", label="linear")

    h, l = ax.get_legend_handles_labels()
    h.append(mpatches.Patch(color="blanchedalmond", label="Overflow"))
    ax.legend(handles=h)

    ax.set_xlabel(r"$\lambda$")

    if power == "boxcox":
        ax.set_ylabel(r"$\text{NLL}_\text{BC}$")
    elif power == "yeojohnson":
        ax.set_ylabel(r"$\text{NLL}_\text{YJ}$")

    if power == "boxcox":
        title = f"{dataset.title()}: {feature} (BC)"
    elif power == "yeojohnson":
        title = f"{dataset.title()}: {feature} (YJ)"
    ax.set_title(title)

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/numerical/loglinear/{power}-{dataset}-{feature}.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    dataset = "blood"
    X, _ = load_data(dataset)

    lmbs = np.linspace(-20, 50, 400)
    main("yeojohnson", X, dataset, "Monetary", lmbs)

    dataset = "cancer"
    X, _ = load_data(dataset)

    lmbs = np.linspace(-20, 40, 400)
    main("yeojohnson", X, dataset, "ID", lmbs)

    lmbs = np.linspace(-20, 60, 400)
    main("yeojohnson", X, dataset, "area1", lmbs)

    lmbs = np.linspace(-30, 80, 400)
    main("yeojohnson", X, dataset, "area3", lmbs)

    dataset = "ecoli"
    X, _ = load_data(dataset)

    lmbs = np.linspace(-150, 30, 400)
    main("yeojohnson", X, dataset, "lip", lmbs)

    lmbs = np.linspace(-1300, 700, 400)
    main("yeojohnson", X, dataset, "chg", lmbs)

    dataset = "house"
    X, _ = load_data(dataset)

    lmbs = np.linspace(-280, 100, 400)
    main("boxcox", X, dataset, "YrSold", lmbs)

    lmbs = np.linspace(-100, 60, 400)
    main("yeojohnson", X, dataset, "YrSold", lmbs)

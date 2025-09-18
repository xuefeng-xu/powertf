import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from optimize import (
    power_expsearch,
    dnll_dlmb,
    boxcox_llf,
    yeojohnson_llf,
    boxcox_mle,
    yeojohnson_mle,
)
from dataloader import load_data


def plot_deriv(power, lmbs, x, true_deriv, ax):
    with np.errstate(all="ignore"):
        dnll = [dnll_dlmb(power, l, x, true_deriv) for l in lmbs]
    for idx, deriv in enumerate(dnll):
        if not np.isfinite(deriv):
            ax.axvline(lmbs[idx], linestyle="-", color="blanchedalmond")

    if true_deriv:
        label = r"$\partial\text{NLL} / \partial\lambda$"
    else:
        label = r"$n\sigma^2_{\psi} \cdot \partial\text{NLL} / \partial\lambda$"
    ax.plot(
        lmbs,
        dnll,
        color="C2",
        label=label,
    )
    ax.axhline(0, linestyle=":", color="k")

    with np.errstate(all="ignore"):
        lmb_exp = power_expsearch(power, x, true_deriv=true_deriv)
    ax.axvline(lmb_exp, linestyle="--", color="r", label="ExpSearch")

    h, l = ax.get_legend_handles_labels()
    h.append(mpatches.Patch(color="blanchedalmond", label="Overflow"))
    ax.legend(handles=h)

    ax.set_xlabel(r"$\lambda$")

    if power == "boxcox":
        if true_deriv:
            ax.set_ylabel(r"$\partial\text{NLL}_\text{BC} / \partial\lambda$")
        else:
            ax.set_ylabel(
                r"$n\sigma^2_{\psi_\text{BC}} \cdot \partial\text{NLL}_\text{BC} / \partial\lambda$"
            )

    elif power == "yeojohnson":
        if true_deriv:
            ax.set_ylabel(r"$\partial\text{NLL}_\text{YJ} / \partial\lambda$")
        else:
            ax.set_ylabel(
                r"$n\sigma^2_{\psi_\text{YJ}} \cdot \partial\text{NLL}_\text{YJ} / \partial\lambda$"
            )


def main(power, X, dataset, feature, lmbs_nll, lmbs_dnll):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    if power == "boxcox":
        title = f"{dataset.title()}: {feature} (Box-Cox)"
    elif power == "yeojohnson":
        title = f"{dataset.title()}: {feature} (Yeo-Johnson)"
    fig.suptitle(title)

    x = X[feature]

    if power == "boxcox":
        nll = [-boxcox_llf(l, x) for l in lmbs_nll]
        lmb_brent = boxcox_mle(x)

    elif power == "yeojohnson":
        nll = [-yeojohnson_llf(l, x) for l in lmbs_nll]
        lmb_brent = yeojohnson_mle(x)

    with np.errstate(all="ignore"):
        lmb_exp = power_expsearch(power, x)

    ax[0].plot(lmbs_nll, nll, color="C0", label="NLL")
    ax[0].axvline(lmb_brent, linestyle=":", color="k", label="Brent")
    ax[0].axvline(lmb_exp, linestyle="--", color="r", label="ExpSearch")

    ax[0].legend()
    ax[0].set_xlabel(r"$\lambda$")

    if power == "boxcox":
        ax[0].set_ylabel(r"$\text{NLL}_\text{BC}$")
    elif power == "yeojohnson":
        ax[0].set_ylabel(r"$\text{NLL}_\text{YJ}$")

    plot_deriv(power, lmbs_dnll, x, true_deriv=False, ax=ax[1])
    plot_deriv(power, lmbs_dnll, x, true_deriv=True, ax=ax[2])

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/numerical/brentexp/{power}-{dataset}-{feature}.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    dataset = "ecoli"
    X, _ = load_data(dataset)

    lmbs_nll = np.linspace(-130, -45, 400)
    lmbs_dnll = np.linspace(-55, -49, 400)
    main("yeojohnson", X, dataset, "lip", lmbs_nll, lmbs_dnll)

    lmbs_nll = np.linspace(-1400, 0, 400)
    lmbs_dnll = np.linspace(-55, -49, 400)
    main("yeojohnson", X, dataset, "chg", lmbs_nll, lmbs_dnll)

    dataset = "house"
    X, _ = load_data(dataset)

    lmbs_nll = np.linspace(30, 70, 400)
    # lmbs_dnll = np.linspace(40.53, 40.6, 400)
    main("boxcox", X, dataset, "YearRemodAdd", lmbs_nll, lmbs_nll)

    lmbs_nll = np.linspace(-110, 10, 400)
    lmbs_dnll = np.linspace(-3.2, -2.8, 400)
    main("boxcox", X, dataset, "YrSold", lmbs_nll, lmbs_dnll)

    lmbs_nll = np.linspace(300, 550, 400)
    lmbs_dnll = np.linspace(300, 530, 400)
    main("yeojohnson", X, dataset, "Street", lmbs_nll, lmbs_dnll)

    lmbs_nll = np.linspace(30, 70, 400)
    # lmbs_dnll = np.linspace(40.57, 40.6, 400)
    main("yeojohnson", X, dataset, "YearRemodAdd", lmbs_nll, lmbs_nll)

    lmbs_nll = np.linspace(-110, 10, 400)
    lmbs_dnll = np.linspace(-1, -0.75, 400)
    main("yeojohnson", X, dataset, "YrSold", lmbs_nll, lmbs_dnll)

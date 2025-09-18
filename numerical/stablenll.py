import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from pathlib import Path
from optimize.utils import _log_var


def boxcox_llf(lmb, x, remove_const=True, lambda_out=True):
    x = np.asarray(x)

    n = len(x)
    logx = np.log(x)

    # Compute the variance of the transformed data.
    if abs(lmb) < np.spacing(1.0):
        logvar = np.log(np.var(logx))

    elif remove_const and lambda_out:
        logbc = lmb * logx
        logvar = _log_var(logbc) - 2 * np.log(abs(lmb))

    elif remove_const and not lambda_out:
        logbc = lmb * logx - np.log(abs(lmb))
        logvar = _log_var(logbc)

    elif not remove_const and lambda_out:
        pij = np.full_like(logx, np.pi * 1j, dtype=np.complex128)
        logbc = logsumexp([lmb * logx, pij], axis=0)
        logvar = _log_var(logbc) - 2 * np.log(abs(lmb))

    elif not remove_const and not lambda_out:
        pij = np.full_like(logx, np.pi * 1j, dtype=np.complex128)
        logbc = logsumexp([lmb * logx, pij], axis=0) - np.log(abs(lmb))
        logvar = _log_var(logbc)

    return (lmb - 1) * np.sum(logx) - n / 2 * logvar


if __name__ == "__main__":
    figsize = (3, 3)

    x = [10, 10, 10, 9.9]

    # Compare: remove constant vs. keep constant
    lmbs1 = np.linspace(-16, -12, 200)

    nll_1const = [
        -boxcox_llf(l, x, remove_const=False, lambda_out=False) for l in lmbs1
    ]
    nll_0const = [-boxcox_llf(l, x, remove_const=True, lambda_out=True) for l in lmbs1]

    fig_const, ax_const = plt.subplots(figsize=figsize)
    ax_const.plot(lmbs1, nll_1const, "C1--", label="keep const")
    ax_const.plot(lmbs1, nll_0const, "C0", label="remove const")

    ax_const.legend()
    ax_const.set_xlabel(r"$\lambda$")
    ax_const.set_ylabel(r"$\text{NLL}_\text{BC}$")

    PROJECT_ROOT = Path(__file__).parent.parent
    img_const_path = PROJECT_ROOT / f"img/numerical/nll_boxcox_const.pdf"
    img_const_path.parent.mkdir(parents=True, exist_ok=True)
    fig_const.tight_layout()
    fig_const.savefig(img_const_path)

    # Compare: lmb out vs. lmb in
    lmbs2 = np.linspace(-1e-5, 1e-5, 200)

    nll_lmbin = [-boxcox_llf(l, x, remove_const=True, lambda_out=False) for l in lmbs2]
    nll_lmbout = [-boxcox_llf(l, x, remove_const=True, lambda_out=True) for l in lmbs2]

    fig_lmb, ax_lmb = plt.subplots(figsize=figsize)
    ax_lmb.plot(lmbs2, nll_lmbin, "C6--", label=r"$\lambda$ in")
    ax_lmb.plot(lmbs2, nll_lmbout, "C0", label=r"$\lambda$ out")
    ax_lmb.legend()
    ax_lmb.set_xlabel(r"$\lambda$")
    ax_lmb.set_ylabel(r"$\text{NLL}_\text{BC}$")

    img_lmb_path = PROJECT_ROOT / f"img/numerical/nll_boxcox_lmb.pdf"
    img_lmb_path.parent.mkdir(parents=True, exist_ok=True)
    fig_lmb.tight_layout()
    fig_lmb.savefig(img_lmb_path)

    plt.show()

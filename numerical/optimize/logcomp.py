import numpy as np
from scipy.stats import yeojohnson_llf as sp_yeojohnson_llf
from scipy.stats._morestats import _yeojohnson_transform
from scipy.special import log1p, boxcox, logsumexp, lambertw
from scipy.optimize import brent
from .utils import _log_var


def boxcox_llf(lmb, x, var_comp="log"):
    # Log-likelihood function for Box-Cox
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    if np.any(x <= 0):
        raise ValueError("x must be strictly positive.")

    n = x.shape[0]
    logx = np.log(x)

    # Compute the variance of the transformed data.
    if abs(lmb) < np.spacing(1.0):
        logvar = np.log(np.var(logx))

    elif var_comp == "log":
        logvar = _log_var(lmb * logx) - 2 * np.log(abs(lmb))

    elif var_comp == "linear":
        # Same as SciPy <= 1.11
        logvar = np.log(np.var(x**lmb / lmb))

    return (lmb - 1) * np.sum(logx) - n / 2 * logvar


def boxcox_mle(x, brack=(-2, 2), var_comp="log"):
    # Maximum Likelihood Estimation of optimal lmbda for Box-Cox
    def _neg_llf(lmb, x):
        return -boxcox_llf(lmb, x, var_comp)

    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    if np.any(x <= 0):
        raise ValueError("x must be strictly positive.")

    return brent(_neg_llf, brack=brack, args=(x,))


def boxcox_inv_lmbda(x, y):
    # Compute lmbda given x and y for Box-Cox
    num = lambertw(-(x ** (-1 / y)) * np.log(x) / y, k=-1)
    return np.real(-num / np.log(x) - 1 / y)


def boxcox_constranined_lmax(lmax, x, ymax):
    # Constrained lmax to ensure |boxcox(x)| <= ymax
    if ymax <= 0:
        raise ValueError("`ymax` must be strictly positive")

    xmin, xmax = min(x), max(x)

    # x > 1, boxcox(x) > 0; x < 1, boxcox(x) < 0
    if xmin >= 1:
        x_treme = xmax
    elif xmax <= 1:
        x_treme = xmin
    else:  # xmin < 1 < xmax
        indicator = boxcox(xmax, lmax) > abs(boxcox(xmin, lmax))
        x_treme = xmax if indicator else xmin

    if abs(boxcox(x_treme, lmax)) > ymax:
        lmax = boxcox_inv_lmbda(x_treme, ymax * np.sign(x_treme - 1))
    return lmax


def yeojohnson_llf(lmb, x, var_comp="log"):
    # Log-likelihood function for Yeo-Johnson
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    if var_comp == "log":
        pos = x >= 0  # binary mask

        if np.all(pos):
            if abs(lmb) < np.spacing(1.0):
                logvar = np.log(np.var(log1p(x)))
            else:
                logvar = _log_var(lmb * log1p(x)) - 2 * np.log(abs(lmb))

        elif np.all(~pos):
            if abs(lmb - 2) < np.spacing(1.0):
                logvar = np.log(np.var(log1p(-x)))
            else:
                logvar = _log_var((2 - lmb) * log1p(-x)) - 2 * np.log(abs(2 - lmb))

        else:  # mixed positive and negative data
            logyj = np.zeros_like(x, dtype=np.complex128)

            # x >= 0
            if abs(lmb) < np.spacing(1.0):
                with np.errstate(divide="ignore"):
                    logyj[pos] = np.log(log1p(x[pos]) + 0j)
            else:  # lmbda != 0
                logm1_pos = np.full_like(x[pos], np.pi * 1j, dtype=np.complex128)
                logyj[pos] = logsumexp(
                    [lmb * log1p(x[pos]), logm1_pos], axis=0
                ) - np.log(lmb + 0j)

            # x < 0
            if abs(lmb - 2) < np.spacing(1.0):
                logyj[~pos] = np.log(-log1p(-x[~pos]) + 0j)
            else:  # lmbda != 2
                logm1_neg = np.full_like(x[~pos], np.pi * 1j, dtype=np.complex128)
                logyj[~pos] = logsumexp(
                    [(2 - lmb) * log1p(-x[~pos]), logm1_neg], axis=0
                ) - np.log(lmb - 2 + 0j)

            logvar = _log_var(logyj)

        n = x.shape[0]
        ll = (lmb - 1) * np.sum(np.sign(x) * log1p(np.abs(x)))
        ll += -n / 2 * logvar

    elif var_comp == "linear":
        ll = sp_yeojohnson_llf(lmb, x)

    return ll


def yeojohnson_mle(x, brack=(-2, 2), var_comp="log"):
    # Maximum Likelihood Estimation of optimal lmbda for Yeo-Johnson
    def _neg_llf(lmb, x):
        return -yeojohnson_llf(lmb, x, var_comp)

    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    return brent(_neg_llf, brack=brack, args=(x,))


def yeojohnson_inv_lmbda(x, y):
    # Compute lmbda given x and y for Yeo-Johnson
    if x >= 0:
        num = lambertw(-((x + 1) ** (-1 / y) * log1p(x)) / y, k=-1)
        return np.real(-num / log1p(x)) - 1 / y
    else:
        num = lambertw(((1 - x) ** (1 / y) * log1p(-x)) / y, k=-1)
        return np.real(num / log1p(-x)) - 1 / y + 2


def yeojohnson_constranined_lmax(lmax, x, ymax):
    # Constrained lmax to ensure |yeojohnson(x)| <= ymax
    if ymax <= 0:
        raise ValueError("`ymax` must be strictly positive")

    xmin, xmax = min(x), max(x)

    # x > 0, yeojohnson(x) > 0; x < 0, yeojohnson(x) < 0
    if xmin >= 0:
        x_treme = xmax
    elif xmax <= 0:
        x_treme = xmin
    else:  # xmin < 0 < xmax
        with np.errstate(over="ignore"):
            indicator = _yeojohnson_transform(xmax, lmax) > abs(
                _yeojohnson_transform(xmin, lmax)
            )
        x_treme = xmax if indicator else xmin

    with np.errstate(over="ignore"):
        if abs(_yeojohnson_transform(x_treme, lmax)) > ymax:
            lmax = yeojohnson_inv_lmbda(x_treme, ymax * np.sign(x_treme))
    return lmax

# Reference:
# SecureFedYJ: a safe feature Gaussianization protocol for Federated Learning
# https://arxiv.org/pdf/2210.01639.pdf

import numpy as np
from scipy.special import boxcox, log1p
from scipy.stats._morestats import _yeojohnson_transform
from scipy.optimize._zeros_py import _iter, _xtol, _rtol


def dboxcox_dlmb(lmb, x):
    # First-order derivative of Box-Cox with respect to lmb
    if abs(lmb) < np.spacing(1.0):
        return np.power(np.log(x), 2) / 2
    else:  # lmb != 0
        return (np.exp(lmb * np.log(x)) * np.log(x) - boxcox(x, lmb)) / lmb


def dyeojohnson_dlmb(lmb, x):
    # First-order derivative of Yeo-Johnson with respect to lmb
    out = np.zeros_like(x, dtype=np.float64)
    pos = x >= 0  # binary mask

    # x >= 0
    if abs(lmb) < np.spacing(1.0):
        out[pos] = np.power(log1p(x[pos]), 2) / 2
    else:  # lmb != 0
        out[pos] = (
            np.exp(lmb * log1p(x[pos])) * log1p(x[pos])
            - _yeojohnson_transform(x[pos], lmb)
        ) / lmb

    # x < 0
    if abs(lmb - 2) < np.spacing(1.0):
        out[~pos] = np.power(log1p(-x[~pos]), 2) / 2
    else:  # lmb != 2
        out[~pos] = (
            np.exp((2 - lmb) * log1p(-x[~pos])) * log1p(-x[~pos])
            + _yeojohnson_transform(x[~pos], lmb)
        ) / (2 - lmb)

    return out


def dnll_dlmb(power, lmb, x, true_deriv=True):
    # First-order derivative of NLL with respect to lmb
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)

    if power == "boxcox":
        sum_cons = np.sum(np.log(x))
        y = boxcox(x, lmb)
        dy_dlmb = dboxcox_dlmb(lmb, x)

    elif power == "yeojohnson":
        sum_cons = np.sum(np.sign(x) * log1p(np.abs(x)))
        y = _yeojohnson_transform(x, lmb)
        dy_dlmb = dyeojohnson_dlmb(lmb, x)

    if true_deriv:
        # True derivative
        return (
            1 / np.var(y) * (np.sum(y * dy_dlmb) - 1 / n * np.sum(y) * np.sum(dy_dlmb))
            - sum_cons
        )
    else:
        # Formula used in the SecureFedYJ paper
        return (
            n * np.sum(y * dy_dlmb)
            - np.sum(y) * np.sum(dy_dlmb)
            - sum_cons * n * np.var(y)
        )


def exp_update(lmb, lmb_pos, lmb_neg, delta: int):
    if delta not in [1, -1]:
        raise ValueError("delta must be either 1 or -1")

    if delta == -1:
        lmb_neg = lmb
        if lmb_pos < np.inf:
            lmb = (lmb_pos + lmb) / 2
        else:
            lmb = max(2 * lmb, 1)
    else:  # delta == 1
        lmb_pos = lmb
        if lmb_neg > -np.inf:
            lmb = (lmb_neg + lmb) / 2
        else:
            lmb = min(2 * lmb, -1)

    return lmb, lmb_pos, lmb_neg


def power_expsearch(power, x, xtol=_xtol, rtol=_rtol, maxiter=_iter, true_deriv=False):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    lmb = 0
    lmb_pos = np.inf
    lmb_neg = -np.inf

    for i in range(maxiter):

        delta = np.sign(dnll_dlmb(power, lmb, x, true_deriv))

        if (
            delta == 0
            or np.isnan(delta)
            or abs((lmb_pos - lmb_neg) / 2) < xtol + rtol * abs(lmb)
            # similar to scipy.optimize.bisect
        ):
            break

        lmb, lmb_pos, lmb_neg = exp_update(lmb, lmb_pos, lmb_neg, delta)

    return lmb

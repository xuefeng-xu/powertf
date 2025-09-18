import numpy as np
from scipy.special import logsumexp, log1p


class FedPowerClient:

    def __init__(self, power, x):
        self.power = power

        x = np.asarray(x, np.float64)
        x = x[~np.isnan(x)]

        if self.power == "boxcox" and np.any(x <= 0):
            raise ValueError("x must be strictly positive for boxcox")
        self.x = x

        # Cache n and the constant term
        self.n = len(x)
        if self.power == "yeojohnson":
            pos = x >= 0
            self.n_pos = np.sum(pos)
            self.n_neg = self.n - self.n_pos

        if self.power == "boxcox":
            self.c = np.sum(np.log(x))
        elif self.power == "yeojohnson":
            self.c = np.sum(np.sign(x) * log1p(np.abs(x)))

    def log_boxcox(self, lmb):
        logx = np.log(self.x)

        if abs(lmb) < np.spacing(1.0):
            with np.errstate(divide="ignore"):
                logbc = np.log(logx + 0j)
        else:  # lmb != 0
            # - np.log(abs(lmb)) is computed at server side
            logbc = lmb * logx

        return logbc

    def log_yeojohnson(self, lmb):
        if self.n_neg == 0:  # all positive
            if abs(lmb) < np.spacing(1.0):
                with np.errstate(divide="ignore"):
                    logyj = np.log(log1p(self.x) + 0j)
            else:  # lmb != 0
                logyj = lmb * log1p(self.x)

        elif self.n_pos == 0:  # all negative
            if abs(lmb - 2) < np.spacing(1.0):
                logyj = np.log(-log1p(-self.x) + 0j)
            else:  # lmb != 2
                logyj = (2 - lmb) * log1p(-self.x)

        else:  # mixed positive and negative
            logyj = np.zeros_like(self.x, dtype=np.complex128)
            pos = self.x >= 0  # binary mask

            # x >= 0
            if abs(lmb) < np.spacing(1.0):
                with np.errstate(divide="ignore"):
                    logyj[pos] = np.log(log1p(self.x[pos]) + 0j)
            else:  # lmbda != 0
                logm1_pos = np.full_like(self.x[pos], np.pi * 1j, dtype=np.complex128)
                logyj[pos] = logsumexp(
                    [lmb * log1p(self.x[pos]), logm1_pos], axis=0
                ) - np.log(lmb + 0j)

            # x < 0
            if abs(lmb - 2) < np.spacing(1.0):
                logyj[~pos] = np.log(-log1p(-self.x[~pos]) + 0j)
            else:  # lmbda != 2
                logm1_neg = np.full_like(self.x[~pos], np.pi * 1j, dtype=np.complex128)
                logyj[~pos] = logsumexp(
                    [(2 - lmb) * log1p(-self.x[~pos]), logm1_neg], axis=0
                ) - np.log(lmb - 2 + 0j)

        return logyj

    def naive_variance(self, logx):
        # log of sum(x) and log of sum(x^2)
        logsum = logsumexp(logx)
        with np.errstate(invalid="ignore"):
            logsumsq = np.real(logsumexp(2 * logx))
        return logsum, logsumsq

    def pairwise_variance(self, logx):
        # log of mean(x) and log of sum of (x-mean(x))^2
        # use two-pass method at client side
        logmean = logsumexp(logx) - np.log(self.n)
        pij = np.full_like(logx, np.pi * 1j, dtype=np.complex128)
        logxmu = logsumexp([logx, logmean + pij], axis=0)
        logM2 = np.real(logsumexp(2 * logxmu, axis=0))
        return logmean, logM2

    def _llf(self, lmb, var_comp):
        if self.n == 0:
            return -np.inf, -np.inf

        if self.power == "boxcox":
            logpsi = self.log_boxcox(lmb)
        elif self.power == "yeojohnson":
            logpsi = self.log_yeojohnson(lmb)

        if var_comp == "pairwise":
            logmean, logM2 = self.pairwise_variance(logpsi)
            return logmean, logM2
        elif var_comp == "naive":
            logsum, logsumsq = self.naive_variance(logpsi)
            return logsum, logsumsq

    def llf(self, lmb, var_comp="pairwise"):
        lmb_arr = np.atleast_1d(lmb)
        if lmb_arr.ndim > 1:
            raise ValueError("lmb must be a scalar or 1D array")

        item1, item2 = [], []
        for l in lmb_arr:
            i1, i2 = self._llf(l, var_comp)
            item1.append(i1)
            item2.append(i2)

        item1 = np.asarray(item1)
        item2 = np.asarray(item2)

        if np.isscalar(lmb):
            item1, item2 = item1[0], item2[0]

        if self.power == "boxcox":
            return self.c, self.n, item1, item2
        elif self.power == "yeojohnson":
            return self.c, self.n_pos, self.n_neg, item1, item2

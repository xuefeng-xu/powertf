import numpy as np
from scipy.special import logsumexp


def _log_mean(logx):
    # compute log of mean of x from log(x)
    return logsumexp(logx, axis=0) - np.log(len(logx))


def _log_var(logx):
    # compute log of variance of x from log(x)
    logmean = _log_mean(logx)
    pij = np.full_like(logx, np.pi * 1j, dtype=np.complex128)
    logxmu = logsumexp([logx, logmean + pij], axis=0)
    return np.real(logsumexp(2 * logxmu, axis=0)) - np.log(len(logx))

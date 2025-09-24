import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brent
from .grid import gridsearch


class FedPowerServer:

    def __init__(self, power, clients):
        self.power = power
        self.clients = clients

    def naive_variance(self, lmb):
        def _naive_var(logsum, logsumsq, n):
            # Compute naive variance from logsum and logsumsq
            logmean = logsumexp(logsum, axis=0) - np.log(n)
            logmean_sq = logsumexp(logsumsq, axis=0) - np.log(n)
            # var = mean_sq - mean^2
            logvar = np.real(logsumexp([logmean_sq, 2 * logmean + np.pi * 1j], axis=0))
            return logvar

        if self.power == "boxcox":
            c, n = 0, 0
            logsum, logsumsq = [], []

            for client in self.clients:
                c_i, n_i, logsum_i, logsumsq_i = client.llf(lmb, var_comp="naive")

                if n_i == 0:
                    continue

                c += c_i
                n += n_i

                logsum.append(logsum_i)
                logsumsq.append(logsumsq_i)

            logvar = _naive_var(logsum, logsumsq, n)

            neq_0 = abs(lmb) >= np.spacing(1.0)
            logvar[neq_0] -= 2 * np.log(abs(lmb[neq_0]))

        elif self.power == "yeojohnson":
            c, n_pos, n_neg, n_mix = 0, 0, 0, 0
            logsum_pos, logsumsq_pos = [], []
            logsum_neg, logsumsq_neg = [], []
            logsum_mix, logsumsq_mix = [], []

            for client in self.clients:
                (
                    c_i,
                    n_pos_i,
                    n_neg_i,
                    logsum_i,
                    logsumsq_i,
                ) = client.llf(lmb, var_comp="naive")

                if n_pos_i + n_neg_i == 0:
                    continue

                c += c_i

                if n_neg_i == 0:
                    n_pos += n_pos_i
                    logsum_pos.append(logsum_i)
                    logsumsq_pos.append(logsumsq_i)
                elif n_pos_i == 0:
                    n_neg += n_neg_i
                    logsum_neg.append(logsum_i)
                    logsumsq_neg.append(logsumsq_i)
                else:
                    n_mix += n_pos_i + n_neg_i
                    logsum_mix.append(logsum_i)
                    logsumsq_mix.append(logsumsq_i)

            if n_neg == 0 and n_mix == 0:  # all positive
                logvar = _naive_var(logsum_pos, logsumsq_pos, n_pos)

                neq_0 = abs(lmb) >= np.spacing(1.0)
                logvar[neq_0] -= 2 * np.log(abs(lmb[neq_0]))

            elif n_pos == 0 and n_mix == 0:  # all negative
                logvar = _naive_var(logsum_neg, logsumsq_neg, n_neg)

                neq_2 = abs(lmb - 2) >= np.spacing(1.0)
                logvar[neq_2] -= 2 * np.log(abs(2 - lmb[neq_2]))

            else:  # mixed positive and negative
                if n_pos > 0:
                    logsum_pos = logsumexp(logsum_pos, axis=0).astype(np.complex128)
                    logsumsq_pos = logsumexp(logsumsq_pos, axis=0).astype(np.complex128)

                    neq_0 = abs(lmb) >= np.spacing(1.0)

                    # (((x+1)^lmb - 1) / lmb)^2
                    # = ((x+1)^(2*lmb) - 2*(x+1)^lmb + 1) / lmb^2
                    logsumsq_pos[neq_0] = logsumexp(
                        [
                            logsumsq_pos[neq_0],
                            logsum_pos[neq_0] + np.log(2) + np.pi * 1j,
                            np.full_like(logsum_pos[neq_0], np.log(n_pos)),
                        ],
                        axis=0,
                    ) - 2 * np.log(abs(lmb[neq_0]))
                    logsum_pos[neq_0] = logsumexp(
                        [
                            logsum_pos[neq_0],
                            np.log(n_pos)
                            + np.full_like(
                                logsum_pos[neq_0], np.pi * 1j, dtype=np.complex128
                            ),
                        ],
                        axis=0,
                    ) - np.log(lmb[neq_0] + 0j)

                if n_neg > 0:
                    logsum_neg = logsumexp(logsum_neg, axis=0).astype(np.complex128)
                    logsumsq_neg = logsumexp(logsumsq_neg, axis=0).astype(np.complex128)

                    neq_2 = abs(lmb - 2) >= np.spacing(1.0)

                    # (((-x+1)^(2-lmb) - 1) / (lmb-2))^2
                    # = ((-x+1)^(2*(2-lmb)) - 2*(-x+1)^(2-lmb) + 1) / (lmb-2)^2
                    logsumsq_neg[neq_2] = logsumexp(
                        [
                            logsumsq_neg[neq_2],
                            logsum_neg[neq_2] + np.log(2) + np.pi * 1j,
                            np.full_like(logsum_neg[neq_2], np.log(n_neg)),
                        ],
                        axis=0,
                    ) - 2 * np.log(abs(lmb[neq_2] - 2))
                    logsum_neg[neq_2] = logsumexp(
                        [
                            logsum_neg[neq_2],
                            np.log(n_neg)
                            + np.full_like(
                                logsum_neg[neq_2], np.pi * 1j, dtype=np.complex128
                            ),
                        ],
                        axis=0,
                    ) - np.log(lmb[neq_2] - 2 + 0j)

                if n_mix > 0:
                    logsum_mix = logsumexp(logsum_mix, axis=0)
                    logsumsq_mix = logsumexp(logsumsq_mix, axis=0)

                logsum, logsumsq = [], []
                if n_pos > 0:
                    logsum.append(logsum_pos)
                    logsumsq.append(logsumsq_pos)
                if n_neg > 0:
                    logsum.append(logsum_neg)
                    logsumsq.append(logsumsq_neg)
                if n_mix > 0:
                    logsum.append(logsum_mix)
                    logsumsq.append(logsumsq_mix)

                logvar = _naive_var(logsum, logsumsq, n_pos + n_neg + n_mix)

            n = n_pos + n_neg + n_mix

        return c, n, logvar

    def pairwise_variance(self, lmb):
        def _pairwise_var(triplet):
            while len(triplet) >= 2:
                n_1, logmean_1, logM2_1 = triplet.pop(0)
                n_2, logmean_2, logM2_2 = triplet.pop(0)

                n = n_1 + n_2
                logdelta = logsumexp([logmean_2, logmean_1 + np.pi * 1j], axis=0)
                logmean = logsumexp([logmean_1, logdelta + np.log(n_2 / n)], axis=0)
                logM2 = logsumexp(
                    [logM2_1, logM2_2, 2 * logdelta + np.log(n_1 * n_2 / n)], axis=0
                )

                triplet.append((n, logmean, logM2))

            n, logmean, logM2 = triplet[0]
            return n, logmean, np.real(logM2)

        if self.power == "boxcox":
            c = 0
            triplet = []

            for client in self.clients:
                c_i, n_i, logmean_i, logM2_i = client.llf(lmb, var_comp="pairwise")

                if n_i == 0:
                    continue

                c += c_i

                triplet.append((n_i, logmean_i, logM2_i))

            n, _, logM2 = _pairwise_var(triplet)

            neq_0 = abs(lmb) >= np.spacing(1.0)
            logM2[neq_0] -= 2 * np.log(abs(lmb[neq_0]))

            logvar = logM2 - np.log(n)

        elif self.power == "yeojohnson":
            c = 0
            triplet_pos, triplet_neg, triplet_mix = [], [], []

            for client in self.clients:
                (
                    c_i,
                    n_pos_i,
                    n_neg_i,
                    logmean_i,
                    logM2_i,
                ) = client.llf(lmb, var_comp="pairwise")

                if n_pos_i + n_neg_i == 0:
                    continue

                c += c_i

                if n_neg_i == 0:
                    triplet_pos.append((n_pos_i, logmean_i, logM2_i))
                elif n_pos_i == 0:
                    triplet_neg.append((n_neg_i, logmean_i, logM2_i))
                else:
                    triplet_mix.append((n_pos_i + n_neg_i, logmean_i, logM2_i))

            n_pos = 0
            if len(triplet_pos) > 0:
                n_pos, logmean_pos, logM2_pos = _pairwise_var(triplet_pos)

                neq_0 = abs(lmb) >= np.spacing(1.0)
                logM2_pos[neq_0] -= 2 * np.log(abs(lmb[neq_0]))

            n_neg = 0
            if len(triplet_neg) > 0:
                n_neg, logmean_neg, logM2_neg = _pairwise_var(triplet_neg)

                neq_2 = abs(lmb - 2) >= np.spacing(1.0)
                logM2_neg[neq_2] -= 2 * np.log(abs(2 - lmb[neq_2]))

            n_mix = 0
            if len(triplet_mix) > 0:
                n_mix, logmean_mix, logM2_mix = _pairwise_var(triplet_mix)

            if n_pos > 0 and (n_neg > 0 or n_mix > 0):
                logmean_pos = logmean_pos.astype(np.complex128)
                logmean_pos[neq_0] = logsumexp(
                    [
                        logmean_pos[neq_0],
                        np.full_like(
                            logmean_pos[neq_0], np.pi * 1j, dtype=np.complex128
                        ),
                    ],
                    axis=0,
                ) - np.log(lmb[neq_0] + 0j)

            if n_neg > 0 and (n_pos > 0 or n_mix > 0):
                logmean_neg = logmean_neg.astype(np.complex128)
                logmean_neg[neq_2] = logsumexp(
                    [
                        logmean_neg[neq_2],
                        np.full_like(
                            logmean_neg[neq_2], np.pi * 1j, dtype=np.complex128
                        ),
                    ],
                    axis=0,
                ) - np.log(lmb[neq_2] - 2 + 0j)

            triplet = []
            if n_pos > 0:
                triplet.append((n_pos, logmean_pos, logM2_pos))
            if n_neg > 0:
                triplet.append((n_neg, logmean_neg, logM2_neg))
            if n_mix > 0:
                triplet.append((n_mix, logmean_mix, logM2_mix))

            n, _, logM2 = _pairwise_var(triplet)
            logvar = logM2 - np.log(n)

        return c, n, logvar

    def aggregate(self, lmb, var_comp="pairwise"):
        if var_comp == "pairwise":
            return self.pairwise_variance(lmb)
        elif var_comp == "naive":
            return self.naive_variance(lmb)

    def llf(self, lmb, var_comp="pairwise"):
        lmb_arr = np.atleast_1d(lmb)
        if lmb_arr.ndim > 1:
            raise ValueError("lmb must be a scalar or 1D array")

        c, n, logvar = self.aggregate(lmb_arr, var_comp)
        ll = (lmb_arr - 1) * c - n / 2 * logvar

        if np.isscalar(lmb):
            ll = ll[0]

        return ll

    def mle(
        self,
        brack=(-2, 2),
        var_comp="pairwise",
        optimize="brent",
        n_points=20,
        full_output=0,
    ):
        def _neg_llf(lmb):
            return -self.llf(lmb, var_comp)

        if optimize == "brent":
            return brent(_neg_llf, brack=brack, full_output=full_output)
        elif optimize == "grid":
            return gridsearch(
                _neg_llf, brack=brack, n_points=n_points, full_output=full_output
            )

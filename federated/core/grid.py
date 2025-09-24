import numpy as np
from scipy.optimize import bracket
from scipy.optimize._zeros_py import _iter, _xtol, _rtol


def gridsearch(
    func,
    args=(),
    n_points=20,
    brack=None,
    xtol=_xtol,
    rtol=_rtol,
    maxiter=_iter,
    full_output=0,
):
    if brack is None:
        xa, xb, xc, fa, fb, fc, n_rounds = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, n_rounds = bracket(
            func, xa=brack[0], xb=brack[1], args=args
        )
    elif len(brack) == 3:
        xa, xb, xc = brack
        if xa > xc:  # swap so xa < xc can be assumed
            xc, xa = xa, xc
        if not ((xa < xb) and (xb < xc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not"
                " fulfill this requirement: (xa < xb) and (xb < xc)"
            )
        fa, fb, fc = func(np.asarray(brack), *args)
        if not ((fb < fa) and (fb < fc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not fulfill"
                " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
            )
        n_rounds = 1
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")

    xl, fl = xa, fa
    xm, fm = xb, fb
    xr, fr = xc, fc

    for i in range(maxiter):
        grid = np.linspace(xl, xr, n_points + 2)
        fgrid = func(grid[1:-1], *args)  # exclude the boundaries
        fgrid = np.r_[fl, fgrid, fr]  # include the boundaries

        if xm not in grid:
            j = np.searchsorted(grid, xm)
            grid = np.insert(grid, j, xm)
            fgrid = np.insert(fgrid, j, fm)

        min_idx = np.argmin(fgrid)
        xm = grid[min_idx]
        fm = fgrid[min_idx]

        if min_idx == 0:
            xl, fl = grid[0], fgrid[0]
            xr, fr = grid[1], fgrid[1]
        elif min_idx == len(grid) - 1:
            xl, fl = grid[-2], fgrid[-2]
            xr, fr = grid[-1], fgrid[-1]
        else:
            xl, fl = grid[min_idx - 1], fgrid[min_idx - 1]
            xr, fr = grid[min_idx + 1], fgrid[min_idx + 1]

        n_rounds += 1

        # similar to scipy.optimize.bisect
        if abs((xr - xl) / 2) < xtol + rtol * abs(xm):
            break

    if full_output:
        return xm, fm, n_rounds
    return xm

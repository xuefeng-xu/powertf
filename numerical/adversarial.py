# https://en.wikipedia.org/wiki/IEEE_754
# binary32  | Single    precision | maxval 3.40e38
# binary64  | Double    precision | maxval 1.80e308
# binary128 | Quadruple precision | maxval 1.19e4932
# binary256 | Octuple   precision | maxval 1.61e78913

from mpmath import mp, nstr
from mpsci.stats import (
    boxcox,
    boxcox_mle,
    boxcox_llf,
    yeojohnson,
    yeojohnson_mle,
    yeojohnson_llf,
)


if __name__ == "__main__":
    mp.dps = 100

    data = {
        "Single": {
            "Box-Cox": {
                "Negative": [0.1] * 3 + [0.1 + 9e-3],
                "Positive": [10] * 3 + [10 - 8e-1],
            },
            "Yeo-Johnson": {
                "Negative": [-10] * 3 + [-10 + 9e-1],
                "Positive": [10] * 3 + [10 - 9e-1],
            },
        },
        "Double": {
            "Box-Cox": {
                "Negative": [0.1] * 3 + [0.1 + 1e-3],
                "Positive": [10] * 3 + [10 - 1e-1],
            },
            "Yeo-Johnson": {
                "Negative": [-10] * 3 + [-10 + 1e-1],
                "Positive": [10] * 3 + [10 - 1e-1],
            },
        },
        "Quadruple": {
            "Box-Cox": {
                "Negative": [0.1] * 3 + [0.1 + 1e-5],
                "Positive": [10] * 3 + [10 - 1e-3],
            },
            "Yeo-Johnson": {
                "Negative": [-10] * 3 + [-10 + 1e-3],
                "Positive": [10] * 3 + [10 - 1e-3],
            },
        },
        "Octuple": {
            "Box-Cox": {
                "Negative": [0.1] * 3 + [0.1 + 1e-6],
                "Positive": [10] * 3 + [10 - 1e-4],
            },
            "Yeo-Johnson": {
                "Negative": [-10] * 3 + [-10 + 1e-4],
                "Positive": [10] * 3 + [10 - 1e-4],
            },
        },
    }

    for precision, precision_data in data.items():
        for power, power_data in precision_data.items():
            for overflow, x in power_data.items():
                print(f"{precision} Precision, {power}, {overflow} Overflow")
                print(f"Data: {x}")

                if power == "Box-Cox":
                    lmb = boxcox_mle(x)
                else:
                    lmb = yeojohnson_mle(x)
                print(f"Optimal lambda: {nstr(lmb)}")

                if overflow == "Negative":
                    xtreme = min(x)
                else:
                    xtreme = max(x)

                if power == "Box-Cox":
                    ytreme = boxcox(xtreme, lmb)
                    nll = boxcox_llf(lmb, x)
                else:
                    ytreme = yeojohnson(xtreme, lmb)
                    nll = yeojohnson_llf(lmb, x)

                print(f"NLL: {nstr(nll)}")
                print(f"Extreme value: {nstr(ytreme)}")
                print("-" * 52)

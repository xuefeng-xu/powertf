# https://en.wikipedia.org/wiki/IEEE_754
# binary64  | Double    precision | maxval 1.80e308
# binary128 | Quadruple precision | maxval 1.19e4932
# binary256 | Octuple   precision | maxval 1.61e78913

from mpmath import mp, nstr
from mpsci.stats import boxcox, boxcox_mle, yeojohnson, yeojohnson_mle


if __name__ == "__main__":
    mp.dps = 100

    data = {
        "Negative": {
            "Double": {
                "Box-Cox": [0.1] * 3 + [0.1 + 1e-3],
                "Yeo-Johnson": [-10] * 3 + [-10 + 1e-1],
            },
            "Quadruple": {
                "Box-Cox": [0.1] * 3 + [0.1 + 1e-5],
                "Yeo-Johnson": [-10] * 3 + [-10 + 1e-3],
            },
            "Octuple": {
                "Box-Cox": [0.1] * 3 + [0.1 + 1e-6],
                "Yeo-Johnson": [-10] * 3 + [-10 + 1e-4],
            },
        },
        "Positive": {
            "Double": {
                "Box-Cox": [10] * 3 + [10 - 1e-1],
                "Yeo-Johnson": [10] * 3 + [10 - 1e-1],
            },
            "Quadruple": {
                "Box-Cox": [10] * 3 + [10 - 1e-3],
                "Yeo-Johnson": [10] * 3 + [10 - 1e-3],
            },
            "Octuple": {
                "Box-Cox": [10] * 3 + [10 - 1e-4],
                "Yeo-Johnson": [10] * 3 + [10 - 1e-4],
            },
        },
    }

    for overflow, overflow_data in data.items():
        for precision, precision_data in overflow_data.items():
            for power, x in precision_data.items():
                print(f"{overflow} overflow, {precision} precision, {power}")
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
                else:
                    ytreme = yeojohnson(xtreme, lmb)
                print(f"Extreme value: {nstr(ytreme)}")
                print("-" * 50)

import timeit
import numpy as np
from numerical.optimize import boxcox_llf


def format_time(seconds, precision=2):
    if seconds < 1e-6:  # nanoseconds
        return f"{seconds * 1e9:.{precision}f} ns"
    elif seconds < 1e-3:  # microseconds
        return f"{seconds * 1e6:.{precision}f} μs"
    elif seconds < 1:  # milliseconds
        return f"{seconds * 1e3:.{precision}f} ms"
    else:  # seconds
        return f"{seconds:.{precision}f} s"


if __name__ == "__main__":
    repeat = 5
    number = 100
    rng = np.random.default_rng(123)
    lmb = 0.5  # arbitrary lambda value

    for size in [1000, 10000, 100000, 1000000]:
        x = rng.random(size=size)

        # Repeat timing 5 times, each with 1000 executions
        linear_times = timeit.repeat(
            'boxcox_llf(lmb, x, var_comp="linear")',
            globals=globals(),
            repeat=repeat,
            number=number,
        )

        log_times = timeit.repeat(
            'boxcox_llf(lmb, x, var_comp="log")',
            globals=globals(),
            repeat=repeat,
            number=number,
        )

        # Calculate mean and std per single execution
        linear_mean = np.mean(linear_times) / number
        linear_std = np.std(linear_times) / number

        log_mean = np.mean(log_times) / number
        log_std = np.std(log_times) / number
        print(f"Data size: {size}")
        print(f"Naive: {format_time(linear_mean)} ± {format_time(linear_std)}")
        print(f"Ours : {format_time(log_mean)} ± {format_time(log_std)}\n")

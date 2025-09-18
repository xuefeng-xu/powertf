from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from federated.core import FedPowerClient, FedPowerServer, IID_partitioner


def main(lmbs, x, power, n_clients, rng):
    x_clients = IID_partitioner(x, n_clients, rng)

    clients = [FedPowerClient(power, x) for x in x_clients]
    server = FedPowerServer(power, clients=clients)

    nll_naive = -server.llf(lmbs, var_comp="naive")
    nll_pairwise = -server.llf(lmbs, var_comp="pairwise")

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].plot(lmbs, nll_naive, "C8")
    ax[1].plot(lmbs, nll_pairwise, "C0")

    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"$\text{NLL}_\text{BC}$")
    ax[0].set_title("Naive")

    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$\text{NLL}_\text{BC}$")
    ax[1].set_title("Pairwise")
    ax[1].ticklabel_format(useOffset=False, style="plain")

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/numerical/fednll_boxcox.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)

    return nll_naive, nll_pairwise


if __name__ == "__main__":
    power = "boxcox"
    n_clients = 100

    rng = np.random.RandomState(304)
    x = rng.normal(1e4, 1e-3, 100)

    b = 50
    lmbs = np.linspace(-b, b, 100)
    main(lmbs, x, power, n_clients, rng)

    plt.show()

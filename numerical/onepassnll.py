from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from federated.core import FedPowerClient, FedPowerServer, IID_partitioner


def main(lmbs, x, n_clients, rng):
    x_clients = IID_partitioner(x, n_clients, rng)

    clients = [FedPowerClient("boxcox", x) for x in x_clients]
    server = FedPowerServer("boxcox", clients=clients)

    nll_naive = -server.llf(lmbs, var_comp="naive")
    nll_pairwise = -server.llf(lmbs, var_comp="pairwise")

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(lmbs, nll_naive, "C8--", label="Naive")
    ax.plot(lmbs, nll_pairwise, "C0", label="Pairwise")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\text{NLL}_\text{BC}$")
    ax.legend()

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/numerical/fednll_boxcox.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)


if __name__ == "__main__":
    rng = np.random.RandomState(304)
    x = rng.normal(1e4, 1e-3, 100)

    lmbs = np.linspace(-50, 50, 100)
    main(lmbs, x, 100, rng)

    plt.show()

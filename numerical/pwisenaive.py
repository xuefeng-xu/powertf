from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from federated.core import FedPowerClient, FedPowerServer, IID_partitioner
from dataloader import load_data


def main(power, X, dataset, feature, lmbs, n_clients, rng):
    x_clients = IID_partitioner(X[feature], n_clients, rng)

    clients = [FedPowerClient(power, x) for x in x_clients]
    server = FedPowerServer(power, clients=clients)

    nll_naive = -server.llf(lmbs, var_comp="naive")
    nll_pairwise = -server.llf(lmbs, var_comp="pairwise")

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot(lmbs, nll_naive, "C8--", label="Naive")
    ax.plot(lmbs, nll_pairwise, "C0", label="Pairwise")
    ax.ticklabel_format(useOffset=False)

    if power == "boxcox":
        ax.set_ylabel(r"$\text{NLL}_\text{BC}$")
        ax.set_title(f"{dataset.title()}: {feature} (BC)")
    elif power == "yeojohnson":
        ax.set_ylabel(r"$\text{NLL}_\text{YJ}$")
        ax.set_title(f"{dataset.title()}: {feature} (YJ)")

    ax.set_xlabel(r"$\lambda$")
    ax.legend()

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = (
        PROJECT_ROOT / f"img/numerical/pwisenaive/{power}-{dataset}-{feature}.pdf"
    )
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    lmbs = np.linspace(-0.01, 0.01, 100)

    dataset = "cancer"
    X, _ = load_data(dataset)
    main("yeojohnson", X, dataset, "concave_points2", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "fractal_dimension1", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "fractal_dimension2", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "smoothness1", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "smoothness2", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "symmetry2", lmbs, n_clients=100, rng=0)

    dataset = "house"
    X, _ = load_data(dataset)
    main("boxcox", X, dataset, "YearRemodAdd", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "YearRemodAdd", lmbs, n_clients=100, rng=0)
    main("boxcox", X, dataset, "YearBuilt", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "YearBuilt", lmbs, n_clients=100, rng=0)

    lmbs = np.linspace(-0.1, 0.1, 100)
    main("boxcox", X, dataset, "YrSold", lmbs, n_clients=100, rng=0)
    main("yeojohnson", X, dataset, "YrSold", lmbs, n_clients=100, rng=0)

from pathlib import Path
from argparse import ArgumentParser
from simulate import run_simulation
import matplotlib.pyplot as plt


def main(power, dataset, feature):
    n_points = [2**i for i in range(11)]
    n_rounds = []

    for nps in n_points:
        results = run_simulation(
            power=power,
            dataset=dataset,
            feature=feature,
            var_comp="pairwise",
            optimize="grid",
            n_points=nps,
            n_clients=100,
            full_output=1,
            n_reps=1,
            print_output=0,
        )
        n_rounds.append(results[0][2])

    base_results = run_simulation(
        power=power,
        dataset=dataset,
        feature=feature,
        var_comp="pairwise",
        optimize="brent",
        n_points=1,  # not used in brent
        n_clients=100,
        full_output=1,
        n_reps=1,
        print_output=0,
    )
    base_n_rounds = base_results[0][2]

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.loglog(n_points, n_rounds, marker="o", label="Grid Search")
    ax.axhline(base_n_rounds, linestyle="--", color="r", label="Brent")
    ax.set_xlabel("Number of points in grid")
    ax.set_ylabel("Number of rounds")
    ax.legend()
    ax.grid()

    if power == "boxcox":
        ax.set_title(f"{dataset.title()}: {feature} (BC)")
    else:
        ax.set_title(f"{dataset.title()}: {feature} (YJ)")

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/federated/comm/{power}-{dataset}-{feature}.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--power",
        type=str,
        default="yeojohnson",
        choices=["boxcox", "yeojohnson"],
        help="Power transform method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="blood",
        choices=["adult", "bank", "credit", "blood", "cancer", "ecoli", "house"],
        help="Dataset name",
    )
    parser.add_argument(
        "--feature", type=str, default="0", help="Feature index or name"
    )
    args = parser.parse_args()

    main(args.power, args.dataset, args.feature)

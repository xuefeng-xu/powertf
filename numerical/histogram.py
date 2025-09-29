import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import load_data


def main(X, dataset, feature, bins=30, log=False):
    fig, ax = plt.subplots(figsize=(3, 3))

    x = X[feature]
    ax.hist(x, bins=bins, log=log)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Counts")
    ax.set_title(f"{dataset.title()}: {feature}")

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/numerical/hist/{dataset}-{feature}.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    dataset = "blood"
    X, _ = load_data(dataset)

    main(X, dataset, "Monetary", bins=40)

    dataset = "cancer"
    X, _ = load_data(dataset)

    main(X, dataset, "ID", bins=30)
    main(X, dataset, "area1", bins=40)
    main(X, dataset, "area3", bins=40)

    dataset = "ecoli"
    X, _ = load_data(dataset)

    main(X, dataset, "lip", bins=30)
    main(X, dataset, "chg", bins=30, log=True)

    dataset = "house"
    X, _ = load_data(dataset)

    main(X, dataset, "Street", bins=30, log=True)
    main(X, dataset, "YearRemodAdd", bins=40)
    main(X, dataset, "YrSold", bins=30)

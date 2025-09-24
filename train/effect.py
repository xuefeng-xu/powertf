import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
from dataloader import load_data


def main(dataset, model, X_train, X_test, y_train, y_test):
    fig, ax = plt.subplots(figsize=(3, 3))

    if model == "LR":
        clf = LogisticRegression(max_iter=20000, random_state=42)
    elif model == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif model == "QDA":
        clf = QuadraticDiscriminantAnalysis()

    # Power Transform
    power = PowerTransformer()
    X_train_power = power.fit_transform(X_train)
    X_test_power = power.transform(X_test)
    clf.fit(X_train_power, y_train)
    RocCurveDisplay.from_estimator(clf, X_test_power, y_test, ax=ax, name="Power")

    # Standardization
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)
    clf.fit(X_train_std, y_train)
    RocCurveDisplay.from_estimator(
        clf, X_test_std, y_test, ax=ax, name="STD", curve_kwargs={"ls": "--"}
    )

    # Raw Features
    with warnings.catch_warnings():
        # ignore convergence warnings since train on raw features
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(
        clf, X_test, y_test, ax=ax, name="Raw", curve_kwargs={"ls": ":"}
    )

    ax.set_title(f"ROC: {dataset.title()} ({model})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/train/effect/roc-{dataset}-{model}.pdf"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="LDA",
        choices=["LDA", "QDA", "LR"],
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "bank", "credit"],
        help="Dataset name",
    )
    args = parser.parse_args()

    X, y = load_data(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    main(args.dataset, args.model, X_train, X_test, y_train, y_test)

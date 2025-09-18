from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data
from numerical.optimize import yeojohnson_mle


def main(dataset, feature, model, x_train, x_test, y_train, y_test, eps, n_points):
    fig, ax = plt.subplots(figsize=(3, 3))

    if model == "LR":
        clf = LogisticRegression(random_state=42)
    elif model == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif model == "QDA":
        clf = QuadraticDiscriminantAnalysis(reg_param=1e-3)

    lmb_opt = yeojohnson_mle(x_train)
    lmbs = np.linspace(lmb_opt * (1 - eps), lmb_opt * (1 + eps), n_points)

    power = PowerTransformer(standardize=False)
    std = StandardScaler()

    auc = []
    for l in lmbs:
        power.lambdas_ = np.array([l])
        x_train_lmb = power.transform(x_train)
        x_train_lmb = std.fit_transform(x_train_lmb)

        x_test_lmb = power.transform(x_test)
        x_test_lmb = std.transform(x_test_lmb)

        clf.fit(x_train_lmb, y_train)
        y_pred = clf.predict_proba(x_test_lmb)[:, 1]

        auc.append(roc_auc_score(y_test, y_pred))

    ax.plot(lmbs, auc, label="AUC")
    ax.axvline(lmb_opt, linestyle=":", color="k", label=r"$\lambda^*$")

    ax.set_title(f"{dataset.title()}: {feature} ({model})")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("AUC")
    ax.legend()

    PROJECT_ROOT = Path(__file__).parent.parent
    img_path = PROJECT_ROOT / f"img/train/deviate/auc-{dataset}-{model}.pdf"
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
    parser.add_argument(
        "--eps",
        type=float,
        default=100,
        help="Epsilon value for deviation",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=100,
        help="Number of points",
    )
    args = parser.parse_args()

    X, y = load_data(args.dataset)

    fea_select = SelectKBest(mutual_info_classif, k=1).fit(X, y)
    feature = fea_select.get_feature_names_out()[0]
    x = fea_select.transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    main(
        args.dataset,
        feature,
        args.model,
        x_train,
        x_test,
        y_train,
        y_test,
        args.eps,
        args.n_points,
    )

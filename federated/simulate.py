from argparse import ArgumentParser
from dataloader import load_data
from core import FedPowerClient, FedPowerServer, IID_partitioner


def parse_arguments():
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
    parser.add_argument(
        "--var_comp",
        type=str,
        default="pairwise",
        choices=["pairwise", "naive"],
        help="Variance computation method",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="brent",
        choices=["brent", "grid"],
        help="Optimization method",
    )
    parser.add_argument(
        "--n_points", type=int, default=20, help="Number of points for grid search"
    )
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument(
        "--full_output",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to return full output",
    )
    parser.add_argument("--n_reps", type=int, default=1, help="Number of repetitions")
    parser.add_argument(
        "--print_output",
        type=int,
        default=1,
        choices=[0, 1],
        help="Print output or not",
    )
    return parser.parse_args()


def run_simulation(
    power,
    dataset,
    feature,
    var_comp,
    optimize,
    n_points,
    n_clients,
    full_output,
    n_reps,
    print_output,
):
    strictly_positive = True if power == "boxcox" else False
    X, _ = load_data(dataset, strictly_positive)

    if feature.isdigit():
        col = X.columns[int(feature)]
    else:
        col = feature

    results = []

    for i in range(n_reps):
        x_clients = IID_partitioner(X[col], n_clients, rng=i)

        clients = [FedPowerClient(power, x) for x in x_clients]
        server = FedPowerServer(power, clients)

        if full_output:
            if optimize == "brent":
                lmb, nll, _, n_rounds = server.mle(
                    var_comp=var_comp,
                    optimize="brent",
                    full_output=1,
                )

            elif optimize == "grid":
                lmb, nll, n_rounds = server.mle(
                    var_comp=var_comp,
                    optimize="grid",
                    n_points=n_points,
                    full_output=1,
                )

            results.append((lmb, nll, n_rounds))
            if print_output:
                print(f"lmb: {lmb}, nll: {nll}, n_rounds: {n_rounds}")

        else:
            lmb = server.mle(
                var_comp=var_comp, optimize=optimize, n_points=n_points, full_output=0
            )

            results.append((lmb,))
            if print_output:
                print(f"lmb: {lmb}")

    return results


if __name__ == "__main__":
    args = parse_arguments()

    run_simulation(
        power=args.power,
        dataset=args.dataset,
        feature=args.feature,
        var_comp=args.var_comp,
        optimize=args.optimize,
        n_points=args.n_points,
        n_clients=args.n_clients,
        full_output=args.full_output,
        n_reps=args.n_reps,
        print_output=args.print_output,
    )

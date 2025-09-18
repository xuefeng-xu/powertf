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
        help="Power transformation method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "bank", "credit", "blood", "cancer", "ecoli", "house"],
        help="Dataset name",
    )
    parser.add_argument(
        "--var_comp",
        type=str,
        default="pairwise",
        choices=["pairwise", "naive"],
        help="Variance computation method",
    )
    parser.add_argument("--n_clients", type=int, default=10, help="Number of clients")
    parser.add_argument(
        "--full_output", type=int, default=0, help="Whether to return full output"
    )
    parser.add_argument("--n_reps", type=int, default=1, help="Number of repetitions")
    return parser.parse_args()


def main():
    args = parse_arguments()

    strictly_positive = True if args.power == "boxcox" else False
    X, _ = load_data(args.dataset, strictly_positive)

    for col in X.columns:
        print("Feature:", col)

        for i in range(args.n_reps):
            x_clients = IID_partitioner(X[col], args.n_clients, rng=i)

            clients = [FedPowerClient(args.power, x) for x in x_clients]
            server = FedPowerServer(args.power, clients)

            if args.full_output:
                lmb, nll, _, n_rounds = server.mle(full_output=1)
                print(f"lmb: {lmb}, nll: {nll}, n_rounds: {n_rounds}")
            else:
                print(f"lmb: {server.mle()}")


if __name__ == "__main__":
    main()

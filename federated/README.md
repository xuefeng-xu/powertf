## Federated Experiments

### Communication Rounds vs Grid Size (Figure 15)

```bash
python comm.py --dataset blood --feature Monetary
python comm.py --dataset cancer --feature area1
python comm.py --dataset ecoli --feature lip
python comm.py --dataset ecoli --feature chg
python comm.py --dataset house --feature Street
python comm.py --dataset house --feature YrSold
```

| Parameter | Description | Values |
|---|---|---|
| `power` | Power transform method | `boxcox` or `yeojohnson` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`credit`](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [`blood`](https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center), [`cancer`](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [`ecoli`](https://archive.ics.uci.edu/dataset/39/ecoli), [`house`](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) |
| `feature` | Feature index or name | Integer (e.g., `0`, `1`, ...) or string (e.g., feature name) |

### Simulate Federated Power Transform

```bash
python simulate.py --power yeojohnson --dataset blood --full_output 1
```

| Parameter | Description | Values |
|---|---|---|
| `power` | Power transform method | `boxcox` or `yeojohnson` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`credit`](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [`blood`](https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center), [`cancer`](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [`ecoli`](https://archive.ics.uci.edu/dataset/39/ecoli), [`house`](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) |
| `feature` | Feature index or name | Integer (e.g., `0`, `1`, ...) or string (e.g., feature name) |
| `var_comp` | Variance computation method | `pairwise` or `naive` |
| `optimize` | Optimization method | `brent` or `grid` |
| `n_points` | Number of points for grid search | Integer (e.g., `20`, `50`) |
| `n_clients` | Number of clients | Integer (e.g., `10`, `100`) |
| `full_output` | Whether to return full output | `0` or `1` |
| `n_reps` | Number of repetitions | Integer (e.g., `1`, `3`) |
| `print_output` | Print output or not | `0` or `1` |

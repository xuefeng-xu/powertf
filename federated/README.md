## Federated Experiments

```bash
python simulate.py --power yeojohnson --dataset blood --full_output 1
```

### Parameters

| Parameter | Description | Values |
|---|---|---|
| `power` | Transformation method | `boxcox` or `yeojohnson` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`credit`](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [`blood`](https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center), [`cancer`](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [`ecoli`](https://archive.ics.uci.edu/dataset/39/ecoli), [`house`](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) |
| `var_comp` | Variance computation method | `pairwise` or `naive` |
| `n_clients` | Number of clients | Integer (e.g., `10`, `100`) |
| `full_output` | Whether to return full output | Integer (e.g., `0`, `1`) |
| `n_reps` | Number of repetitions | Integer (e.g., `1`, `3`) |

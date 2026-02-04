## Model Training Experiments

### Effect on Models (Figure 7 and 11)

```bash
python effect.py --model LDA --dataset adult
python effect.py --model LDA --dataset bank
python effect.py --model LDA --dataset credit
```

```bash
python effect.py --model LR --dataset adult
python effect.py --model LR --dataset bank
python effect.py --model LR --dataset credit
```

```bash
python effect.py --model XGB --dataset adult
python effect.py --model XGB --dataset bank
python effect.py --model XGB --dataset credit
```

Plots are saved in `../img/train/effect/roc-{dataset}-{model}.pdf`.

| Parameter | Description | Values |
|---|---|---|
| `model` | Model name | `LDA`, `LR`, or `XGB` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`credit`](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |

---

### Deviate of $\lambda$ (Figure 8 and 12)

```bash
python deviate.py --model LDA --dataset adult  --eps 500
python deviate.py --model LDA --dataset bank   --eps 300
python deviate.py --model LDA --dataset credit --eps 200
```

```bash
python deviate.py --model LR --dataset adult  --eps 500
python deviate.py --model LR --dataset bank   --eps 300
python deviate.py --model LR --dataset credit --eps 200
```

```bash
python deviate.py --model XGB --dataset adult  --eps 500
python deviate.py --model XGB --dataset bank   --eps 300
python deviate.py --model XGB --dataset credit --eps 200
```

Plots are saved in `../img/train/deviate/auc-{dataset}-{model}.pdf`.

| Parameter | Description | Values |
|---|---|---|
| `model` | Model name | `LDA`, `LR`, or `XGB` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`credit`](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| `eps` | Epsilon value for deviation | Float (e.g., `200`) |
| `n_points` | Number of points | Integer (e.g., `100`) |

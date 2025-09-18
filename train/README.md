## Model Training Experiments

### Effect on Models (Figure 6 and 9)

```bash
python effect.py --model LDA --dataset adult
python effect.py --model LDA --dataset bank
python effect.py --model LDA --dataset credit
```

```bash
python effect.py --model QDA --dataset adult
python effect.py --model QDA --dataset bank
python effect.py --model QDA --dataset credit
```

```bash
python effect.py --model LR --dataset adult
python effect.py --model LR --dataset bank
python effect.py --model LR --dataset credit
```

#### Parameters

| Parameter | Description | Values |
|---|---|---|
| `model` | Model name | `LDA`, `QDA`, or `LR` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`credit`](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |


### Deviate of $\lambda$ (Figure 7 and 10)

```bash
python deviate.py --model LDA --dataset adult --eps 500
python deviate.py --model LDA --dataset bank --eps 300
python deviate.py --model LDA --dataset credit --eps 200
```

```bash
python deviate.py --model QDA --dataset adult --eps 500
python deviate.py --model QDA --dataset bank --eps 300
python deviate.py --model QDA --dataset credit --eps 200
```

```bash
python deviate.py --model LR --dataset adult --eps 500
python deviate.py --model LR --dataset bank --eps 300
python deviate.py --model LR --dataset credit --eps 200
```

#### Parameters

| Parameter | Description | Values |
|---|---|---|
| `model` | Model name | `LDA`, `QDA`, or `LR` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`credit`](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| `eps` | Epsilon value for deviation | Float (e.g., `200`) |
| `n_points` | Number of points | Integer (e.g., `100`) |

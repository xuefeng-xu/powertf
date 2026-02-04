## Numerical Experiments

### Adversarial Data (Table 1)

```bash
python adversarial.py
```

Results are printed in the terminal.

---

### Overflow using Exponential Search (Figure 4)

```bash
python overflow.py
```

Plots are saved in `../img/numerical/boxcox_expsearch_*.pdf`.

---

### Numerical Stability Tests of NLL (Figure 5)

```bash
python stablenll.py
```

Plots are saved in `../img/numerical/nll_boxcox_*.pdf`.

---

### One-Pass Federated NLL (Figure 6)

```bash
python onepassnll.py
```

Plot is saved in `../img/numerical/fednll_boxcox.pdf`.

---

### Time comparison (Table 4)

```bash
python time.py
```

Results are printed in the terminal.

---

### The following experiments need to download the [`train.csv`](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv) file and put it under the folder `dataset/house/`.

### Brent vs ExpSearch (Figure 9 and 14)

```bash
python brentexp.py
```

Plots are saved in `../img/numerical/brentexp/{power}-{dataset}-{feature}.pdf`.

---

### Log-domain vs linear-domain (Figure 10 and 15)

```bash
python loglinear.py
```

Plots are saved in `../img/numerical/loglinear/{power}-{dataset}-{feature}.pdf`.

---

### Pairwise vs Naive (Figure 16)

```bash
python pwisenaive.py
```

Plots are saved in `../img/numerical/pwisenaive/{power}-{dataset}-{feature}.pdf`.

---

### Feature histogram (Figure 13)

```bash
python histogram.py
```

Plots are saved in `../img/numerical/hist/{dataset}-{feature}.pdf`.

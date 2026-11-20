# Predictive maintenance

A finished milling cycle has six readings. The app turns those readings into a stop-or-run decision, so the risky machines are inspected and the rest keep running.

The dataset is [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository. `data/ai4i2020.csv` is the whole published set: 10,000 cycles, 339 failures (3.39%). `data/one_shift.csv` is a 24-cycle upload example taken from that file.

## Run

```powershell
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

`train.py` downloads the CSV if it is missing, compares models, writes the test scores to `data/holdout_review.csv`, then fits the shipped model on all 10,000 cycles.

## Model

The comparison uses a stratified 80/20 split, `random_state` 42 (8,000 train / 2,000 test, 68 test failures). The threshold is chosen on out-of-fold training scores only: catch at least 80% of those failures, then take the highest precision. The test is scored once after that choice. Random forest wins on out-of-fold PR-AUC. After the test is saved, the same random forest is fit again on all 10,000 cycles. The app loads that second fit and keeps the earlier threshold.

A model that never stops a machine is already right on 96.6% of cycles.

| Model | OOF PR-AUC | Holdout PR-AUC | Holdout ROC-AUC | Recall | Precision | Caught | False alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.455 | 0.430 | 0.938 | 83.8% | 20.4% | 57 | 222 |
| Random forest (shipped) | 0.891 | 0.864 | 0.969 | 80.9% | 91.7% | 55 | 5 |
| Gradient boosting | 0.860 | 0.903 | 0.984 | 83.8% | 86.4% | 57 | 9 |

On the locked test the forest stops 60 cycles: 55 real failures and 5 unnecessary stops. It misses 13 failures. Nine of those misses are pure tool-wear failures. Tool wear in this dataset is drawn partly at random between 200 and 240 minutes.

## Inputs

- Product type L / M / H
- Air temperature (K)
- Process temperature (K)
- Rotational speed (rpm)
- Torque (N·m)
- Tool wear (min)
- Temperature gap (process − air)
- Mechanical power (W) = torque × rpm × 2π / 60
- Wear × torque

`UDI`, `Product ID`, and the flags `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are not inputs. Those flags are labels. The target is the published `Machine failure` column.

Permutation importance on the locked test, as a drop in PR-AUC: rotational speed, temperature gap, wear × torque, mechanical power, torque.

## How the labels were generated

These rules describe the published dataset. They are not hard-coded as features.

- Heat dissipation: gap under 8.6 K and speed under 1,380 rpm.
- Power: mechanical power under 3,500 W or over 9,000 W.
- Overstrain: wear × torque over 11,000 / 12,000 / 13,000 for L / M / H.
- Tool wear: a failure drawn at random between 200 and 240 minutes.
- Random: about 0.1% of cycles, with no sensor pattern.

24 cycles have more than one flag. 9 failures have no flag. 18 flagged cycles are not marked `Machine failure`.

## Layout

```
app.py                    Streamlit app
train.py                  Compare, save the test, refit on all 10,000
features.py               Shared features and local explanations
data/ai4i2020.csv         All 10,000 cycles
data/holdout_review.csv   Test scores
data/one_shift.csv        24-cycle upload example
models/model.joblib       Model fit on all 10,000 cycles
models/metrics.json       Comparison numbers
```

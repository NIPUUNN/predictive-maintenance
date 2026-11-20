# Milling inspection

A finished milling cycle has six readings. This project turns those readings into a stop-or-run decision, so a technician inspects the risky machines instead of every machine.

The dataset is [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository. The file `data/ai4i2020.csv` is the whole published set: 10,000 cycles, 339 failures (3.39%). Nothing was added to it. The 24-row file `data/one_shift.csv` is only an upload example taken from those 10,000.

## Run

```powershell
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

`train.py` downloads the CSV if it is missing, compares models, writes the locked test to `data/holdout_review.csv`, then fits the shipped model on all 10,000 cycles.

## What an interviewer can check

Open the app.

1. The first lines state the use and the dataset.
2. **Check a cycle.** An ordinary cycle stays running. A tight temperature gap at low spindle speed, a low-torque cycle, and a worn tool at high torque each say stop, for a different reason.
3. **Test on 2,000 held-out cycles.** That table is the result that matters. It is not the form.
4. **The dataset.** Row counts, the model comparison, and which columns were excluded.

The form uses the model fit on all 10,000 cycles, so its example machines are not unseen. The test tab is.

## Result

A model that never stops a machine is already 96.6% accurate. Accuracy is not the number to quote.

Comparison used a stratified 80/20 split, `random_state` 42 (8,000 train / 2,000 test, 68 test failures). The threshold was chosen only on out-of-fold training scores: catch at least 80% of those failures, then take the highest precision. The test was scored once after that choice. Random forest won on out-of-fold PR-AUC, so it is the shipped model. After the test was saved, that same random forest was fit again on all 10,000 cycles. The app loads the second fit and keeps the earlier threshold.

| Model | OOF PR-AUC | Holdout PR-AUC | Holdout ROC-AUC | Recall | Precision | Caught | False alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.455 | 0.430 | 0.938 | 83.8% | 20.4% | 57 | 222 |
| **Random forest (shipped)** | **0.891** | **0.864** | **0.969** | **80.9%** | **91.7%** | **55** | **5** |
| Gradient boosting | 0.860 | 0.903 | 0.984 | 83.8% | 86.4% | 57 | 9 |

On the locked test the forest stops 60 cycles: 55 real failures and 5 unnecessary stops. It misses 13 failures. Nine of those misses are pure tool-wear failures. Tool wear in this dataset is drawn partly at random between 200 and 240 minutes, so the sensors do not fully determine it.

Logistic regression, at a similar recall, raises 222 false alarms. That is the reason for a tree model.

## What the model sees

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

## Dataset rules

These are how the published labels were generated. They are not hard-coded as features.

- Heat dissipation: gap under 8.6 K and speed under 1,380 rpm.
- Power: mechanical power under 3,500 W or over 9,000 W.
- Overstrain: wear × torque over 11,000 / 12,000 / 13,000 for L / M / H.
- Tool wear: a failure drawn at random between 200 and 240 minutes.
- Random: about 0.1% of cycles, with no sensor pattern.

24 cycles have more than one flag. 9 failures have no flag. 18 flagged cycles are not marked `Machine failure`.

## Resume line

Predictive maintenance for a milling cycle | Python, scikit-learn, Random Forest, Pandas

- Built a stop-or-run check from the six process readings on AI4I 2020 (UCI, 10,000 cycles, 3.39% failures). Failure-mode flags were labels, not inputs. The deployed forest was fit on all 10,000 cycles.
- Compared logistic regression, random forest, and gradient boosting on a locked 2,000-cycle test. The forest caught 55 of 68 failures at 92% precision, with 5 false alarms. Logistic regression at a similar recall fell to 20% precision.
- The test set, not the demo buttons, is the result. Nine of the 13 misses are tool-wear failures, which this dataset assigns partly at random.

## Layout

```
app.py                    Inspection page
train.py                  Compare, lock the test, refit on all 10,000
features.py               Shared features and local explanations
data/ai4i2020.csv         All 10,000 cycles
data/holdout_review.csv   Locked test scores
data/one_shift.csv        24-cycle upload example
models/model.joblib       Model fit on all 10,000 cycles
models/metrics.json       Comparison numbers
```

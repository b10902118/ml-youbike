# Results
## 7days
```
Train: 10/02 - 11/19
Test: 11/08 - 11/14
Ein = 0.2610810510255989
MAE 3.644026360544218
Score 0.2298803781481282
Kaggle: 0.29165
```
## brute err (group by sno,time,holiday)
```
Train: 10/02 - 11/19
Test: 11/08 - 11/14
    Ein = 0.31449796693431414
    MAE 4.1863307823129245
    Score 0.2776735544033527
    Kaggle: 0.28772

Train: 10/02 - 10/20
Test: 11/08 - 11/14
    Ein = 0.31625753705296
    MAE 4.749450821995466
    Score 0.34012555926254384
    Kaggle: 0.37409

Train: 10/02 - 11/07
Test: 11/08 - 11/14
    Ein = 0.30575774115991416
    MAE 4.667676445578231
    Score 0.3146647423497437
    Kaggle: 0.28409
---------------------------------
10/2 - 10/20
Ein = 0.31493194869909713 (step = 0.1)
Ein = 0.31625753705296 (step = 1)
Test: 10/25-10/31
MAE 4.274447278911564
Score 0.35013584858607594
```
## k-means
```
MAE 4.817353558541193
Score 0.397607157557498
Kaggle: 0.40395
```
## simple_mean
```
TRAIN_START = "2023-10-02 00:00"
TRAIN_END = "2023-11-07 23:59:00"
TEST_START = "2023-11-08 00:00"
TEST_END = "2023-11-14 23:59"

MAE 4.3378884457222044
Score 0.3482707080848104
Kaggle: 0.37186
-----------------------------------
TRAIN_START = "2023-10-02 00:00"
TRAIN_END = "2023-10-20 23:59"
TEST_START = "2023-10-25 00:00"
TEST_END = "2023-10-28 23:59"

MAE 4.949248624639249
Score 0.4601133171423995
Kaggle: 0.41493
-----------------------------------

```
## baseline_xgboost T predict T+1
```
MAE 5.811096263316508
Score 0.4270636654857978
Kaggle: 
```

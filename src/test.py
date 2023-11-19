from datetime import datetime
import numpy as np
from operator import index, is_, sub
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import *


TRAIN_START = "20231002"
TRAIN_END = "20231020"
TEST_START = "20231025"
TEST_END = "20231028"
# PRED_DATE_RANGES = [("20231021",20231024),("20231204",20231210)]

with open("./cache/small_data_cache.pkl", "rb") as f:
    df = pickle.load(f)

with open("../html.2023.final.data/sno_test_set.txt") as f:
    ntu_snos = [l.strip() for l in f.read().splitlines()]

ntu_tots = get_tot(df, ntu_snos)
# the data looks like this:
"""
     datetime               sno      tot   sbi   bemp  act
0    2023-10-02 00:00:00  500101001  28.0  12.0  16.0   1
1    2023-10-02 00:01:00  500101001  28.0  12.0  16.0   1
2    2023-10-02 00:02:00  500101001  28.0  13.0  15.0   1
...
"""


holidays = [
    d for d in pd.date_range(start=TRAIN_START, end=TEST_END).date if is_holiday(d)
]

tb = (
    pd.pivot_table(df, index="time", columns="sno", values="sbi")
    .resample("20min")
    .agg("first")
)
train = tb[tb.index.to_series().dt.date.isin(date_range(TRAIN_START, TRAIN_END))]
test = tb[tb.index.to_series().dt.date.isin(date_range(TEST_START, TEST_END))]

y_test = test.values

means = train.groupby(
    by=[
        train.index.time,
        train.index.to_series().dt.date.isin(holidays),
    ]
).mean()
# print(means.loc[~means.index.get_level_values(1)])
# print(means.dtypes) # float64
# exit()

ftr = list(
    np.stack([test.index.time, test.index.to_series().dt.date.apply(is_holiday)]).T
)
y_pred = means.loc[ftr].values
evaluation(y_test, y_pred, ntu_tots)

exit()


# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range("2023/10/21 00:00", "2023/10/24 23:59", freq="20min")
# list makes indexer 1D, or it is an 2D indexer
ftr = list(
    np.stack(
        [public_test_range.time, np.vectorize(is_holiday)(public_test_range.date)]
    ).T
)
# print(ftr)
# ftr = list(ftr)
# print(ftr)

y_public_test = means.loc[ftr].values
public_test_df = pd.DataFrame(y_public_test, columns=ntu_snos, index=public_test_range)

# we haven't do this yet, but it is required for submission
private_test_range = pd.date_range("2023/12/04 00:00", "2023/12/10 23:59", freq="20min")
private_test_df = pd.DataFrame(0, columns=ntu_snos, index=private_test_range)


# convert the prediction to the required format
tmp = pd.concat(
    [
        public_test_df,
        private_test_df,
    ]
)
# reset_index: old index => "time" column
# id_vars: fixed column like index
# var_name: columns.name to "sno" column
# value_name: value => "sbi" column
tmp = tmp.reset_index(names="time").melt(
    id_vars="time", var_name="sno", value_name="sbi"
)
out_df = pd.DataFrame(
    {
        "id": (
            tmp["time"].dt.strftime("%Y%m%d")
            + "_"
            + tmp["sno"]
            + "_"
            + tmp["time"].dt.strftime("%H:%M")
        ),
        "sbi": tmp["sbi"],
    }
)
out_df.to_csv(
    f"../submission/{datetime.now().strftime('%m-%d-%H-%M')}.csv", index=False
)

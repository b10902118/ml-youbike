from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import *
from median_optimization import optimal_median

TRAIN_START = "2023-10-02 00:00"
# TRAIN_END = "2023-11-07 23:59:00"
# TRAIN_END = "2023-10-20 23:59"
TRAIN_END = "2023-11-30 23:59"

# TEST_START = "2023-11-08 00:00"
# TEST_END = "2023-11-14 23:59"
TEST_START = "2023-10-25 00:00"
TEST_END = "2023-10-31 23:59"


PUBLIC_START = "2023-10-21 00:00"
PUBLIC_END = "2023-10-24 23:59"

PRIVATE_START = "2023-12-04 00:00"
PRIVARE_END = "2023-12-10 23:59"

with open("./cache/small_data_cache.pkl", "rb") as f:
    df = pd.read_pickle(f)

with open("../html.2023.final.data/sno_test_set.txt") as f:
    ntu_snos = [l.strip() for l in f.read().splitlines()]

ntu_tots = get_tot(df, ntu_snos)
# the data looks like this:
"""
        time               sno      tot   sbi   bemp  act
0    2023-10-02 00:00:00  500101001  28.0  12.0  16.0   1
1    2023-10-02 00:01:00  500101001  28.0  12.0  16.0   1
2    2023-10-02 00:02:00  500101001  28.0  13.0  15.0   1
...
"""


holidays = [d for d in date_range(start=TRAIN_START, end=PRIVARE_END) if is_holiday(d)]
exit()

tb = (
    pd.pivot_table(df, index="time", columns="sno", values="sbi")
    .resample("20min")
    .agg("first")
)
# [] only provides view,so assigning to it cause warning
train = tb[tb.index.to_series().dt.date.isin(date_range(TRAIN_START, TRAIN_END))].copy()
train.reset_index(names="time", inplace=True)
train["weekday"] = train["time"].dt.weekday
train.set_index(["time", "weekday"], inplace=True)

test = tb[tb.index.to_series().dt.date.isin(date_range(TEST_START, TEST_END))]
# print(test)
y_test = test.values


# final format:
"""
        sno 500101001 500101002
time  sbi
00:00          1.1         2.0
00:20
"""
result_df = pd.DataFrame(
    columns=ntu_snos,
    index=pd.MultiIndex.from_product(
        [pd.date_range("00:00", "23:59", freq="20min").time, range(7)],
        names=("time", "weekday"),
    ),
    dtype=np.float64,
)

Ein = 0.0
for sno, tot in zip(ntu_snos, ntu_tots):
    # sd = station data
    sd = train[sno].to_frame()
    sd.rename(columns={sno: "sbi"}, inplace=True)
    sd.reset_index(["time", "weekday"], inplace=True)
    sd["date"] = sd["time"].dt.date
    sd["time"] = sd["time"].dt.time
    # print(sd)
    # exit()
    psd = pd.pivot_table(sd, index=["date", "weekday"], columns="time", values="sbi")
    # sno col have its sbi
    # print(psd)
    for day in range(7):  # 0 to 6
        for t in psd.columns:
            # print(t, sno)
            sbi, err = optimal_median(
                y_true=psd.loc[psd.index.get_level_values("weekday") == day, t].values,
                tot=tot,
            )
            Ein += err
            # print(f"{t} sbi:{sbi}   err: {err}")
            result_df.at[(t, day), sno] = sbi
            # result_df.at[[t, isholiday], sno] = np.float64(sbi)
            # print(result_df.at[t, sno])

# print(result_df)
# we have avg by #date, now by #sno and #time
Ein /= result_df.size
print_time_ranges(TRAIN_START, TRAIN_END, TEST_START, TEST_END)
print(f"Ein = {Ein}")

# exit()

# self evaluation
ftr = list(np.stack([test.index.time, test.index.to_series().dt.weekday]).T)
y_pred = result_df.loc[ftr].values
local_test_range = pd.date_range(TEST_START, TEST_END, freq="20min")

# print(y_test)
# print(y_pred)
# exit()
evaluation(y_test, y_pred, ntu_tots, local_test_range)

# exit()


# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range(PUBLIC_START, PUBLIC_END, freq="20min")
# list makes indexer 1D, or it is an 2D indexer
ftr = list(np.stack([public_test_range.time, public_test_range.weekday]).T)
# print(ftr)
# ftr = list(ftr)
# print(ftr)
y_public_test = result_df.loc[ftr].values
public_test_df = pd.DataFrame(y_public_test, columns=ntu_snos, index=public_test_range)

# we haven't do this yet, but it is required for submission
private_test_range = pd.date_range(PRIVATE_START, PRIVARE_END, freq="20min")
ftr = list(np.stack([private_test_range.time, private_test_range.weekday]).T)
y_private_test = result_df.loc[ftr].values
private_test_df = pd.DataFrame(
    y_private_test, columns=ntu_snos, index=private_test_range
)


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
    f"../submission/pub_pri_{datetime.now().strftime('%m-%d-%H-%M')}.csv", index=False
)
print("csv created")

print("plotting all")
import calendar

df["datetime"] = df["time"]
df["date"] = df["time"].dt.date
df["time"] = df["time"].dt.time
abbr = list(calendar.day_abbr)
for d in range(7):
    day_df = df[df["datetime"].dt.weekday == d]
    plot_line(day_df, ntu_snos, abbr[d])

holiday_df = df[df["date"].isin(holidays)]
plot_line(holiday_df, ntu_snos, "holiday")
weekday_df = df[~df["date"].isin(holidays)]
plot_line(weekday_df, ntu_snos, "weekday")

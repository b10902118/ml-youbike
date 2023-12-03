from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import *
from median_optimization import optimal_median

TRAIN_START = "2023-10-02 00:00"
TRAIN_END = "2023-12-03 23:59"

TEST_START = "2023-10-28 00:00"
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
     datetime               sno      tot   sbi   bemp  act
0    2023-10-02 00:00:00  500101001  28.0  12.0  16.0   1
1    2023-10-02 00:01:00  500101001  28.0  12.0  16.0   1
2    2023-10-02 00:02:00  500101001  28.0  13.0  15.0   1
...
"""


holidays = [d for d in date_range(start=TRAIN_START, end=PRIVARE_END) if is_holiday(d)]

old_tb = pd.pivot_table(df, index="time", columns="sno", values="sbi")
tb = (
    pd.pivot_table(df, index="time", columns="sno", values="sbi")
    .resample("20min")
    .agg("first")
)
# exclude long holidays
tb = tb[~tb.index.to_series().dt.date.isin(long_holiday)]
# [] only provides view,so assigning to it cause warning
train = tb[tb.index.to_series().dt.date.isin(date_range(TRAIN_START, TRAIN_END))].copy()
train.reset_index(names="time", inplace=True)
train["weekday"] = train["time"].dt.weekday
train.set_index(["time", "weekday"], inplace=True)

test = tb[tb.index.to_series().dt.date.isin(date_range(TEST_START, TEST_END))]
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
        [pd.date_range("00:00", "23:59", freq="20min").time, [0, 5, 6]],
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
    for day in [0, 5, 6]:  # 0 to 6
        for t in psd.columns:
            # print(t, sno)
            sbi, err = 0, 0
            # print(
            #    psd.loc[psd.index.get_level_values("weekday").isin(range(5)), t].values
            # )
            if day == 0:  # weekday
                sbi, err = optimal_median(
                    y_true=psd.loc[
                        psd.index.get_level_values("weekday").isin(range(5)), t
                    ].values,
                    tot=tot,
                )
            else:
                sbi, err = optimal_median(
                    y_true=psd.loc[
                        psd.index.get_level_values("weekday") == day, t
                    ].values,
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


def trans(s):
    if s in range(5):
        return 0
    return s


# self evaluation
ftr = list(
    np.stack([test.index.time, test.index.to_series().dt.weekday.apply(trans)]).T
)
y_pred = result_df.loc[ftr].values
local_test_range = pd.date_range(TEST_START, TEST_END, freq="20min")

# print(y_test)
# print(y_pred)
# exit()
evaluation(y_test, y_pred, ntu_tots, local_test_range)


# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range(PUBLIC_START, PUBLIC_END, freq="20min")
# list makes indexer 1D, or it is an 2D indexer
ftr = list(
    np.stack([public_test_range.time, np.vectorize(trans)(public_test_range.weekday)]).T
)
y_public_df = result_df.loc[ftr]
print(y_public_df)

# TODO patch
# Set the initial time

current_datetime = pd.to_datetime("2023-10-20 23:59")
current_time = current_datetime.time()
cur_data = old_tb[old_tb.index == current_datetime]
print(cur_data)
cur = old_tb[old_tb.index.to_series().dt.time == current_time]

# Loop to fetch data for the next 20 minutes
end_datetime = pd.to_datetime("2023-10-21 03:59")
next_datetime = current_datetime
td = timedelta(minutes=20)
total_td = timedelta(minutes=0)
while next_datetime <= end_datetime:
    # Increment current_time by 20 minutes
    next_datetime += td
    next_time = next_datetime.time()
    total_td += td
    # Filter data from old_tb for the current time
    nxt = old_tb[old_tb.index.to_series().dt.time == next_time]
    diff = nxt - cur.shift(freq=total_td)
    mean_diff = pd.pivot_table(diff.mean().reset_index(), columns="sno")
    mean_diff.set_index(cur_data.index, inplace=True)
    upd = cur_data + mean_diff
    # print(upd)
    patch_datetime = next_datetime + timedelta(minutes=1)
    patch_time = patch_datetime.time()
    upd.set_index([[patch_time], [5]], inplace=True)
    y_public_df.loc[(patch_time, 5)] = upd
print(y_public_df)

y_public_test = y_public_df.values
public_test_df = pd.DataFrame(y_public_test, columns=ntu_snos, index=public_test_range)

private_test_range = pd.date_range(PRIVATE_START, PRIVARE_END, freq="20min")
ftr = list(
    np.stack(
        [private_test_range.time, np.vectorize(trans)(private_test_range.weekday)]
    ).T
)
y_private_df = result_df.loc[ftr]
# TODO patch
# Set the initial time

current_datetime = pd.to_datetime("2023-12-03 23:40")
current_time = current_datetime.time()
cur_data = old_tb[old_tb.index == current_datetime]
print(cur_data)
cur = old_tb[old_tb.index.to_series().dt.time == current_time]

# Loop to fetch data for the next 20 minutes
end_datetime = pd.to_datetime("2023-12-04 03:59")
next_datetime = current_datetime
td = timedelta(minutes=20)
total_td = timedelta(minutes=0)
while next_datetime <= end_datetime:
    # Increment current_time by 20 minutes
    next_datetime += td
    next_time = next_datetime.time()
    total_td += td
    # Filter data from old_tb for the current time
    nxt = old_tb[old_tb.index.to_series().dt.time == next_time]
    diff = nxt - cur.shift(freq=total_td)
    mean_diff = pd.pivot_table(diff.mean().reset_index(), columns="sno")
    mean_diff.set_index(cur_data.index, inplace=True)
    upd = cur_data + mean_diff
    # print(upd)
    patch_datetime = next_datetime + timedelta(minutes=1)
    patch_time = patch_datetime.time()
    upd.set_index([[patch_time], [0]], inplace=True)
    y_private_df.loc[(patch_time, 0)] = upd
print(y_private_df)

y_private_test = y_private_df.values
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

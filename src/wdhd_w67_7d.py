from datetime import datetime
from operator import le
from matplotlib import legend
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import *
from median_optimization import optimal_median

TRAIN_START = "2023-10-02 00:00"
TRAIN_END = "2023-12-10 23:59"

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

# include 6 7
holidays = [d for d in date_range(start=TRAIN_START, end=PRIVARE_END) if is_holiday(d)]

tb = (
    pd.pivot_table(df, index="time", columns="sno", values="sbi")
    .resample("20min")
    .agg("first")
)

test = tb[tb.index.to_series().dt.date.isin(date_range(TEST_START, TEST_END))]
y_test = test.values

# TODO wdhd
train = tb[tb.index.to_series().dt.date.isin(date_range(TRAIN_START, TRAIN_END))].copy()
train.reset_index(names="time", inplace=True)
train["holiday"] = train["time"].dt.date.apply(is_holiday)
train.set_index(["time", "holiday"], inplace=True)

wdhd_df = pd.DataFrame(
    columns=ntu_snos,
    index=pd.MultiIndex.from_product(
        [pd.date_range("00:00", "23:59", freq="20min").time, [True, False]],
        names=("time", "holiday"),
    ),
    dtype=np.float64,
)

Ein = 0.0
for sno, tot in zip(ntu_snos, ntu_tots):
    sd = train[sno].to_frame()
    sd.rename(columns={sno: "sbi"}, inplace=True)
    sd.reset_index(["time", "holiday"], inplace=True)
    sd["date"] = sd["time"].dt.date
    sd["time"] = sd["time"].dt.time
    # print(sd)
    # exit()
    psd = pd.pivot_table(sd, index=["date", "holiday"], columns="time", values="sbi")
    # sno col have its sbi
    # print(psd)
    for holiday in [False, True]:
        for t in psd.columns:
            # print(t, sno)
            sbi, err = optimal_median(
                y_true=psd.loc[
                    psd.index.get_level_values("holiday") == holiday, t
                ].values,
                tot=tot,
            )  # majority of sbi are int
            Ein += err
            wdhd_df.at[(t, holiday), sno] = sbi

Ein /= wdhd_df.size
print_time_ranges(TRAIN_START, TRAIN_END, TEST_START, TEST_END)
print("wdhd")
print(f"Ein = {Ein}")

ftr = list(
    np.stack([test.index.time, test.index.to_series().dt.date.apply(is_holiday)]).T
)
y_pred = wdhd_df.loc[ftr].values
local_test_range = pd.date_range(TEST_START, TEST_END, freq="20min")
evaluation(y_test, y_pred, ntu_tots, local_test_range,prefix="wdhd")


# TODO 7d
tb = tb[~tb.index.to_series().dt.date.isin(long_holiday)]

train = tb[tb.index.to_series().dt.date.isin(date_range(TRAIN_START, TRAIN_END))].copy()
train.reset_index(names="time", inplace=True)
train["weekday"] = train["time"].dt.weekday
train.set_index(["time", "weekday"], inplace=True)

allday_df = pd.DataFrame(
    columns=ntu_snos,
    index=pd.MultiIndex.from_product(
        [pd.date_range("00:00", "23:59", freq="20min").time, range(7)],
        names=("time", "weekday"),
    ),
    dtype=np.float64,
)

Ein = 0.0
for sno, tot in zip(ntu_snos, ntu_tots):
    sd = train[sno].to_frame()
    sd.rename(columns={sno: "sbi"}, inplace=True)
    sd.reset_index(["time", "weekday"], inplace=True)
    sd["date"] = sd["time"].dt.date
    sd["time"] = sd["time"].dt.time
    psd = pd.pivot_table(sd, index=["date", "weekday"], columns="time", values="sbi")
    for day in range(7):  # 0 to 6
        for t in psd.columns:
            sbi, err = optimal_median(
                y_true=psd.loc[psd.index.get_level_values("weekday") == day, t].values,
                tot=tot,
            )
            Ein += err
            allday_df.at[(t, day), sno] = sbi

Ein /= allday_df.size
print_time_ranges(TRAIN_START, TRAIN_END, TEST_START, TEST_END)
print("7d")
print(f"Ein = {Ein}")

ftr = list(np.stack([test.index.time, test.index.to_series().dt.weekday]).T)
y_pred = allday_df.loc[ftr].values
local_test_range = pd.date_range(TEST_START, TEST_END, freq="20min")

evaluation(y_test, y_pred, ntu_tots, local_test_range,prefix="7d")


# TODO wd67
train = tb[tb.index.to_series().dt.date.isin(date_range(TRAIN_START, TRAIN_END))].copy()
train.reset_index(names="time", inplace=True)
train["weekday"] = train["time"].dt.weekday
train.set_index(["time", "weekday"], inplace=True)

wd67_df = pd.DataFrame(
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
    psd = pd.pivot_table(sd, index=["date", "weekday"], columns="time", values="sbi")
    # sno col have its sbi
    for day in [0, 5, 6]:  # 0 to 6
        for t in psd.columns:
            sbi, err = 0, 0
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
            wd67_df.at[(t, day), sno] = sbi

Ein /= wd67_df.size
print_time_ranges(TRAIN_START, TRAIN_END, TEST_START, TEST_END)
print("wd67")
print(f"Ein = {Ein}")

def trans(s):
    if s in range(5):
        return 0
    return s

ftr = list(
    np.stack([test.index.time, test.index.to_series().dt.weekday.apply(trans)]).T
)
y_pred = wd67_df.loc[ftr].values
local_test_range = pd.date_range(TEST_START, TEST_END, freq="20min")
evaluation(y_test, y_pred, ntu_tots, local_test_range,prefix="wd67")

exit()
# TODO plot all pred

df["datetime"] = df["time"]
df["date"] = df["datetime"].dt.date
df["time"] = df["datetime"].dt.time

for sno in ntu_snos:
    sd = df[df["sno"] == sno]
    for d in range(7):
        day_df = sd[sd["datetime"].dt.weekday == d]
        table = pd.pivot_table(day_df, values="sbi", index="time", columns="date")
        ax = table.plot(figsize=(14, 4), color="cyan", lw=0.8, alpha=0.4,  legend=False)
        mean = table.mean(axis=1)
        ax = mean.plot(ax=ax, color="red", lw=2, legend=False)
        std = table.std(axis=1)
        ax = std.plot(ax=ax, color="orange", legend=False)
        # pred
        ishoilday = True
        if d in range(5):
            ishoilday=False
        wdhd_tb = wdhd_df.loc[wdhd_df.index.get_level_values('holiday')==ishoilday,f"{sno}"].droplevel("holiday")
        ax =  wdhd_tb.plot(ax=ax, color="magenta", lw=2, legend=False)
        wd67_tb =  wd67_df.loc[wd67_df.index.get_level_values('weekday')== (0 if not ishoilday else d),f"{sno}"].droplevel('weekday')
        ax =  wd67_tb.plot(ax=ax, color="green", lw=2, legend=False)

        allday_tb = allday_df.loc[allday_df.index.get_level_values('weekday')==d,f"{sno}"].droplevel('weekday')
        ax =  allday_tb.plot(ax=ax, color="blue", lw=2, legend=False)

        plt.savefig(f"./lines/{sno}-{d+1}.png")
        plt.close(ax.get_figure())

holiday_df = df[df["date"].isin(holidays)]
plot_line(holiday_df, ntu_snos, name_suffix="holiday")
weekday_df = df[~df["date"].isin(holidays)]
plot_line(weekday_df, ntu_snos, name_suffix="weekday")


exit()



# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range(PUBLIC_START, PUBLIC_END, freq="20min")
# list makes indexer 1D, or it is an 2D indexer
ftr = list(
    np.stack([public_test_range.time, np.vectorize(trans)(public_test_range.weekday)]).T
)
# print(ftr)
# ftr = list(ftr)
# print(ftr)
y_public_test = result_df.loc[ftr].values
public_test_df = pd.DataFrame(y_public_test, columns=ntu_snos, index=public_test_range)

# we haven't do this yet, but it is required for submission
private_test_range = pd.date_range(PRIVATE_START, PRIVARE_END, freq="20min")
ftr = list(
    np.stack(
        [private_test_range.time, np.vectorize(trans)(private_test_range.weekday)]
    ).T
)
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

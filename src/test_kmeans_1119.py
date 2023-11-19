from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from tqdm import trange
import torch

DATA_DIR = Path("../html.2023.final.data")
DEMOGRAPHICS_PATH = DATA_DIR / "demographic.json"
RELEASE_DIR = DATA_DIR / "release"
SNO_TEST_SET = DATA_DIR / "sno_test_set.txt"


def load_demographics(path: Path):
    with open(path) as f:
        demo = json.load(f)  # { key: DATA }
        ar = []
        for k, v in demo.items():
            v["sno"] = k  # station number
            ar.append(v)
    return pd.DataFrame(ar)


def load_data_file(path: Path, base_date="20231001"):
    with open(path) as f:
        data = json.load(f)
        ar = []
        for k, v in data.items():
            t = pd.to_datetime(f"{base_date} {k}")
            v["time"] = t
            v["sno"] = path.stem
            ar.append(v)
    return pd.DataFrame(ar).bfill()  # fill NaN with next value


def load_all_data(demographics, cache_path=Path("all_data_cache.pkl")):
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    with ProcessPoolExecutor(max_workers=8) as executor:
        df_over_dates = []
        for date_dir in sorted(list(RELEASE_DIR.iterdir())):
            date = date_dir.name
            files = [
                file
                for sno in demographics["sno"]
                if (file := date_dir / f"{sno}.json").exists()
            ]
            not_exist = [
                sno
                for sno in demographics["sno"]
                if (date_dir / f"{sno}.json").exists() is False
            ]
            print(f"{date} not exist: {not_exist}")
            dates = [date] * len(files)
            results = executor.map(load_data_file, files, dates)
            dfs = list(tqdm(results, total=len(files), desc=f"Loading {date}"))
            df = pd.concat(dfs)
            df_over_dates.append(df)
    df = pd.concat(df_over_dates)
    df.to_pickle(cache_path)
    return df


def get_station_sno_df(df, snos):
    ret = df[df["sno"].isin(snos)].groupby("sno")["tot"].first().reset_index()
    ret.set_index("sno", inplace=True)
    ret.columns = ["tot"]
    return ret


ntu_snos = [l.strip() for l in open(SNO_TEST_SET).read().splitlines()]

demographics = load_demographics(DEMOGRAPHICS_PATH)
df = load_all_data(demographics)
station_sno_df = get_station_sno_df(df, ntu_snos)
print(
    "Amount of data points by stations", df.groupby(by="sno")["time"].size().describe()
)
print("Number of stations", df["sno"].nunique())
assert df.groupby(by="sno")["time"].is_monotonic_increasing.all(), "WTF?!"

bad_station = [
    "500105087",
    "500108169",
    "500108170",
]  # these two stations are added very late, so we remove them for simplicity
df = df[~df["sno"].isin(bad_station)]

N_STATIONS = df["sno"].nunique()

# the data looks like this:
"""
                    time        sno   tot   sbi  bemp act
0    2023-10-02 00:00:00  500101001  28.0  12.0  16.0   1
1    2023-10-02 00:01:00  500101001  28.0  12.0  16.0   1
2    2023-10-02 00:02:00  500101001  28.0  13.0  15.0   1
...
"""


1 / 0

# sno is the station number
# and we want to predict the value of sbi at time t+1


# dataframe for stations' bike number at time t
dfp = df.pivot(index="time", columns="sno", values="sbi").bfill()
dfp = dfp[ntu_snos]
time_split = pd.to_datetime("20231107 23:59:00")
train = dfp[dfp.index <= time_split].copy()
test = dfp[dfp.index > time_split].copy()

# dataframe for each day's properties (not used currently)
long_holiday = pd.date_range(start="2023-10-07", end="2023-10-10")
datetime_range = pd.date_range("2023/10/01 00:00", "2023/11/30 23:59", freq="min")
datetime_df = pd.DataFrame(
    {
        "is_holiday": datetime_range.isin(long_holiday)
        | datetime_range.weekday.isin((5, 6)),
    },
    index=datetime_range,
)

# dataframe for time at each day
day_time_20 = pd.date_range("00:00", "23:59", freq="20min")
day_time_20_df = pd.DataFrame(
    {
        "hour": day_time_20.hour,
        "minute": day_time_20.minute,
    }
)

to_group_df = station_sno_df.copy().reset_index()
to_group_df = to_group_df.merge(day_time_20_df, how="cross")
to_group_df.set_index(["sno", "hour", "minute"], inplace=True)

# written by GPT-4 :)

# Melt train DataFrame to long format
long_train = train.reset_index().melt(
    id_vars="time", var_name="sno", value_name="value"
)

# Extract hour and minute from time
long_train["hour"] = long_train["time"].dt.hour
long_train["minute"] = (
    long_train["time"].dt.minute // 20
) * 20  # Grouping minutes into 20-min intervals

# Group by sno, hour, and minute, then calculate mean and std
aggregated_train = long_train.groupby(["sno", "hour", "minute"]).agg(["mean", "std"])[
    "value"
]
aggregated_train.columns = ["mean", "std"]

# Merge with to_group_df
to_group_df = to_group_df.merge(
    aggregated_train, left_index=True, right_index=True, how="left"
)

# end of GPT-4's work

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=300, random_state=0, n_init="auto")
to_group_df["group"] = kmeans.fit_predict(to_group_df[["tot", "mean", "std"]])

# this is "training" result
# only mean_of_mean will be used for prediction
# std_of_mean (in group variance) is used for estimating whether clustering is good
group_df = pd.DataFrame(
    {
        "mean_of_mean": to_group_df.groupby("group")["mean"].mean(),
        "std_of_mean": to_group_df.groupby("group")["mean"].std(),
    }
)
print(group_df)

# doing prediction for test set


def get_prediction(time_range):
    time_range = time_range.to_series().resample("20min").agg("first")
    tmp_df = pd.DataFrame(
        {
            "time": time_range,
            "hour": time_range.dt.hour,
            "minute": time_range.dt.minute,
        }
    ).merge(pd.Series(ntu_snos, name="sno"), how="cross")
    tmp_df = tmp_df.merge(station_sno_df, how="left", on="sno")
    tmp_df.set_index(["sno", "time", "hour", "minute"], inplace=True)
    tmp_df = tmp_df.merge(
        aggregated_train, left_index=True, right_index=True, how="left"
    )
    tmp_df["group"] = kmeans.predict(tmp_df[["tot", "mean", "std"]])
    tmp_df = tmp_df.reset_index().merge(group_df, how="left", on="group")
    tmp_df = tmp_df[["time", "sno", "mean_of_mean"]]
    tmp_df.columns = ["time", "sno", "sbi"]
    return tmp_df.sort_values(by=["sno", "time"], ignore_index=True)


test_pred = get_prediction(test.index)
test_true = (
    test.resample("20min")
    .agg("first")
    .reset_index()
    .melt(id_vars="time", var_name="sno", value_name="sbi")
    .sort_values(by=["sno", "time"], ignore_index=True)
)


def evaluation(y_true, y_pred, df_):
    print("MAE", mean_absolute_error(y_true, y_pred))
    sarr = station_sno_df.loc[df_["sno"]].values.reshape(-1)
    err = (
        3
        * np.abs((y_pred - y_true) / sarr)
        * (np.abs(y_true / sarr - 1 / 3) + np.abs(y_true / sarr - 2 / 3))
    )
    print("Score", err.mean())


evaluation(
    test_true["sbi"].values,
    test_pred["sbi"].values,
    test_true,
)

"""
MAE 4.817353558541193
Score 0.397607157557498
"""

# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range("2023/10/21 00:00", "2023/10/24 23:59", freq="20min")
public_test_predict = get_prediction(public_test_range)
private_test_range = pd.date_range("2023/12/04 00:00", "2023/12/10 23:59", freq="20min")
private_test_predict = get_prediction(private_test_range)


tmp = pd.concat([public_test_predict, private_test_predict])
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
out_df.to_csv("public_test_submission_1119_2146.csv", index=False)

from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor

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


def get_stops_number(df, snos):
    result_df = df[df["sno"].isin(snos)].groupby("sno")["tot"].first().reset_index()
    result_df.set_index("sno", inplace=True)
    result_df.columns = ["tot"]
    s_arr = result_df.values.reshape(-1)
    return s_arr


ntu_snos = [l.strip() for l in open(SNO_TEST_SET).read().splitlines()]

demographics = load_demographics(DEMOGRAPHICS_PATH)
df = load_all_data(demographics)
ntu_stops = get_stops_number(df, ntu_snos)
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
# sno is the station number
# and we want to predict the value of sbi at time t+1


# Prepare the data for linear regression
"""
        sno     500101001 500101002 ...
time                
10-02 00:00:00      sbi     sbi
...
"""
dfp = df.pivot(index="time", columns="sno", values="sbi").bfill()
dfp = dfp[ntu_snos]
time_split = pd.to_datetime("20231020 23:59:00")
train = dfp[dfp.index <= time_split]
test = dfp[dfp.index > time_split]


# mean by hour
train_mean = train.groupby(by=train.index.hour).mean()

train_mean_2 = train.groupby(
    by=[train.index.hour, train.index.weekday.isin((5, 6))]
).mean()

train_mean_3 = train.groupby(by=train.index.strftime("%H:%M")).mean()


"""
(nparray)
        sno     500101001 500101002 ...
time                
10-02 00:00:00      sbi     sbi
...
"""


def evaluation(y_true, y_pred):
    print("MAE", mean_absolute_error(y_true, y_pred))

    # yt is a row of many station's sbi
    errors = np.array([mean_absolute_error(yt, yp) for yt, yp in zip(y_true, y_pred)])
    xs = np.arange(len(errors))
    plt.plot(xs, errors)
    plt.show()

    # errors2 = np.abs(y_pred - y_true).reshape(-1)
    # counts, edges, bars = plt.hist(errors2, bins=len(set(errors2)))
    # plt.bar_label(bars)
    # plt.show()

    # error function defined in the problem description
    err = (
        3
        * np.abs((y_pred - y_true) / ntu_stops)
        * (np.abs(y_true / ntu_stops - 1 / 3) + np.abs(y_true / ntu_stops - 2 / 3))
    )
    print("Score", err.mean())


y_test = test[ntu_snos].values
# time x sno
"""
        sno     500101001 500101002 ...
time                
10-02 00:00:00      sbi     sbi
...
"""
# y_pred = train_mean.loc[test.index.hour].values
ftr = list(np.stack([test.index.hour.values, test.index.weekday.isin((5, 6))]).T)
"""
 [
  day1:
    (hour, is_weekend) 
  day2:
    (hour, is_weekend) 
  day3:
    (hour, is_weekend) 
 ]
"""

# after apply mean() it becomes a dataframe
# so this is (hour, is_weekend) x sno
# it will repeatly select the same (hour, is_weekend)
y_pred = train_mean_2.loc[ftr].values

# y_pred = train_mean_3.loc[test.index.strftime('%H:%M')].values
print(y_pred.shape, y_test.shape)
evaluation(y_test, y_pred)
"""
MAE 5.149000804436923
Score 0.48484917661484556
"""
"""
MAE 5.051879777485238
Score 0.4740869384380755
"""
"""
MAE 5.108978077721973
Score 0.4804552008227539
"""

# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range("2023/10/21 00:00", "2023/10/24 23:59", freq="min")
ftr = list(
    np.stack([public_test_range.hour.values, public_test_range.weekday.isin((5, 6))]).T
)
y_public_test = train_mean_2.loc[ftr].values
public_test_df = pd.DataFrame(y_public_test, columns=ntu_snos, index=public_test_range)

# we haven't do this yet, but it is required for submission
private_test_range = pd.date_range("2023/12/04 00:00", "2023/12/10 23:59", freq="min")
private_test_df = pd.DataFrame(0, columns=ntu_snos, index=private_test_range)


# convert the prediction to the required format
tmp = pd.concat(
    [
        public_test_df.resample("20min").agg("first"),
        private_test_df.resample("20min").agg("first"),
    ]
)
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
out_df.to_csv("public_test_submission_1104_2138.csv", index=False)

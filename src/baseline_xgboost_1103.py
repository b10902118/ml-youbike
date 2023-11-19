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
dfp = df.pivot(index="time", columns="sno", values="sbi").bfill()
dfp = dfp[ntu_snos]
dfp["min_of_day"] = dfp.index.hour * 60 + dfp.index.minute
input_features = ntu_snos + ["min_of_day"]
time_split = pd.to_datetime("20231020 23:59:00")
train = dfp[dfp.index <= time_split]
test = dfp[dfp.index > time_split]
X = train[input_features].values[:-1]
y = train[ntu_snos].values[1:]
print(X.shape, y.shape)
print("FIT")
model = XGBRegressor()
model.fit(X, y)
print("FIT DONE")


def evaluation(y_true, y_pred):
    print("MAE", mean_absolute_error(y_true, y_pred))

    errors = np.array([mean_absolute_error(yt, yp) for yt, yp in zip(y_true, y_pred)])
    xs = np.arange(len(errors))
    plt.plot(xs, errors)
    plt.show()
    # plt.savefig("baseline-xgboost.png")

    errors2 = np.abs(y_pred - y_true).reshape(-1)
    counts, edges, bars = plt.hist(errors2, bins=len(set(errors2)))
    plt.bar_label(bars)
    plt.show()

    # error function defined in the problem description
    err = (
        3
        * np.abs((y_pred - y_true) / ntu_stops)
        * (np.abs(y_true / ntu_stops - 1 / 3) + np.abs(y_true / ntu_stops - 2 / 3))
    )
    print("Score", err.mean())


# Predict the test data (wrong)
# this part is meaningless
X_test = test[input_features].values[:-1]
y_test = test[ntu_snos].values[1:]
y_pred = model.predict(X_test).round()
print(y_pred.shape, y_test.shape)
evaluation(y_test, y_pred)
"""
MAE 0.3973489452209393
Score 0.03549132425707825
"""


def next_day(inp):
    return model.predict(inp.reshape(1, -1)).round()[0]


# use the first data point of test data as the initial state
# and predict the next day iteratively
state = test[ntu_snos].iloc[0].values
out = []
for i in tqdm(range(len(test) - 1)):
    out.append(state.round())
    state = next_day(np.insert(state, -1, i % 1440))
y_pred_next = np.array(out, dtype=np.int32)
evaluation(y_test, y_pred_next)
"""
MAE 5.811096263316508
Score 0.4270636654857978
"""

# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range("2023/10/21 00:00", "2023/10/24 23:59", freq="min")
state = train[ntu_snos].loc["20231020 23:59"].values
out = []
for t in tqdm(public_test_range):
    t_min = t.hour * 60 + t.minute
    state = next_day(np.insert(state, -1, t_min))
    out.append(state.round())
y_public_test = np.array(out, dtype=np.int32)
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
out_df.to_csv("public_test_submission_1103_1558.csv", index=False)

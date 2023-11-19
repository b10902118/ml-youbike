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
    return pd.DataFrame(ar).set_index("sno")


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


def get_tot_by_sno(df, snos):
    result_df = df[df["sno"].isin(snos)].groupby("sno")["tot"].first().reset_index()
    result_df.set_index("sno", inplace=True)
    result_df.columns = ["tot"]
    s_arr = result_df.values.reshape(-1)
    return result_df


ntu_snos = [l.strip() for l in open(SNO_TEST_SET).read().splitlines()]

demographics = load_demographics(DEMOGRAPHICS_PATH)
df = load_all_data(demographics)
sno_tot = get_tot_by_sno(df, ntu_snos)
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

long_holiday = pd.date_range(start="2023-10-07", end="2023-10-10")

# Prepare the data for linear regression
dfp = df.pivot(index="time", columns="sno", values="sbi").bfill()
dfp = dfp[ntu_snos]
time_split = pd.to_datetime("20231020 23:59:00")
train = dfp[dfp.index <= time_split].copy()
test = dfp[dfp.index > time_split].copy()

train["min_of_day"] = train.index.hour * 60 + train.index.minute
train["is_holiday"] = (
    train.index.weekday.isin((5, 6)) | train.index.isin(long_holiday)
).astype(int)
test["min_of_day"] = test.index.hour * 60 + test.index.minute
test["is_holiday"] = (
    test.index.weekday.isin((5, 6)) | test.index.isin(long_holiday)
).astype(int)

feature_columns = ["min_of_day", "is_holiday"]

train_ = train.melt(id_vars=feature_columns, var_name="sno", value_name="sbi")
test_ = test.melt(id_vars=feature_columns, var_name="sno", value_name="sbi")

train_["lat"] = demographics.loc[train_["sno"]]["lat"].reset_index(drop=True)
train_["lng"] = demographics.loc[train_["sno"]]["lng"].reset_index(drop=True)
test_["lat"] = demographics.loc[test_["sno"]]["lat"].reset_index(drop=True)
test_["lng"] = demographics.loc[test_["sno"]]["lng"].reset_index(drop=True)

feature_columns += ["lat", "lng"]

x_train = train_[feature_columns].values
y_train = train_["sbi"].values
x_test = test_[feature_columns].values
y_test = test_["sbi"].values


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.predict(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(n_feature=x_train.shape[1], n_hidden1=32, n_hidden2=16, n_output=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

sarr_tensor = (
    torch.tensor(sno_tot.loc[train_["sno"]].values.reshape(-1))
    .reshape(-1, 1)
    .to(device)
)

model.train()
for t in trange(500):
    prediction = model(x_train_tensor)
    loss = (
        3
        * torch.abs((prediction - y_train_tensor) / sarr_tensor)
        * (
            torch.abs(y_train_tensor / sarr_tensor - 1 / 3)
            + torch.abs(y_train_tensor / sarr_tensor - 2 / 3)
        )
    ).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 100 == 0:
        print(loss.item())


def evaluation(y_true, y_pred, df_):
    print("MAE", mean_absolute_error(y_true, y_pred))
    sarr = sno_tot.loc[df_["sno"]].values.reshape(-1)
    err = (
        3
        * np.abs((y_pred - y_true) / sarr)
        * (np.abs(y_true / sarr - 1 / 3) + np.abs(y_true / sarr - 2 / 3))
    )
    print("Score", err.mean())


model.eval()
with torch.no_grad():
    y_train_pred = model(x_train_tensor).detach().cpu().numpy().reshape(-1)
    y_test_pred = model(x_test_tensor).detach().cpu().numpy().reshape(-1)
evaluation(y_train, y_train_pred, train_)
evaluation(y_test, y_test_pred, test_)

"""
MAE 7.740782209025374
Score 0.5527443398229321
MAE 5.186690773595576
Score 0.380102024540055
"""

# does the same at public test set (2023/10/21 - 2023/10/24)
public_test_range = pd.date_range("2023/10/21 00:00", "2023/10/24 23:59", freq="min")
public_test_df = pd.concat(
    [
        public_test_range.to_series().dt.hour * 60
        + public_test_range.to_series().dt.minute,
        (
            public_test_range.to_series().dt.weekday.isin((5, 6))
            | public_test_range.to_series().isin(long_holiday)
        ).astype(int),
    ],
    axis=1,
)
public_test_df.columns = ["min_of_day", "is_holiday"]
x_public_test = public_test_df.values
y_public_test = (
    model(torch.tensor(x_public_test, dtype=torch.float32).to(device))
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
)
pred = pd.Series(y_public_test, index=public_test_range, name="pred")
public_test_df = pd.concat([pred.rename(sno) for sno in ntu_snos], axis=1)

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
out_df.to_csv("public_test_submission_1110_1848.csv", index=False)

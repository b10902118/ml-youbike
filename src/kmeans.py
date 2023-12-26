from pathlib import Path
import pandas as pd
import ujson as json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_DIR = Path("../html.2023.final.data")
DEMOGRAPHICS_PATH = DATA_DIR / "demographic.json"
RELEASE_DIR = DATA_DIR / "release"
SNO_TEST_SET = DATA_DIR / "sno_test_set.txt"
SMALL_CACHE_DIR = Path("./cache")


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
    return pd.DataFrame(ar).bfill(limit=3)  # fill NaN with next value


def load_all_data():
    df = pd.read_pickle("./cache/small_data_cache.pkl")
    return df


def get_station_sno_df(df, snos):
    ret = df[df["sno"].isin(snos)].groupby("sno")["tot"].first().reset_index()
    ret.set_index("sno", inplace=True)
    ret.columns = ["tot"]
    return ret


# TRAIN_END = "2023-10-30 23:59"
# TEST_START = "2023-12-01 00:00"
# TEST_END = "2023-12-07 23:59"
def train_and_predict(TRAIN_START, TRAIN_END, TEST_START, TEST_END):
    ntu_snos = [l.strip() for l in open(SNO_TEST_SET).read().splitlines()]

    # print("Loading data...")
    df = load_all_data()
    df = df[df["act"] == "1"]  # filter out inactive stations
    station_sno_df = get_station_sno_df(df, ntu_snos)
    # print(
    #    "Amount of data points by stations",
    #    df.groupby(by="sno")["time"].size().describe(),
    # )
    # print("Number of stations", df["sno"].nunique())
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

    # dataframe for stations' bike number at time t
    dfp = df.pivot(index="time", columns="sno", values="sbi")
    never_active_snos = set(ntu_snos) - set(dfp.columns)
    # preserve order
    ntu_snos = [sno for sno in ntu_snos if sno not in never_active_snos]
    # ntu_snos = list(set(ntu_snos) - never_active_snos)
    dfp = dfp[ntu_snos]
    time_split = pd.to_datetime(TRAIN_END)
    train = dfp[dfp.index <= time_split].copy()
    # train = dfp[(dfp.index >= pd.to_datetime("20231101 00:00:00")) & (dfp.index <= time_split)]
    test = dfp[(dfp.index >= TEST_START) & (dfp.index <= TEST_END)].copy()

    # use full data for training to get better performance on public test set
    train = dfp.copy()

    # special holidays (only for training data as my method can't handle special holidays in weekday that is not in training data)
    long_holiday = (
        pd.date_range(start="2023-10-07", end="2023-10-10")
        # .union(pd.date_range(start="2023-11-15", end="2023-11-15"))
        # .union(pd.date_range(start="2023-11-24", end="2023-11-24"))
    )

    def is_holiday(s):
        return s.dt.weekday.isin((5, 6))
        return s.isin(long_holiday) | s.dt.weekday.isin((5, 6))

    def quantile(x):
        return lambda y: y.quantile(x)

    base_index_names = [
        "sno",
        "weekday",
        "hour",
        "minute",
    ]
    index_names = base_index_names + ["is_holiday"]
    property_names = [
        "mean_20",
        "std_20",
        # "var_20",
        # "skew_20",
        # "kurt_20",
        "q25_20",
        "q50_20",
        "q75_20",
        "mean_sta",
        "std_sta",
        # "var_sta",
        # "skew_sta",
        # "kurt_sta",
        # "mean_1h",
        # "std_1h",
    ]

    # datetime_range = pd.date_range("2023/10/01 00:00", "2023/11/30 23:59", freq="min")
    # datetime_df = pd.DataFrame(
    #     {"is_holiday": is_holiday(datetime_range.to_series())},
    #     index=datetime_range,
    # )

    # to_group_df's index is `index_names` a.k.a. input
    # to_group_df's columns are `property_names` a.k.a. properties

    def prepare_data(train):
        # dataframe for time at each day
        day_time_20 = pd.date_range(
            "2023/11/05 00:00", "2023/11/11 23:59", freq="20min"
        )
        day_time_20_df = pd.DataFrame(
            {
                "weekday": day_time_20.weekday,
                "hour": day_time_20.hour,
                "minute": day_time_20.minute,
            }
        )

        to_group_df = station_sno_df.copy().reset_index()
        to_group_df = (
            to_group_df.merge(day_time_20_df, how="cross")
            # .merge(pd.DataFrame({"is_holiday": [0, 1]}), how="cross")
            # .merge(pd.DataFrame({"is_daytime": [0, 1]}), how="cross")
        )
        to_group_df.set_index(base_index_names, inplace=True)

        # Melt train DataFrame to long format
        long_train = (
            train.reset_index()
            .melt(id_vars="time", var_name="sno", value_name="sbi")
            .dropna()
        )

        # Extract hour and minute from time
        long_train["weekday"] = long_train["time"].dt.weekday
        long_train["hour"] = long_train["time"].dt.hour
        long_train["minute"] = (
            long_train["time"].dt.minute // 20
        ) * 20  # Grouping minutes into 20-min intervals
        long_train["is_holiday"] = is_holiday(long_train["time"])

        rg = long_train.set_index("time").groupby("sno").rolling("1D")
        long_train["sbi_roll_1h"] = rg["sbi"].mean().values

        # Group by sno, hour, and minute, then calculate mean and std
        aggregated_train = (
            long_train.groupby(index_names)
            .agg(
                mean_20=("sbi", "mean"),
                std_20=("sbi", "std"),
                var_20=("sbi", "var"),
                skew_20=("sbi", "skew"),
                kurt_20=("sbi", lambda x: x.kurt()),
                q25_20=("sbi", quantile(0.25)),
                q50_20=("sbi", quantile(0.5)),
                q75_20=("sbi", quantile(0.75)),
                mean_1h=("sbi_roll_1h", "mean"),
                std_1h=("sbi_roll_1h", "std"),
            )
            .fillna(0)
        )
        aggregated_train = (
            aggregated_train.reset_index()
            .merge(
                long_train.groupby("sno").agg(
                    mean_sta=("sbi", "mean"),
                    std_sta=("sbi", "std"),
                    var_sta=("sbi", "var"),
                    skew_sta=("sbi", "skew"),
                    kurt_sta=("sbi", lambda x: x.kurt()),
                ),
                on="sno",
            )
            .set_index(index_names)
        )

        # Merge with to_group_df
        to_group_df = to_group_df.merge(
            aggregated_train, left_index=True, right_index=True, how="left"
        )
        return long_train, aggregated_train, to_group_df

    def error(y_true: np.ndarray, y_pred: np.ndarray, tots: np.ndarray) -> np.float64:
        return 3 * np.dot(
            np.abs((y_pred - y_true) / tots),
            np.abs(y_true / tots - 1 / 3) + np.abs(y_true / tots - 2 / 3),
        )

    def brute(
        y_true: np.ndarray, tots: np.ndarray, step: np.float64 = 0.5
    ) -> (np.float64, np.float64):
        tots = tots[~np.isnan(y_true)]
        y_true = y_true[~np.isnan(y_true)]
        arr_len = y_true.shape[0]
        assert arr_len == tots.shape[0]

        best_sbi = 0.0
        best_err = 9999999999.0  # biggest

        for sbi in np.arange(0, max(tots), step):
            sbis = np.full(arr_len, sbi)
            err = error(y_true, sbis, tots)
            if err < best_err:
                best_sbi, best_err = sbi, err

        return best_sbi, best_err / arr_len

    def first_greater_prefix_sum_idx(arr, target):
        prefix_sum = 0

        for i, element in enumerate(arr):
            prefix_sum += element

            if prefix_sum > target:
                return i

    def optimal_median(y_true: np.ndarray, tot: int) -> (np.float64, np.float64):
        # assert len(y_true.shape) == 1, "optimal_median: shape error"
        # print(f"y_true: {y_true}")

        nan_indices = np.isnan(y_true)

        # Use boolean indexing to drop NaN values
        y_true = y_true[~nan_indices]

        arr_len = y_true.shape[0]
        tots = np.full(arr_len, tot)

        # generalized median
        y_sorted = np.sort(y_true)
        weight = np.abs(y_sorted / tot - 1 / 3) + np.abs(y_sorted / tot - 2 / 3)
        w_mid = np.sum(weight) / 2
        w_cur = 0

        # if odd, first > w_mid
        # if even, before m1 must less than w_mid, m2 must greater than w_mid
        best_sbi = y_sorted[first_greater_prefix_sum_idx(weight, w_mid)]
        best_err = error(y_true, np.full(arr_len, best_sbi), tots)

        return best_sbi, best_err

    def get_group_assignment_df(kmeans, time_range, aggregated_train):
        global tmp_df
        time_range = (
            time_range.to_series()
            .resample("20min")
            .agg("first")
            .dropna()
            .index.to_series()
        )
        tmp_df = pd.DataFrame(
            {
                "time": time_range,
                "weekday": time_range.dt.weekday,
                "hour": time_range.dt.hour,
                "minute": time_range.dt.minute,
                "is_holiday": is_holiday(time_range),
            }
        )
        tmp_df = tmp_df.merge(pd.Series(ntu_snos, name="sno"), how="cross")
        tmp_df = tmp_df.merge(station_sno_df, how="left", on="sno")
        tmp_df.set_index(["time"] + index_names, inplace=True)
        tmp_df = tmp_df.merge(
            aggregated_train, left_index=True, right_index=True, how="left"
        )
        tmp_df["group"] = kmeans.predict(tmp_df[property_names])
        return tmp_df

    def do_train(kmeans, t_df, long_train, n_clusters):
        # this is needed for brute force
        t_df = t_df.reset_index().merge(long_train, on=index_names + ["time"])

        group_df = pd.DataFrame({}, index=range(n_clusters))
        group_df.index.name = "group"
        # find best sbi for each group
        for grp_id in sorted(t_df["group"].unique()):
            ys = t_df[t_df["group"] == grp_id]["sbi"].values
            tots = t_df[t_df["group"] == grp_id]["tot"].values
            best_sbi, err = brute(ys, tots)
            # best_sbi, err = optimal_median(ys, tots)
            # print(
            #     f"group {grp_id}, {best_sbi = }, {err = }",
            # )
            group_df.loc[grp_id, "best_sbi"] = best_sbi
            group_df.loc[grp_id, "err"] = err
        return t_df, group_df

    def get_prediction(kmeans, time_range, group_df, aggregated_train):
        tmp_df = get_group_assignment_df(kmeans, time_range, aggregated_train)
        tmp_df = tmp_df.reset_index().merge(group_df, how="left", on="group")
        tmp_df = tmp_df[["time", "sno", "best_sbi"]]
        tmp_df.columns = ["time", "sno", "sbi"]
        return tmp_df.sort_values(by=["sno", "time"], ignore_index=True)

    def evaluation(y_true, y_pred, df_):
        print("MAE", mean_absolute_error(y_true, y_pred))
        sarr = station_sno_df.loc[df_["sno"]].values.reshape(-1)
        err = (
            3
            * np.abs((y_pred - y_true) / sarr)
            * (np.abs(y_true / sarr - 1 / 3) + np.abs(y_true / sarr - 2 / 3))
        )
        print("Score", err.mean())

    test_true = (
        test.resample("20min")
        .agg("first")
        .reset_index()
        .melt(id_vars="time", var_name="sno", value_name="sbi")
        .sort_values(by=["sno", "time"], ignore_index=True)
    )
    test_true["sbi"] = test_true["sbi"].bfill()
    test_true.set_index("time", inplace=True)

    long_train, aggregated_train, to_group_df = prepare_data(train)

    if long_train.isnull().any().any():
        # print("long_train has NaN")
        long_train.dropna(inplace=True)
    if aggregated_train.isnull().any().any():
        # print("aggregated_train has NaN")
        aggregated_train.dropna(inplace=True)
    if to_group_df.isnull().any().any():
        # print("to_group_df has NaN")
        to_group_df.dropna(inplace=True)

    n_clusters = 2000
    # print("=" * 40)
    # print(f"{n_clusters = }")
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    to_group_df["group"] = km.fit_predict(to_group_df[property_names])
    t_df = get_group_assignment_df(km, train.index, aggregated_train)
    t_df, group_df = do_train(km, t_df, long_train, n_clusters)
    # print(t_df.groupby("group").size().describe())
    # print(group_df["err"].describe())
    test_pred = get_prediction(km, test_true.index, group_df, aggregated_train)
    test_pred.set_index("time", inplace=True)
    # print(test_pred)

    # pivot test_pred by sno to columns
    test_pred_p = test_pred.pivot(columns="sno", values="sbi")
    test_pred_p = test_pred_p.sort_index(axis=0)
    for e in never_active_snos:
        test_pred_p[e] = 0

    # ensure that time index is sorted
    # sort test_pred_p by sno , don't use ntu_snos because some stations are never active
    test_pred_p = test_pred_p[sorted(test_pred_p.columns)]

    # print(test_pred_p)

    return test_pred_p

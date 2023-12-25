from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
from utils import *
from median_optimization import optimal_median
from scipy.spatial.distance import euclidean


def train_and_predict(TRAIN_START, TRAIN_END, TEST_START, TEST_END):
    # TRAIN_START = "2023-10-02 00:00"
    # TRAIN_END = "2023-12-17 23:59"
    # TEST_START = "2023-10-28 00:00"
    # TEST_END = "2023-10-31 23:59"

    with open("./cache/small_data_cache.pkl", "rb") as f:
        df = pd.read_pickle(f)
    with open("../html.2023.final.data/sno_test_set.txt") as f:
        ntu_snos = [l.strip() for l in f.read().splitlines()]
    with open("./cache/1003-1222_rain.pkl", "rb") as f:
        rain_df = pd.read_pickle(f)

    ntu_tots = get_tot(df, ntu_snos)

    holidays = [d for d in date_range(start=TRAIN_START, end=TEST_END) if is_holiday(d)]

    df["datehour"] = df["time"].dt.floor("H")
    rain_df.rename(columns={"datetime": "datehour"}, inplace=True)

    df = df.merge(rain_df, on="datehour", how="left")
    df["rain"].fillna(0, inplace=True)

    morning_filter = (df["datehour"].dt.hour >= 7) & (df["datehour"].dt.hour <= 21)
    rain_hours = df["datehour"][
        (df["sno"] == "500101001") & morning_filter & (df["rain"] >= 0.3)
    ].drop_duplicates()
    rain_dates = rain_hours.dt.date.drop_duplicates()

    date_rain_hour_cnt = rain_hours.dt.date.value_counts()

    long_rain_dates = rain_dates[
        rain_dates.isin(date_rain_hour_cnt.index[date_rain_hour_cnt >= 7])
    ]

    rainy_dates = long_rain_dates.array

    del rain_df
    del morning_filter
    del rain_hours, rain_dates
    del date_rain_hour_cnt
    del long_rain_dates

    tb = (
        pd.pivot_table(df, index="time", columns="sno", values="sbi")
        .resample("20min")
        .first()
    )

    def prep_train_data(tb) -> pd.DataFrame:
        train = tb[
            tb.index.to_series().dt.date.isin(date_range(TRAIN_START, TRAIN_END))
        ].copy()
        train.reset_index(names="time", inplace=True)
        train["weekday"] = train["time"].dt.weekday
        train.set_index(["time", "weekday"], inplace=True)
        return train

    def prep_sd_data(train, sno) -> pd.DataFrame:
        sd = train[sno].to_frame()
        sd.rename(columns={sno: "sbi"}, inplace=True)
        sd.reset_index(["time", "weekday"], inplace=True)
        sd["date"] = sd["time"].dt.date
        sd["time"] = sd["time"].dt.time
        sd = pd.pivot_table(sd, index=["date", "weekday"], columns="time", values="sbi")
        return sd

    train = prep_train_data(tb)

    result_df = pd.DataFrame(
        columns=ntu_snos,
        index=pd.MultiIndex.from_product(
            [
                [True, False],
                pd.date_range("00:00", "23:59", freq="20min").time,
                [d for d in range(7)],
            ],
            names=("rain", "time", "weekday"),
        ),
        dtype=np.float64,
    )

    Ein = {True: 0.0, False: 0.0}

    def find_specials(sd):
        grouped_stats = sd.droplevel("date").groupby(level=0).mean().drop([5, 6]).T
        time_series_data = grouped_stats.iloc[:, :5].values.T

        distance_matrix = np.zeros((5, 5))

        for i in range(5):
            for j in range(i):
                distance = euclidean(time_series_data[i], time_series_data[j])
                distance_matrix[i][j] = distance_matrix[j][i] = distance
        means = np.mean(distance_matrix, axis=1)
        std = np.std(means)
        mean = np.mean(means)

        specials = []
        for i in range(5):
            if np.abs(means[i] - mean) > 1.7 * std:
                specials.append(i)
        return specials

    for sno, tot in zip(ntu_snos, ntu_tots):
        sd = prep_sd_data(train, sno)

        specials = find_specials(sd) + [5, 6]
        normal = [d for d in range(7) if d not in specials]

        for rain in [True, False]:
            if not rain:
                sd = sd[~sd.index.get_level_values("date").isin(rainy_dates)]

            for day in specials:
                for t in sd.columns:
                    sbi, err = optimal_median(
                        y_true=sd.loc[
                            sd.index.get_level_values("weekday") == day, t
                        ].to_numpy(),
                        tot=tot,
                    )
                    Ein[rain] += err

                    result_df.at[(rain, t, day), sno] = sbi

            for t in sd.columns:
                sbi, err = optimal_median(
                    y_true=sd.loc[
                        sd.index.get_level_values("weekday").isin(normal), t
                    ].to_numpy(),
                    tot=tot,
                )

                for day in normal:
                    result_df.at[(rain, t, day), sno] = sbi
                    Ein[rain] += err

    assert not result_df.isnull().values.any()

    # for rain in [True, False]:
    #    Ein[rain] /= result_df.xs(rain).size
    #    print(("All" if rain else "Sunny") + f" Ein = {Ein[rain]}")

    test = tb[tb.index.to_series().dt.date.isin(date_range(TEST_START, TEST_END))]
    y_test = test.values

    # print_time_ranges(TRAIN_START, TRAIN_END, TEST_START, TEST_END)

    ftr = list(
        np.stack(
            [
                test.index.to_series().dt.date.isin(rainy_dates),
                test.index.time,
                test.index.to_series().dt.weekday,
            ]
        ).T
    )
    y_pred = result_df.loc[ftr]
    y_pred.reset_index(drop=True, inplace=True)
    y_pred.set_index(test.index, inplace=True)
    print(y_pred)
    # local_test_range = pd.date_range(TEST_START, TEST_END, freq="20min")
    # assert y_test.shape == y_pred.shape, "test pred shape not matched"
    return y_pred

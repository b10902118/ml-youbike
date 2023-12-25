from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import *
from median_optimization import optimal_median
from prophet import Prophet
import logging
import concurrent.futures

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def train_and_predict(sno, train_data, holidays_df, test_df):
    station_train = train_data[train_data["sno"] == sno]
    m = Prophet(holidays=holidays_df)
    m.add_regressor("rain", prior_scale=5)
    m.fit(station_train[["ds", "y", "rain"]])

    forecast = m.predict(test_df)

    return sno, forecast, m


def train_and_predict_prophet(TRAIN_START, TRAIN_END, TEST_START, TEST_END):
    # TRAIN_START = "2023-10-02 00:00"
    # TRAIN_END = "2023-12-17 23:59"
    # TEST_START = "2023-12-01 00:00"
    # TEST_END = "2023-12-14 23:59"
    # test_rain_dates = []

    with open("./cache/small_data_cache.pkl", "rb") as f:
        df = pd.read_pickle(f)
    with open("../html.2023.final.data/sno_test_set.txt") as f:
        ntu_snos = [l.strip() for l in f.read().splitlines()]
    with open("./cache/10-03_12_09_rain.pkl", "rb") as f:
        rain_df = pd.read_pickle(f)

    ntu_tots = get_tot(df, ntu_snos)

    """
        datetime               sno      tot   sbi   bemp  act
    0    2023-10-02 00:00:00  500101001  28.0  12.0  16.0   1
    1    2023-10-02 00:01:00  500101001  28.0  12.0  16.0   1
    2    2023-10-02 00:02:00  500101001  28.0  13.0  15.0   1
    ...
    """
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

    df["rain"] = df["time"].dt.date.isin(rainy_dates)

    del rain_df
    del morning_filter
    del rain_hours, rain_dates
    del date_rain_hour_cnt
    del long_rain_dates

    df.drop(columns=["tot", "datehour", "act"], inplace=True)
    df.rename(columns={"time": "ds", "sbi": "y"}, inplace=True)

    holidays_df = pd.DataFrame(
        {
            "holiday": "holiday",
            "ds": pd.to_datetime(holidays),
            "lower_window": 0,
            "upper_window": 1,
        }
    )

    train = df[df["ds"].dt.date.isin(date_range(TRAIN_START, TRAIN_END))].copy()
    pred_dfs = {}
    models = {}

    test_df = pd.DataFrame({"ds": pd.date_range(TEST_START, TEST_END, freq="20min")})
    test_df["rain"] = test_df["ds"].dt.date.isin(rainy_dates)

    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        future_to_sno = {
            executor.submit(train_and_predict, sno, train, holidays_df, test_df): sno
            for sno in ntu_snos
        }

        for future in concurrent.futures.as_completed(future_to_sno):
            sno = future_to_sno[future]
            try:
                result_sno, forecast_sno, model_sno = future.result()
                pred_dfs[result_sno] = forecast_sno
                models[result_sno] = model_sno
            except Exception as e:
                print(f"Error processing station {sno}: {e}")

    test_range = pd.date_range(TEST_START, TEST_END, freq="20min")
    test_len = len(list(test_range))
    test_df = df[df["ds"].isin(test_range)]
    test_tb = (
        pd.pivot_table(test_df, index="ds", columns="sno", values="y")
        .resample("20min")
        .agg("first")
        .bfill()
        .ffill()
    )

    # y_pred = np.empty([test_len, 0])
    y_pred = pd.DataFrame({"ds": test_range})

    # errors = {}
    for sno, tot in zip(ntu_snos, ntu_tots):
        pred_df = pred_dfs[sno]
        # m = models[sno]
        # fig = m.plot_components(pred_df)
        # plt.savefig(f"./prophet_lines/{sno}_components.png")
        # plt.close(fig)

        y_pred[sno] = pred_df["yhat"].clip(lower=0, upper=tot).to_numpy()
        # ans = test_tb[sno]

        # err = error(ans.to_numpy(), pred, np.full(test_len, tot))
        # errors[sno] = err

        # ax = pred_df.plot(x="ds", y="yhat", figsize=(20, 6), title=f"score: {err}")
        # ans.plot(ax=ax, x="ds", y="y")
        # plt.savefig(f"./prophet_lines/{sno}.png")
        # plt.close()
    y_pred.set_index("ds")
    print(y_pred)
    return y_pred

from datetime import time, date, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def print_time_ranges(TRAIN_START: str, TRAIN_END: str, TEST_START: str, TEST_END: str):
    print(
        f"Train: {pd.to_datetime(TRAIN_START).strftime('%m/%d')} - {pd.to_datetime(TRAIN_END).strftime('%m/%d')}"
    )
    print(
        f"Test: {pd.to_datetime(TEST_START).strftime('%m/%d')} - {pd.to_datetime(TEST_END).strftime('%m/%d')}"
    )


def plot_line(dataframe, snos, name_prefix=""):
    for sno in snos:
        station_data = dataframe[dataframe["sno"] == sno]
        table = pd.pivot_table(station_data, values="sbi", index="time", columns="date")
        # dfp = df.pivot(index="time", columns="date", values="sbi").bfill()
        ax = table.plot(figsize=(12, 3), color="blue", legend=False)
        mean = table.mean(axis=1)
        ax = mean.plot(ax=ax, color="red", lw=2, legend=False)
        std = table.std(axis=1)
        ax = std.plot(ax=ax, color="orange", legend=False)
        # table.resample("10T").plot(figsize=(12, 3), legend=False)
        plt.savefig(f"./line/{name_prefix}-{sno}.png")


def is_holiday(date: date):
    long_holiday = pd.date_range(start="2023-10-07", end="2023-10-10").date

    if date in long_holiday:
        return True
    # Check if the given date is a Saturday or Sunday
    if date.weekday() in [5, 6]:
        return True

    return False


def get_tot(df, snos):
    result_df = (
        df[df["sno"].isin(snos)]
        .groupby("sno", observed=True)["tot"]
        .first()
        .reset_index()
    )
    result_df.set_index("sno", inplace=True)
    result_df.columns = ["tot"]
    s_arr = result_df.values.reshape(-1)
    return s_arr


def date_range(start: str, end: str):
    return pd.date_range(start, end).date


# def evaluation(y_true, y_pred, tots):
#    print("MAE", mean_absolute_error(y_true, y_pred))
#
#    # yt is a row of many station's sbi
#    errors = np.array([mean_absolute_error(yt, yp) for yt, yp in zip(y_true, y_pred)])
#    # TODO make it time
#    xs = np.arange(len(errors))
#    plt.plot(xs, errors)
#    plt.savefig("error.png")
#
#    # errors2 = np.abs(y_pred - y_true).reshape(-1)
#    # counts, edges, bars = plt.hist(errors2, bins=len(set(errors2)))
#    # plt.bar_label(bars)
#    # plt.show()
#
#    # error function defined in the problem description
#    err = (
#        3
#        * np.abs((y_pred - y_true) / tots)
#        * (np.abs(y_true / tots - 1 / 3) + np.abs(y_true / tots - 2 / 3))
#    )
#    print("Score", err.mean())


# should make tot an array,which may change though time, sbi from 0 to tot.max


def error(y_true: np.ndarray, y_pred: np.ndarray, tots: np.ndarray) -> np.float64:
    return (
        3
        * np.abs((y_pred - y_true) / tots)
        * (np.abs(y_true / tots - 1 / 3) + np.abs(y_true / tots - 2 / 3))
    ).mean()


def evaluation(y_true, y_pred, tots, time_range):
    print("MAE: ", mean_absolute_error(y_true / tots, y_pred / tots))

    # yt is a row of many station's sbi at a time
    MAEs = np.array(
        [mean_absolute_error(yt, yp) for yt, yp in zip(y_true / tots, y_pred / tots)]
    )

    # Plot errors against timestamps
    plt.plot(time_range, MAEs, color="magenta")

    # error function defined in the problem description
    # tots is 1d array, but still fine for matrix /
    print("Score: ", error(y_true, y_pred, tots))

    # yt is a row of many station's sbi
    scores = np.array([error(yt, yp, tots) for yt, yp in zip(y_true, y_pred)])

    # plot score error
    plt.plot(time_range, scores, color="blue")

    # set up figure and save
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Score(blue) & MAE(magenta)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"error-{datetime.now().strftime('%m-%d-%H-%M')}.png")

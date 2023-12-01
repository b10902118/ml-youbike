from datetime import date, time
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

DATA_DIR = Path("../html.2023.final.data")
DEMOGRAPHICS_PATH = DATA_DIR / "demographic.json"
RELEASE_DIR = DATA_DIR / "release"
SNO_TEST_SET = DATA_DIR / "sno_test_set.txt"
BASE_DATE = "20231002"
CPU_CNT = os.cpu_count()
MAX_WORKER = 8 if CPU_CNT < 16 else CPU_CNT // 4  # personal pc or workstation


# add sno to dic (preserve all attr)
# Creating a DataFrame by passing a dictionary of objects
# where the keys are the column labels and the values are the column values.
def load_demographics(path: Path):
    with open(SNO_TEST_SET) as f:
        ntu_snos = [l.strip() for l in f.read().splitlines()]
    with open(path) as f:
        demo = json.load(f)  # { key: DATA }
        ar = []
        for k, v in demo.items():
            if k in ntu_snos:
                v["sno"] = k  # station number
                ar.append(v)
    return pd.DataFrame(ar)


# try date & time
def load_data_file(path: Path, base_date=BASE_DATE):
    with open(path) as f:
        data = json.load(f)
        ar = []
        for k, v in data.items():
            t = pd.to_datetime(f"{base_date} {k}")
            v["time"] = t
            v["sno"] = path.stem
            # redundant
            if "bemp" in v:
                v.pop("bemp")
            ar.append(v)
    # fill NaN with next value and then previous value (for last empty)
    return pd.DataFrame(ar).bfill(limit=5).ffill(limit=5)  # TODO do after concat


# 1. conform column "time" with timestamp
# 2. drop "bemp" (bemp = tot - sbi)
# 3. use compact dtype category and int8 (current cache size: 120MB)
"""
                    time        sno  tot  sbi act
0    2023-10-02 00:00:00  500101001   28   12   1
1    2023-10-02 00:01:00  500101001   28   12   1
2    2023-10-02 00:02:00  500101001   28   13   1
3    2023-10-02 00:03:00  500101001   28   13   1
4    2023-10-02 00:04:00  500101001   28   13   1
...                  ...        ...  ...  ...  ..
1435 2023-11-16 23:55:00  500119091   18    1   1
1436 2023-11-16 23:56:00  500119091   18    1   1
1437 2023-11-16 23:57:00  500119091   18    1   1
1438 2023-11-16 23:58:00  500119091   18    1   1
1439 2023-11-16 23:59:00  500119091   18    1   1

[6289920 rows x 5 columns]
time    datetime64[ns]
sno           category
tot               int8
sbi               int8
act           category
dtype: object
"""


def load_all_data(demographics, cache_path=Path("./cache/small_data_cache.pkl")):
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    with ProcessPoolExecutor(max_workers=MAX_WORKER) as executor:
        df_over_dates = []
        for date_dir in sorted(list(RELEASE_DIR.iterdir())):
            date = date_dir.name
            files = []
            not_exist = []
            for sno in demographics["sno"]:
                if (p := date_dir / f"{sno}.json").exists():
                    files.append(p)
                else:
                    not_exist.append(sno)
            print(f"{date} not exist: {not_exist}")
            dates = [date] * len(files)
            results = executor.map(load_data_file, files, dates)
            dfs = list(tqdm(results, total=len(files), desc=f"Loading {date}"))
            # dfs = [result for result in results]
            df = pd.concat(dfs)
            df_over_dates.append(df)

    df = pd.concat(df_over_dates)
    df.dropna(inplace=True)
    # compact data
    to_category = ["sno", "act"]
    df[to_category] = df[to_category].astype("category")
    to_int8 = ["tot", "sbi"]
    df[to_int8] = df[to_int8].astype("int8")

    df.to_pickle(cache_path)
    assert df.groupby(by="sno", observed=True)["time"].is_monotonic_increasing.all()
    N_STATIONS = df["sno"].nunique()
    assert N_STATIONS == 112, "ntu station number not matched"
    print(df)
    print(df.dtypes)
    print("small cache created")
    # return df


print(f"CPU_CNT = {CPU_CNT}")
print(f"MAX_WORKER = {MAX_WORKER}")
demographics = load_demographics(DEMOGRAPHICS_PATH)
load_all_data(demographics)

# groupby only contains statistical infomation

# bad_station = [
#    "500105087",
#    "500108169",
#    "500108170",
# ]  # these two stations are added very late, so we remove them for simplicity

# all_data = df[~df["sno"].isin(bad_station)]

# print(N_STATIONS)
# print(all_data)

# %%
import json

# %% [markdown]
# import pandas as pd
#

# %% [markdown]
# with open("./test.json", "r") as f:
#

# %%
with open("./weather10-03_11-02.json", "r") as f:
    file_data1 = f.read()  # .lstrip().rstrip()
    data1 = json.loads(file_data1)
with open("./1029-1128.json", "r") as f:
    file_data2 = f.read()  # .lstrip().rstrip()
    data2 = json.loads(file_data2)
with open("./1109_1209.json", "r") as f:
    file_data3 = f.read()  # .lstrip().rstrip()
    data3 = json.loads(file_data3)

# %%
taipei_data1 = data1["cwaopendata"]["resources"]["resource"]["data"]["surfaceObs"][
    "location"
][3]["stationObsTimes"]["stationObsTime"]
taipei_data2 = data2["cwaopendata"]["resources"]["resource"]["data"]["surfaceObs"][
    "location"
][3]["stationObsTimes"]["stationObsTime"]
taipei_data3 = data3["cwaopendata"]["resources"]["resource"]["data"]["surfaceObs"][
    "location"
][3]["stationObsTimes"]["stationObsTime"]

# %%
print(type(taipei_data1))

# %%


# have some duplicate (handle by pandas)
taipei_data = taipei_data1 + taipei_data2 + taipei_data3
print(taipei_data3[-1])

# %%

rain_ar = []
all_ar = []
rain_dic = {}
Tcnt = 0
Xcnt = 0
Rcnt = 0
T = 0.3
X = None
for d in taipei_data:
    all = d["weatherElements"]
    rain = d["weatherElements"]["Precipitation"]
    datetime = d["DateTime"]
    if rain == "X":
        Xcnt += 1
        rain = X
    elif rain == "T":
        Tcnt += 1
        rain = T
    else:
        rain = float(rain)
        if rain < 0.0:
            raise ValueError("rain is negative")
        if rain > 0.0:
            Rcnt += 1
        if rain in rain_dic:
            rain_dic[rain] += 1
        else:
            rain_dic[rain] = 1
    rain_ar.append({"datetime": datetime, "rain": rain})
    all = {"datetime": datetime, **all}
    all_ar.append(all)
print(f"Total: {len(taipei_data)} (dup)")
print(Tcnt)
print(Rcnt)
print(Xcnt)
print(rain_dic)

# %%
for e in rain_ar:
    print(e)

# %%
for e in all_ar:
    print(e)

# %% [markdown]
# To dataframe

# %%
import pandas as pd

rain_df = pd.DataFrame(rain_ar)
rain_df.drop_duplicates(subset="datetime", inplace=True)
print(rain_df)


# %% [markdown]
#                        datetime  rain
# 0     2023-10-03T01:00:00+08:00   0.0
# 1     2023-10-03T02:00:00+08:00   0.0
# 2     2023-10-03T03:00:00+08:00   0.0
# 3     2023-10-03T04:00:00+08:00   0.0
# 4     2023-10-03T05:00:00+08:00   0.0
# ...                         ...   ...
# 2227  2023-12-09T20:00:00+08:00   0.0
# 2228  2023-12-09T21:00:00+08:00   0.0
# 2229  2023-12-09T22:00:00+08:00   0.0
# 2230  2023-12-09T23:00:00+08:00   0.0
# 2231  2023-12-09T24:00:00+08:00   0.0
#
# [1632 rows x 2 columns]
#
# >>> 1632/(31+30+7)
# 24.0
#

# %%
all_df = pd.DataFrame(all_ar)
all_df.drop_duplicates(subset="datetime", inplace=True)
print(all_df)

# %% [markdown]
#                        datetime AirPressure AirTemperature RelativeHumidity  \
# 0     2023-10-03T01:00:00+08:00      1008.2           24.9               80
# 1     2023-10-03T02:00:00+08:00      1007.5           24.5               82
# 2     2023-10-03T03:00:00+08:00      1006.9           24.2               83
# 3     2023-10-03T04:00:00+08:00      1006.8           24.1               82
# 4     2023-10-03T05:00:00+08:00      1006.9           24.1               77
# ...                         ...         ...            ...              ...
# 2227  2023-12-09T20:00:00+08:00      1012.1           22.1               85
# 2228  2023-12-09T21:00:00+08:00      1011.9           21.7               88
# 2229  2023-12-09T22:00:00+08:00      1011.6           21.5               89
# 2230  2023-12-09T23:00:00+08:00      1011.2           21.0               91
# 2231  2023-12-09T24:00:00+08:00      1010.9           20.6               92
#
#      WindSpeed WindDirection Precipitation SunshineDuration
# 0          1.2         東南,SE           0.0             None
# 1          1.5          東,SE           0.0             None
# 2          1.4       南南東,SSE           0.0             None
# 3          1.2          東,SE           0.0             None
# 4          1.8          東,SE           0.0             None
# ...        ...           ...           ...              ...
# 2227       2.2           東,E           0.0             None
# 2228       1.8           東,E           0.0             None
# 2229       1.4       東南東,ESE           0.0             None
# 2230       0.9          東,SE           0.0             None
# 2231       0.8         東南,SE           0.0             None
#
# [1632 rows x 8 columns]

# %%


rain_df.to_pickle("./10-03_12_09_rain.pkl")
all_df.to_pickle("./10-03_12_09_all.pkl")

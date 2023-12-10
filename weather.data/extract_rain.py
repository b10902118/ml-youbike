import json
import pandas as pd
import datetime as dt

key_order = [
    "datetime",
    "now",
    "10min",
    "1hr",
    "3hr",
    "6hr",
    "12hr",
    "24hr",
    "2day",
    "3day",
]
# format 1
key_mapping1 = {
    "NOW": "now",
    "MIN_10": "10min",
    "RAIN": "1hr",  # I assume
    "HOUR_3": "3hr",
    "HOUR_6": "6hr",
    "HOUR_12": "12hr",
    "HOUR_24": "24hr",
    "latest_2days": "2day",
    "latest_3days": "3day",
}
# format 2
key_mapping2 = {
    "Now": "now",
    "Past10Min": "10min",
    "Past1hr": "1hr",
    "Past3hr": "3hr",
    "Past6Hr": "6hr",
    "Past12hr": "12hr",
    "Past24hr": "24hr",
    "Past2days": "2day",
    "Past3days": "3day",
}


# with open("./test.json", "r") as f:
with open("./rain/A0A010.json", "r") as f:
    file_data = f.read().lstrip().rstrip()
    # Split the lines and parse each line individually
    cnt = 0
    ar = []
    for line in file_data.split("\n"):
        # print("'", line, "'")
        data = json.loads(line)
        entry = {}
        try:
            if "location" in data:
                datetime = data["location"][0]["time"]["obsTime"]
                rainfall = data["location"][0]["weatherElement"]
                rainfall.pop(0)  # ELEV

                entry = {"datetime": pd.to_datetime(datetime)}
                for e in rainfall:
                    key = key_mapping1[e["elementName"]]
                    value = float(e["elementValue"])
                    if value < 0:
                        value = 0.0
                    entry[key] = value
                entry = {key: entry[key] for key in key_order}
            else:
                # datetime = data["Station"][0]["ObsTime"]["DateTime"]
                # for filtering all station returned
                datetime = None
                for d in data["Station"]:
                    if d["StationName"] == "臺灣大學":
                        datetime = d["ObsTime"]["DateTime"]
                        break
                if datetime is None:
                    raise ValueError("Variable 'datetime' cannot be None")

                rainfall = data["Station"][0]["RainfallElement"]

                entry = {"datetime": pd.to_datetime(datetime).tz_localize(None)}
                for k, v in rainfall.items():
                    key = key_mapping2[k]
                    value = v["Precipitation"]
                    entry[key] = value
            ar.append(entry)
        except Exception as e:
            print("Error:", e)
            print(json.dumps(data, indent=2))

        # print(f"DateTime: {datetime}")
        # print(f"weatherElement: {rainfall}")
        # print("\n")
        # print(entry)
    df = pd.DataFrame(ar)  # TODO do after concat

current_datetime = dt.datetime.now()
datetime_string = current_datetime.strftime("%m-%d-%H-%M")
df.to_pickle(f"./rain_A0A010_1107-{datetime_string}.pkl")

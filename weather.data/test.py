import json
import requests
import schedule
import time

WEATHER_API = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001?Authorization=CWA-97D2D089-612F-450C-B753-115122222DEA&parameterName=0&StationId="
RAIN_API = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001?Authorization=CWA-97D2D089-612F-450C-B753-115122222DEA&parameterName=0&StationId="


def fetch_weather(max_retries=10):
    # Define the URLs you want to fetch data from
    # fmt: off
    weather_stations = ["CAAH60", # daan park
                        "466920"] # 氣象局
    # fmt: on

    for ws in weather_stations:
        for retry in range(max_retries):
            try:
                response = requests.get(WEATHER_API + ws)
                response.raise_for_status()
                data = response.json()  # Assuming the response is in JSON format
                print(data)
                data = data["success"]
                print(data["records"])
                # with open("./weather/" + ws + ".json", "a") as file:
                #    json.dump(data, file)
                #    file.write("\n")
            except requests.exceptions.RequestException as e:
                print(f"Weather: Error fetching data from {ws}: {e}")
                time.sleep(5)
                continue
            except json.JSONDecodeError as e:
                print(f"Weather: Error parsing JSON response: {e}")
                time.sleep(5)
                continue
            break


def fetch_rain(max_retries=10):
    # fmt: off
    rain_stations = ["A0A010", # ntu
                     "CAAH60", # daan park
                     "A1A9Z0"] # 中正國中
    # fmt: on

    for rs in rain_stations:
        for retry in range(max_retries):
            try:
                response = requests.get(RAIN_API + rs)
                response.raise_for_status()
                data = response.json()  # Assuming the response is in JSON format
                print(data)
                # data = data["records"]
                # print(data["records"])
                # with open("./rain/" + rs + ".json", "a") as file:
                #    json.dump(data, file)
                #    file.write("\n")
            except requests.exceptions.RequestException as e:
                print(f"Rain: Error fetching data from {rs}: {e}")
                time.sleep(5)
                continue
            except json.JSONDecodeError as e:
                print(f"Rain: Error parsing JSON response: {e}")
                time.sleep(5)
                continue
            break


# Schedule the job to run every hour
fetch_weather()
fetch_rain()
schedule.every(10).minutes.do(fetch_weather)
schedule.every(10).minutes.do(fetch_rain)

# while True:
#    schedule.run_pending()
#    time.sleep(1)

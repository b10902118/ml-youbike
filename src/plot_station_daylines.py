# ntu_stops = [item for item in ntu_stops for _ in range(3 * 24)] * 4
# print(len(ntu_stops))
# exit()
# print(holidays)

# snos = [f"50010100{i}" for i in range(1, 6)]  # sampled stations to plot
# long_holiday = pd.date_range(start="2023-10-07", end="2023-10-10").date

# holiday_df = df[df["date"].isin(holidays)]
# weekday_df = df[~df["date"].isin(holidays)]
# print(holiday_df)
# plot_line(df, snos, "all")
# plot_line(weekday_df, snos, "weekday")
# plot_line(holiday_df, snos, "holiday")

# dfp = pd.pivot_table(df, values="sbi", index="time", columns="date")


# df = df[~df["time"].dt.date.isin(holidays)]
# print(df)
# exit()

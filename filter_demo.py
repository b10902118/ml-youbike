import json
from pathlib import Path

selected_keys = ["500119089", "500119091"]
selected_keys = ["500119089", "500119091"]

DATA_DIR = Path("./html.2023.final.data")
DEMOGRAPHICS_PATH = DATA_DIR / "demographic.json"
RELEASE_DIR = DATA_DIR / "release"
SNO_TEST_SET = DATA_DIR / "sno_test_set.txt"
BASE_DATE = "20231002"

with open(SNO_TEST_SET) as f:
    ntu_snos = [l.strip() for l in f.read().splitlines()]
with open("./html.2023.final.data/demographic.json") as f:
    input_json = json.load(f)
# Filter and print the selected entries to a file
with open("test_demo.json", "w") as output_file:
    for key in ntu_snos:
        if key in input_json:
            json.dump({key: input_json[key]}, output_file, ensure_ascii=False, indent=2)
            output_file.write(",\n")

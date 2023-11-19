from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

DATA_DIR = Path("./html.2023.final.data")
DEMOGRAPHICS_PATH = DATA_DIR / "demographic.json"
RELEASE_DIR = DATA_DIR / "release"
SNO_TEST_SET = DATA_DIR / "sno_test_set.txt"
BASE_DATE = "20231002"


def print_demographics(path: Path = DEMOGRAPHICS_PATH):
    with open(SNO_TEST_SET) as f:
        ntu_snos = [l.strip() for l in f.read().splitlines()]
    with open(path) as f:
        demo = json.load(f)  # { key: DATA }
        for k, v in demo.items():
            if k in ntu_snos:
                print(v)


print_demographics()

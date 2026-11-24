
import pandas as pd
from pathlib import Path

for file in Path("data").glob("*.csv"):
    print("Loading:", file.name, flush=True)

    df = pd.read_csv(file, nrows=5)

    print("Columns:", len(df.columns))
    print("Sample shape:", df.shape)
    print(df.head(2))
    print("-" * 40)

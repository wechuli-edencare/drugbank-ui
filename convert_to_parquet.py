import pandas as pd
import json

def flatten_value(val):
    if isinstance(val, list):
        return "; ".join(str(flatten_value(x)) for x in val)
    if isinstance(val, dict):
        # Try to get a representative string
        if "#text" in val:
            return str(val["#text"])
        return "; ".join(f"{k}:{flatten_value(v)}" for k, v in val.items())
    return str(val) if val is not None else ""

with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten all values in each record
flat_data = [
    {k: flatten_value(v) for k, v in record.items()}
    for record in data
]

df = pd.DataFrame(flat_data)
df.to_parquet("output.parquet")
print(f"Converted {len(df)} records to output.parquet")
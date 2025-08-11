import os
import datetime

def save_tsv(df, prefix, folder="data"):
    """Save curated results as a TSV"""
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.tsv"
    path = os.path.join(folder, filename)
    df.to_csv(path, sep="\t", index=False)
    return path
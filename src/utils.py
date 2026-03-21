import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import pandas as pd

def clean_text(text, max_len=500):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def load_and_prepare_data(parquet_path, batch_indices=[0,10,20,30], test_size=0.2, random_state=42):
    print("Loading data...")
    parquet_file = pq.ParquetFile(parquet_path)
    sampled_batches = []
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=100000)):
        if i in batch_indices:
            sampled_batches.append(batch.to_pandas())
    df = pd.concat(sampled_batches, ignore_index=True)
    print(f"Loaded: {df.shape}")

    texts = [clean_text(t) for t in df["DATA"].tolist()]
    le = LabelEncoder()
    labels = le.fit_transform(df["TOPIC"].tolist())
    del df

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_val, y_train, y_val, le

def save_artifacts(path, **kwargs):
    for name, obj in kwargs.items():
        pickle.dump(obj, open(f"{path}/{name}.pkl", "wb"))
        print(f"Saved {name}.pkl")

def load_artifacts(path, *names):
    return [pickle.load(open(f"{path}/{name}.pkl", "rb")) for name in names]

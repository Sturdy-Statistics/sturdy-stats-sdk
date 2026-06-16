import os
import sys, tempfile
sys.path.insert(0, "..")

import pandas as pd
from uuid import uuid4
from sturdystats import Dataset, ClfBase, ClfModel

suffix = str(uuid4())[:8]

ORG_ID = os.environ["TMP_SILAND_LOCAL_ORG"]
API_KEY = os.environ["TMP_SILAND_LOCAL_KEY"]
BASE_URL = "http://localhost:3333"
PARQUET_PATH = "/Users/kian/ML/clojure/Siland/slack_reviews.parquet"

conn = dict(org_id=ORG_ID, api_key=API_KEY, base_url=BASE_URL)

print("--- prepare parquet with 5_star field ---")
df = pd.read_parquet(PARQUET_PATH)
df["5_star"] = df["rating"] == 5
with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
    tmp_path = tmp.name
df.to_parquet(tmp_path, index=False)
print(df[["rating", "5_star"]].value_counts().sort_index())

print("\n--- create dataset ---")
ds = Dataset.create(f"smoke-clf-{suffix}", **conn)
print(ds)

print("\n--- append parquet ---")
print(ds.append(tmp_path))

print("\n--- commit ---")
ds.commit()
print(ds.wait())

print("\n--- create clf-base ---")
base = ClfBase.create(dataset_id=ds.id, model_name=f"smoke-base-{suffix}", **conn)
print(base)
print("waiting for clf-base...")
base.wait()
print(base.status())

print("\n--- create clf-model ---")
model = ClfModel.create(
    clf_base_id=base.id,
    dataset_id=ds.id,
    model_name=f"smoke-model-{suffix}",
    fields=["5_star"],
    **conn,
)
print(model)
print("waiting for clf-model...")
model.wait()
print(model.status())

print("\n--- predict_one ---")
sample_doc = df["doc"].iloc[0]
print(f"doc: {sample_doc[:80]}...")
print(model.predict_one(sample_doc))

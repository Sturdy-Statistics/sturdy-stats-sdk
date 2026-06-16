import sys
import os
sys.path.insert(0, "..")

from sturdystats import Dataset, Index
from uuid import uuid4

suffix = str(uuid4())[:8]

ORG_ID = os.environ["TMP_SILAND_LOCAL_ORG"]
API_KEY = os.environ["TMP_SILAND_LOCAL_KEY"]
BASE_URL = "http://localhost:3333"
PARQUET_PATH = "/Users/kian/ML/clojure/Siland/slack_reviews.parquet"

conn = dict(org_id=ORG_ID, api_key=API_KEY, base_url=BASE_URL)

print("--- list datasets ---")
print(Dataset(**conn).list())

print("\n--- create dataset ---")
ds = Dataset.create(f"smoke-test-{suffix}", **conn)
print(ds)

print("\n--- append parquet ---")
print(ds.append(PARQUET_PATH))

print("\n--- commit ---")
ds.commit()
print(ds.wait())

print("\n--- status ---")
print(ds.status())  # GET /datasets/{id}

print("\n--- create index ---")
idx = Index.create(
    f"smoke-test-index-{suffix}",
    dataset_id=ds.id,
    private_annotations=True,
    model_arch="aalda-para",
    burn_in=10,
    sample=10,
    n_topics=32,
    **conn,
)
print(idx)
print("waiting for index...")
idx.wait()
print(idx)

print("\n--- topics_search ---")
print(idx.topics_search(
    level="doc",
    topic_mention_cutoff=2.0,
    semantic_search_cutoff=0.1,
    semantic_search_weight=0.3,
).head())

print("\n--- docs_search ---")
print(idx.search(
    level="doc",
    sort_by="relevance",
    topic_mention_cutoff=2.0,
    semantic_search_cutoff=0.1,
    semantic_search_weight=0.3,
    limit=5,
).head())

print("\n--- sql ---")
print(idx.sql(
    sql="SELECT doc_id, text, topic_prevalence FROM doc LIMIT 5",
    topic_mention_cutoff=2.0,
    semantic_search_cutoff=0.1,
    semantic_search_weight=0.3,
))

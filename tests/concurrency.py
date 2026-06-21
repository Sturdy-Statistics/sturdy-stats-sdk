import sys, os, time
sys.path.insert(0, "..")
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4
from sturdystats import Dataset, Index

suffix = str(uuid4())[:8]
conn = dict(
    org_id=os.environ["TMP_SILAND_LOCAL_ORG"],
    api_key=os.environ["TMP_SILAND_LOCAL_KEY"],
    base_url="http://localhost:3333",
)
PARQUET_PATH = "/Users/kian/ML/clojure/Siland/slack_reviews.parquet"
N = int(os.environ.get("N", "20"))   # number of concurrent queries

# --- build dataset ---
ds = Dataset.create(f"load-test-{suffix}", **conn)
ds.append(PARQUET_PATH)
ds.commit()          # blocks until ready
ds.wait()
print("dataset ready:", ds.id)

# --- train index ---
idx = Index.create(
    f"load-test-index-{suffix}",
    dataset_id=ds.id,
    private_annotations=True,
    model_arch="aalda-para",
    burn_in=10, sample=10, n_topics=32,
    **conn,
)
idx.wait()           # blocks until training complete
print("index ready:", idx.id)

# --- fire N concurrent SQL queries ---
def run_query(i):
    t0 = time.perf_counter()
    try:
        df = idx.sql(sql="SELECT * FROM doc LIMIT 10",
                     search_query="unread messages are so annoying",
                     search_level="doc")
        return (i, "ok", len(df), time.perf_counter() - t0)
    except Exception as e:
        return (i, f"err: {e}", 0, time.perf_counter() - t0)

print(f"\n--- firing {N} concurrent queries ---")
wall0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=N) as ex:
    futures = [ex.submit(run_query, i) for i in range(N)]
    results = [f.result() for f in as_completed(futures)]
wall = time.perf_counter() - wall0

ok = [r for r in results if r[1] == "ok"]
print(f"\n{len(ok)}/{N} succeeded in {wall:.2f}s wall")
lat = sorted(r[3] for r in ok)
if lat:
    print(f"latency min={lat[0]:.3f}s  median={lat[len(lat)//2]:.3f}s  max={lat[-1]:.3f}s")
for r in results:
    if r[1] != "ok":
        print("  ", r)


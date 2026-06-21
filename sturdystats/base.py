import os
import io
import requests


class SturdyStatsSdkError(Exception):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class SturdyStatsBase:
    def __init__(
        self,
        org_id: str = None,
        api_key: str = None,
        base_url: str = None,
        id: str = None,
    ):
        """Connection args fall back to environment variables when omitted or None:
            org_id   ← STURDY_STATS_ORG_ID
            api_key  ← STURDY_STATS_API_KEY
            base_url ← STURDY_STATS_BASE_URL  (default https://api.sturdystatistics.com)
        """
        org_id = org_id or os.environ.get("STURDY_STATS_ORG_ID")
        api_key = api_key or os.environ.get("STURDY_STATS_API_KEY")
        base_url = base_url or os.environ.get("STURDY_STATS_BASE_URL", "https://api.sturdystatistics.com")
        if not org_id:
            raise ValueError("org_id is required (or set STURDY_STATS_ORG_ID)")
        if not api_key:
            raise ValueError("api_key is required (or set STURDY_STATS_API_KEY)")

        self.org_id = org_id
        self.api_key = api_key
        self.id = id
        self.base_url = base_url.rstrip("/") + f"/api/v1/orgs/{org_id}"

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def __repr__(self):
        cls = self.__class__.__name__
        return f"<{cls} id={self.id!r}>" if self.id else f"<{cls}>"

    def _url(self, path: str) -> str:
        return self.base_url + "/" + path.lstrip("/")

    def _raise(self, resp: requests.Response):
        if not resp.ok:
            raise SturdyStatsSdkError(resp.status_code, resp.text)

    def _get(self, path: str, params: dict = None) -> dict:
        resp = self._session.get(self._url(path), params=params)
        self._raise(resp)
        return resp.json()

    def _post(self, path: str, body: dict = None) -> dict:
        resp = self._session.post(self._url(path), json=body or {})
        self._raise(resp)
        return resp.json()

    def _send_parquet(self, path: str, filepath: str) -> list[dict]:
        """Upload one parquet file or all *.parquet files in a directory, sequentially.
        → POST path  (e.g. datasets/{id}/append)
        """
        from pathlib import Path
        p = Path(filepath)
        files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
        if not files:
            raise ValueError(f"No parquet files found at {filepath}")
        results = []
        for f in files:
            with open(f, "rb") as fh:
                resp = self._session.post(
                    self._url(path),
                    files={"file": (f.name, fh, "application/octet-stream")},
                )
            self._raise(resp)
            results.append(resp.json())
        return results

    def _load_parquet(self, path: str, body: dict = None, transform=None):
        """POST to a parquet-returning endpoint, load into a pandas DataFrame via DuckDB.
        → POST path  (e.g. indices/{id}/sql)
        Bytes are read into an Arrow table in memory, then handed to DuckDB so MAP
        columns — e.g. topic_count MAP(SMALLINT, FLOAT) — materialize as real Python
        dicts rather than pandas' list-of-tuples. No temp files.
        Optionally apply transform(df) -> df before returning.
        """
        import duckdb
        import pyarrow.parquet as pq

        resp = self._session.post(self._url(path), json=body or {})
        self._raise(resp)
        arrow_table = pq.read_table(io.BytesIO(resp.content))
        con = duckdb.connect()
        try:
            df = con.from_arrow(arrow_table).df()
        finally:
            con.close()
        if transform is not None:
            df = transform(df)
        return df

    def _wait_for_dataset(self, dataset_id: str, poll_interval_start: float = 2.0,
                          poll_interval_max: float = 15.0, timeout: float = 1800.0):
        """Block until dataset current-state is ready or failed.
        → GET /datasets/{dataset-id}
        """
        import time
        deadline = time.time() + timeout
        interval = poll_interval_start
        while time.time() < deadline:
            data = self._get(f"datasets/{dataset_id}")
            state = data.get("status")
            if state == "ready":
                return data
            if state == "failed":
                raise SturdyStatsSdkError(0, f"Dataset {dataset_id} failed: {data}")
            time.sleep(interval)
            interval = min(interval * 1.5, poll_interval_max)
        raise TimeoutError(f"Dataset {dataset_id} did not become ready within {timeout}s")

    def _wait_for_index(self, index_id: str, poll_interval_start: float = 5.0,
                        poll_interval_max: float = 60.0):
        """Block until index status is ready or failed. No timeout — training can take hours.
        → GET /indices/{index-id}
        """
        import time
        interval = poll_interval_start
        while True:
            data = self._get(f"indices/{index_id}")
            status = data.get("status")
            if status == "ready":
                return data
            if status == "failed":
                raise SturdyStatsSdkError(0, f"Index {index_id} failed: {data}")
            time.sleep(interval)
            interval = min(interval * 1.5, poll_interval_max)

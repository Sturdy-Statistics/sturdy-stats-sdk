import os
import io
import pandas as pd
import requests


class SturdyStatsSdkError(Exception):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class SturdyStatsBase:
    def __init__(
        self,
        org_id: str = os.environ.get("STURDY_STATS_ORG_ID"),
        api_key: str = os.environ.get("STURDY_STATS_API_KEY"),
        base_url: str = os.environ.get("STURDY_STATS_BASE_URL", "https://api.sturdystatistics.com"),
        id: str = None,
    ):
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
        """POST to a parquet-returning endpoint, load into a pandas DataFrame.
        → POST path  (e.g. indices/{id}/sql)
        Optionally apply transform(df) -> df before returning.
        """
        resp = self._session.post(self._url(path), json=body or {})
        self._raise(resp)
        buf = io.BytesIO(resp.content)
        df = pd.read_parquet(buf)
        if transform is not None:
            df = transform(df)
        return df

    def _wait_for_dataset(self, dataset_id: str, poll_interval_start: float = 2.0,
                          poll_interval_max: float = 15.0, timeout: float = 1800.0):
        """Block until dataset current-state is ready or failed.
        → GET /datasets/{dataset-id}/status
        """
        import time
        deadline = time.time() + timeout
        interval = poll_interval_start
        while time.time() < deadline:
            data = self._get(f"datasets/{dataset_id}/status")
            state = data.get("current-state")
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

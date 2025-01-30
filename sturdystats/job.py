import requests
import os
from time import sleep
import json

import pandas as pd
import srsly
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential
) 

# for type checking
from typing import Dict 
from requests.models import Response


class Job:
    def __init__(self, API_key: str, 
                 job_id: str, poll_seconds: int = 1, 
                 msgpack: bool = True,
                 _base_url: str= "https://sturdystatistics.com/api/v1/job"):
        self.API_key = API_key or os.environ["STURDY_STATS_API_KEY"]
        self.job_id = job_id
        self.poll_seconds = poll_seconds
        self.base_url = _base_url 
        self.msgpack = msgpack

    def _check_status(self, info: Response) -> None:
        if info.status_code != 200:
            raise requests.HTTPError(info.content)

    def _post(self, url: str, params: Dict) -> Response:
        payload = {**params}
        res = requests.post(self.base_url + url, json=payload, headers={"x-api-key": self.API_key})
        self._check_status(res)
        return res

    @retry(wait=wait_exponential(),
           stop=(stop_after_delay(240)))
    def _get_retry(self, url: str, params: Dict) -> Response:
        res = requests.get(self.base_url + url , params=params, headers={"x-api-key": self.API_key})
        return res

    #@retry(wait=wait_exponential(),
    #       stop=(stop_after_delay(2)))
    def _get(self, url: str, params: Dict) -> Response:
        res = self._get_retry(url, params)
        self._check_status(res)
        return res

    def get_status(self):
        res = self._get("/"+self.job_id, dict(msgpack=self.msgpack))
        res = srsly.msgpack_loads(res.content) if self.msgpack else res.json()
        return res

    def print_status(self):
        st = self.get_status()
        t0 = pd.Timestamp(st['finishedAt']) if 'finishedAt' in st else pd.Timestamp.now(tz='UTC')
        dt = t0 - pd.Timestamp(st['startedAt'])
        # format time elapsed as hh:mm:ss
        y = dt.total_seconds()
        h = 3600
        tstr = f'{int(y/h):02d}h:{int(y%h/60):02d}m:{int(y%60):02d}s'
        # one-line message
        print(f"""{st['status']} - {tstr}""")

    def _is_running(self):
        status = self.get_status()
        return status["status"] not in ["FAILED", "SUCCEEDED", "CANCELLED"]

    def wait(self):
        poll_seconds = .5
        while True:
            if not self._is_running():
                break
            sleep(poll_seconds)
            poll_seconds = min(self.poll_seconds, poll_seconds+.3)
        status = self.get_status()
        if status["status"] == "FAILED":
            raise Exception(f"Job {self.job_id} failed with the following error: {status['error']}")
        return status

    def cancel(self):
        return self._post(f"/{self.job_id}/cancel", dict())

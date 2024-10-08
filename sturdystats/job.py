import requests
import os
from time import sleep
import json

import srsly
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential
) 

# for type checking
from typing import Dict 
from requests.models import Response


class Job:
    def __init__(self, API_key: str, 
                 job_id: str, poll_seconds: int = 1, _base_url: str= "https://sturdystatistics.com/api/text/v1/job"):
        self.API_key = API_key or os.environ["STURDY_STATS_API_KEY"]
        self.job_id = job_id
        self.poll_seconds = poll_seconds
        self.base_url = _base_url 

    def _check_status(self, info: Response) -> None:
        if (200 != info.status_code):
            print(f"""error code {info.status_code}""")
            print(info.content.decode("utf-8"))
        assert(200 == info.status_code)

    def _post(self, url: str, params: Dict) -> Response:
        payload = {**params}
        res = requests.post(self.base_url + url, json=payload, headers={"x-api-key": self.API_key})
        self._check_status(res)
        return res

    def _get(self, url: str, params: Dict) -> Response:
        res = requests.get(self.base_url + url , params=params, headers={"x-api-key": self.API_key})
        self._check_status(res)
        return res

    @retry(wait=wait_exponential(),
           stop=(stop_after_delay(240)))
    def get_status(self):
        res = self._get("/"+self.job_id, dict())
        res = res.json()
        if "result" in res:
            res["result"] = json.loads(res["result"])
        return res

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

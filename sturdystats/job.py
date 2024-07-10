import requests
from time import sleep
import json

import srsly

# for type checking
from typing import Dict 
from requests.models import Response


class Job:
    def __init__(self, API_key, job_id, poll_seconds = 1):
        self.API_key = API_key
        self.job_id = job_id
        self.poll_seconds = poll_seconds
        self.base_url = "https://sturdystatistics.com/api/text/v1/job"

    def _check_status(self, info: Response) -> None:
        if (200 != info.status_code):
            print(f"""error code {info.status_code}""")
            print(info.content.decode("utf-8"))
        assert(200 == info.status_code)

    def _post(self, url: str, params: Dict) -> Response:
        payload = {"api_key": self.API_key, **params}
        res = requests.post(self.base_url + url, json=payload)
        self._check_status(res)
        return res

    def _get(self, url: str, params: Dict) -> Response:
        params = {"api_key": self.API_key, **params}
        res = requests.get(self.base_url + url , params=params)
        self._check_status(res)
        return res


    def get_status(self):
        res = self._get("/"+self.job_id, dict())
        res = res.json()
        if "result" in res:
            res["result"] = json.loads(res["result"])
        return res

    def _is_running(self):
        status = self.get_status()
        return status["status"] not in ["FAILED", "SUCCEEDED"]


    def wait(self):
        while True:
            if not self._is_running():
                break
            sleep(self.poll_seconds)
        status = self.get_status()
        if status["status"] == "FAILED":
            raise Exception(f"Job {self.job_id} failed with the following error: {status['error']}")
        return status


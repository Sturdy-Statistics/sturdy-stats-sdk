import numpy as np
import requests
import os
import tempfile
from typing import Dict, Optional 
from requests.models import Response

import srsly
import arviz as az

from sturdystats.job import Job

_base_url = "https://sturdystatistics.com/api/v1/numeric"

class _BaseModel:
    def __init__(self, model_type: str, API_key: Optional[str] = None, _base_url: str = _base_url):
        self.API_key = API_key or os.environ["STURDY_STATS_API_KEY"]
        self.base_url = _base_url 
        self.model_type = model_type

    def _check_status(self, info: Response) -> None:
        if info.status_code != 200:
            raise requests.HTTPError(info.content)

    def _post(self, url: str, data: Dict) -> Response:
        payload = srsly.msgpack_dumps(data)
        res = requests.post(self.base_url + url, data=payload, headers={"x-api-key": self.API_key})
        self._check_status(res)
        return res
    
    def sample(self, X, Y, additional_args: str = "", background = False):
        assert len(X) == len(Y)
        X = np.array(X)
        Y = np.array(Y)
        data = dict(X=X, Y=Y, override_args=additional_args)
        job_id = self._post(f"/{self.model_type}", data).json()["job_id"]
        job = Job(API_key=self.API_key, msgpack=True, job_id=job_id, _base_url=self._job_base_url())
        if background:
            return job
        bdata: bytes = job.wait()["result"] #type: ignore
        with tempfile.TemporaryDirectory() as tempdir:
            with open(tempdir+"netcdf", "wb") as handle:
                handle.write(bdata)
        return az.from_netcdf(tempdir+"netcdf")
    
    def _job_base_url(self) -> str:
        return self.base_url.replace("numeric", "job")

class LinearRegressor(_BaseModel):
    def __init__(self, API_key: Optional[str] = None, _base_url: str= _base_url, ):
        super().__init__("linear", API_key, _base_url)

class LogisticRegressor(_BaseModel):
    def __init__(self, API_key: Optional[str], _base_url: str = _base_url):
        super().__init__("logistic", API_key, _base_url)

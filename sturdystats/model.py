from __future__ import annotations

import numpy as np
import xarray as xr

import requests
import os
import tempfile
from typing import Dict, Optional
from requests.models import Response

import srsly

from sturdystats.job import Job


_base_url = "https://api.sturdystatistics.com/api/v1/numeric"

class RegressionResult(Job):
    def getTrace(self):
        import arviz as az
        bdata: bytes = self.wait()["result"] #type: ignore
        with tempfile.TemporaryDirectory() as tempdir:
            with open(tempdir+"netcdf", "wb") as handle:
                handle.write(bdata)
        return az.from_netcdf(tempdir+"netcdf")

def _append_data(ds: xr.Dataset, X: np.array, Y: np.array):
    return ds.add_groups(
        {
            'constant_data': xr.Dataset(
                data_vars={
                    "X":  (("N", "D"), X),
                },
                coords={
                    "N": 1+np.arange(X.shape[0]),
                    "D": 1+np.arange(X.shape[1]),
                }),
            'observed_data': xr.Dataset(
                data_vars={
                    "Y":  (("N", "Q"), Y),
                },
                coords={
                    "N": 1+np.arange(Y.shape[0]),
                    "Q": 1+np.arange(Y.shape[1]),
                })
        })

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
        import arviz as az
        assert len(X) == len(Y)
        X = np.array(X)
        Y = np.array(Y)
        data = dict(X=X, Y=Y, override_args=additional_args)
        job_id = self._post(f"/{self.model_type}", data).json()["job_id"]
        job = RegressionResult(API_key=self.API_key, msgpack=True, job_id=job_id, _base_url=self._job_base_url())
        if background:
            return job
        bdata: bytes = job.wait()["result"] #type: ignore
        with tempfile.TemporaryDirectory() as tempdir:
            with open(tempdir+"netcdf", "wb") as handle:
                handle.write(bdata)
        ds = az.from_netcdf(tempdir+"netcdf")
        _append_data(ds, X, Y)
        return ds

    def _job_base_url(self) -> str:
        return self.base_url.replace("numeric", "job")

class LinearRegressor(_BaseModel):
    def __init__(self, API_key: Optional[str] = None, _base_url: str= _base_url, ):
        super().__init__("linear", API_key, _base_url)

class LogisticRegressor(_BaseModel):
    def __init__(self, API_key: Optional[str] = None, _base_url: str = _base_url):
        super().__init__("logistic", API_key, _base_url)

class SturdyLogisticRegressor(_BaseModel):
    def __init__(self, API_key: Optional[str] = None, _base_url: str = _base_url):
        super().__init__("sturdy", API_key, _base_url)

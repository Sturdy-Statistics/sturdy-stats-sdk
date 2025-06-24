from __future__ import annotations

import numpy as np
import xarray as xr
import arviz as az

import requests
import os
import tempfile
from typing import Dict, Optional, Union
from requests.models import Response

from pathlib import Path

import srsly

from sturdystats.job import Job


_base_url = "https://api.sturdystatistics.com/api/v1/numeric"

class RegressionResult(Job):
    """subclass of Job which fetches InferenceData for regression models"""
    def getTrace(self):
        bdata: bytes = self.wait()["result"] #type: ignore

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trace.nc"
            path.write_bytes(bdata)
            inference_data = az.from_netcdf(path)
        return inference_data

def _append_data(inference_data: az.InferenceData, X: np.array, Y: np.array) -> None:
    """Add training data (constant & observed) to an ArviZ InferenceData object.
    This modifies `inference_data` in-place and returns None.
    """
    inference_data.add_groups(
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
        self.inference_data: Optional[az.InferenceData] = None

    def _require_inference_data(self):
        if self.inference_data is None:
            raise RuntimeError("Model has no inference data. Did you forget to call `.sample()` or `.from_disk()`?")

    def _check_status(self, info: Response) -> None:
        if info.status_code != 200:
            raise requests.HTTPError(info.content)

    def _post(self, url: str, data: Dict) -> Response:
        payload = srsly.msgpack_dumps(data)
        res = requests.post(self.base_url + url, data=payload, headers={"x-api-key": self.API_key})
        self._check_status(res)
        return res

    def sample(self, X, Y, additional_args: str = "", background = False):
        # validate input data
        assert len(X) == len(Y)
        X = np.array(X, copy=True)
        Y = np.array(Y, copy=True)
        data = dict(X=X, Y=Y, override_args=additional_args)

        # submit training job and make a job object
        job_id = self._post(f"/{self.model_type}", data).json()["job_id"]
        job = RegressionResult(API_key=self.API_key, msgpack=True, job_id=job_id, _base_url=self._job_base_url())

        # run in background: return job object
        if background:
            return job

        # wait for results: unpack into arviz dataset
        inference_data = job.getTrace()
        inference_data.attrs["model_type"] = self.model_type
        _append_data(inference_data, X, Y)
        self.inference_data = inference_data

        return self

    def to_disk(self, path: Union[Path, str]):
        if self.inference_data is None:
            raise ValueError("No inference data to save.")
        path = Path(path)
        self.inference_data.to_netcdf(str(path.absolute()))

    # permits, eg, `lr = LinearRegressor.from_disk("trace.nc")`
    @classmethod
    def from_disk(cls, path: str, API_key: Optional[str] = None) -> _BaseModel:
        instance = cls(API_key=API_key)
        instance.inference_data = az.from_netcdf(path)
        return instance

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

# permits, eg, `lr = Model.from_disk("lr_model.netcdf")`
class Model(_BaseModel):
    """Helper class to generically load a model from disk
    based on its recorded `model_type` attribute.
    """
    @staticmethod
    def from_disk(path: Union[Path, str], API_key: Optional[str] = None) -> _BaseModel:
        """Load a saved model from disk and return an instance
        of the appropriate subclass based on its `model_type`.
        """
        path = Path(path)
        inference_data = az.from_netcdf(str(path.absolute()))

        model_type = inference_data.attrs.get("model_type")
        if not model_type:
            raise ValueError("Missing 'model_type' in InferenceData attrs.")

        model_cls = {
            "linear": LinearRegressor,
            "logistic": LogisticRegressor,
            "sturdy": SturdyLogisticRegressor,
        }.get(model_type)
        if not model_cls:
            raise ValueError(f"Unknown model_type: {model_type}")

        instance = model_cls(API_key=API_key)
        instance.inference_data = inference_data
        return instance

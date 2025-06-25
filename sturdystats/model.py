from __future__ import annotations

import numpy as np
import xarray as xr
import arviz as az

import requests
import os
import tempfile
from typing import Dict, Optional, Union, Callable
from requests.models import Response

from pathlib import Path
from scipy import special

import srsly

from sturdystats.job import Job


_base_url = "https://api.sturdystatistics.com/api/v1/numeric"

class RegressionResult(Job):
    """subclass of Job which fetches InferenceData for regression models"""
    def getTrace(self):
        """
        Wait for job completion and return the resulting InferenceData.
        Assumes job returns a NetCDF binary in 'result'.
        """
        bdata: bytes = self.wait()["result"] #type: ignore

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "trace.nc"
            path.write_bytes(bdata)

            # force az to load the file into memory since file will be deleted
            with az.rc_context(rc={"data.load": "eager"}):
                inference_data = az.from_netcdf(path)

        return inference_data

def _append_data(inference_data: az.InferenceData, X: np.ndarray, Y: np.ndarray) -> None:
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

def _predict_regression(
        dataset: az.InferenceData,
        X: np.ndarray,
        link: Callable[[np.ndarray], np.ndarray]
) -> xr.Dataset:
    """
    General posterior predictive computation for regression models.

    Parameters:
        dataset (az.InferenceData): InferenceData containing posterior `eta` and `b`.
        X (np.ndarray): Input features, shape (N, D).
        link (callable): Link function applied to linear predictor (e.g. expit, identity).

    Returns:
        xr.Dataset: Dataset containing 'Yp' with dimensions (chain, draw, N, Q).
    """
    X = np.array(X, copy=True)                            # sample=n dim=k

    pos = dataset.posterior

    eta = pos.eta.transpose("chain", "draw", "Q", "dim")  # chain=c draw=d Q=q dim=k
    eta = np.array(eta)

    assert eta.shape[-1] == X.shape[1], f"Dim mismatch: eta has {eta.shape[-1]}, X has {X.shape[1]}"

    # einsum: (n, k) x (c, d, Q, k) -> (n, c, d, Q)
    theta = np.einsum("nk,cdqk->ncdq", X, eta)            # sample=n chain=c draw=d Q=q

    b = pos.b.transpose("chain", "draw", "Q")             #          chain=c draw=d Q=q
    theta = theta + np.array(b)

    preds = link(theta)                                   # sample=n chain=c draw=d Q=q

    # assemble predictions into a dataset with named coordinates
    ds = xr.Dataset(
        data_vars={"Yp": (("N", "chain", "draw", "Q"), preds)},
        coords={
            "N":     1+np.arange(preds.shape[0]),
            "chain": 1+np.arange(preds.shape[1]),
            "draw":  1+np.arange(preds.shape[2]),
            "Q":     1+np.arange(preds.shape[3])}
    )

    # re-order coordinates to match arviz convention and return
    return xr.Dataset({"Yp": ds.Yp.transpose("chain", "draw", "N", "Q")})

def predict_logistic(dataset: az.InferenceData, X: np.ndarray) -> xr.Dataset:
    """
    Compute posterior predictive probabilities for a logistic regression model.

    This function applies the posterior samples of regression coefficients and
    intercepts from the provided InferenceData to new input data X, and returns
    the predicted probabilities as a DataArray with dimensions (chain, draw, N, Q).

    Parameters:
        dataset (az.InferenceData): Fitted model containing posterior samples of `eta` and `b`.
        X (np.ndarray): New input data of shape (N, D), where D matches the number of features.

    Returns:
        xr.Dataset: A dataset with one variable `Yp` representing posterior predictive
                    probabilities, with dimensions (chain, draw, N, Q).
    """
    return _predict_regression(dataset, X, link=special.expit)

def predict_linear(dataset: az.InferenceData, X: np.ndarray) -> xr.Dataset:
    """
    Compute posterior predictive samples for a linear regression model.

    This function applies the posterior samples of regression coefficients and
    intercepts from the provided InferenceData to new input data X, and returns
    the predicted probabilities as a DataArray with dimensions (chain, draw, N, Q).

    Parameters:
        dataset (az.InferenceData): Fitted model containing posterior samples of `eta` and `b`.
        X (np.ndarray): New input data of shape (N, D), where D matches the number of features.

    Returns:
        xr.Dataset: A dataset with one variable `Yp` representing posterior predictive
                    probabilities, with dimensions (chain, draw, N, Q).
    """
    return _predict_regression(dataset, X, link=lambda x: x)

class _BaseModel:
    def __init__(self, model_type: str, API_key: Optional[str] = None, _base_url: str = _base_url):
        self.API_key = API_key or os.environ["STURDY_STATS_API_KEY"]
        self.base_url = _base_url
        self.model_type = model_type
        self.inference_data: Optional[az.InferenceData] = None

    def __repr__(self):
        return f"<{self.__class__.__name__}(model_type={self.model_type!r}, fitted={self.inference_data is not None})>"

    def _require_inference_data(self):
        if self.inference_data is None:
            raise RuntimeError("Model has no inference data. Did you forget to call `.sample()` or `.from_disk()`?")

    def predict(self, X: np.ndarray, Y: Optional[np.ndarray] = None, save=False) -> np.ndarray:
        """
        Compute the posterior mean prediction for new input data X.

        Parameters:
            X (np.ndarray): New input features, shape (N, D).
            save (bool): If True, adds prediction results to self.inference_data under
                         'predictions' and 'predictions_constant_data'.

        Returns:
            xr.DataArray: Posterior mean prediction with shape (N, Q)
        """
        self._require_inference_data()
        ps = self.sample_posterior_predictive(X)

        if save:
            vars = {"X":  (("N", "D"), X)}

            coords = {"N": 1+np.arange(X.shape[0]),
                      "D": 1+np.arange(X.shape[1])}

            if Y is not None:
                assert(Y.shape[0] == X.shape[0])
                vars["Y"] = (("N", "Q"), Y)
                coords["Q"] = 1+np.arange(Y.shape[1])

            self.inference_data.add_groups(
                {
                    'predictions_constant_data': xr.Dataset(
                        data_vars=vars,
                        coords=coords),
                    'predictions': ps
                })

        if "Yp" not in ps:
            raise ValueError("Returned dataset from sample_posterior_predictive() must contain 'Yp'")

        return ps.mean(dim=('chain', 'draw'))

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> az.InferenceData:
        """
        Save predictions and ground truth labels for evaluation/comparison.
        """
        self.predict(X, Y, save=True)
        return self.inference_data

    def sample_posterior_predictive(self, X: np.ndarray) -> xr.Dataset:
        """
        Compute posterior predictive draws for new input data.

        Subclasses must implement this method to return a Dataset containing
        a variable 'Yp' with dimensions ('chain', 'draw', 'N', 'Q').

        Returns:
            xr.Dataset: Dataset containing variable 'Yp' with posterior predictive samples.
        """
        raise NotImplementedError(f"Predict not implemented for model type: {self.model_type}")

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
        X = np.array(X, copy=True).astype(np.float32)
        Y = np.array(Y, copy=True).astype(np.float32)
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

        # sample the posterior predictive
        self.inference_data.add_groups(
            posterior_predictive=self.sample_posterior_predictive(X))

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

    def sample_posterior_predictive(self, X: np.ndarray) -> xr.Dataset:
        self._require_inference_data()
        return predict_linear(self.inference_data, X)

class LogisticRegressor(_BaseModel):
    def __init__(self, API_key: Optional[str] = None, _base_url: str = _base_url):
        super().__init__("logistic", API_key, _base_url)

    def sample_posterior_predictive(self, X: np.ndarray) -> xr.Dataset:
        self._require_inference_data()
        return predict_logistic(self.inference_data, X)

class SturdyLogisticRegressor(_BaseModel):
    def __init__(self, API_key: Optional[str] = None, _base_url: str = _base_url):
        super().__init__("sturdy", API_key, _base_url)

    def sample_posterior_predictive(self, X: np.ndarray) -> xr.Dataset:
        self._require_inference_data()
        return predict_logistic(self.inference_data, X)

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

        _model_dispatch = {
            "linear": LinearRegressor,
            "logistic": LogisticRegressor,
            "sturdy": SturdyLogisticRegressor,
        }

        model_cls = _model_dispatch.get(model_type)
        if not model_cls:
            raise ValueError(f"Unknown model_type: {model_type!r}. Expected one of: {', '.join(_model_dispatch.keys())}")

        instance = model_cls(API_key=API_key)
        instance.inference_data = inference_data
        return instance

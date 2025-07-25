from __future__ import annotations

import numpy as np
import xarray as xr
import arviz as az

import requests
import os
import tempfile
from typing import Dict, Optional, Union, Callable, List
from requests.models import Response

from pathlib import Path
from scipy import special

import srsly

from sturdystats.job import Job


_base_url = "https://api.sturdystatistics.com/api/v1/numeric"

class RegressionResult(Job):
    """subclass of Job which fetches InferenceData for regression models"""

    def __init__(self, API_key: str, 
                 job_id: str, poll_seconds: int = 1, 
                 msgpack: bool = True,
                 _base_url: str= "https://api.sturdystatistics.com/api/v1/job",
                 label_names: Optional[List] = None,
                 feature_names: Optional[List] = None,
                 sample_posterior_predictive = xr.Dataset,
                 X: np.ndarray,
                 Y: np.ndarray):
        super().__init__(API_key=API_key,
                         job_id=job_id,
                         poll_seconds=poll_seconds,
                         msgpack=msgpack,
                         _base_url=_base_url)
        self.label_names = label_names
        self.feature_names = feature_names
        self.X = X
        self.Y = Y
        self.sample_posterior_predictive = sample_posterior_predictive
        

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


        inference_data = inference_data.assign_coords({"Q": self.label_names, "dim": self.feature_names})
        inference_data.attrs["model_type"] = self.model_type
        inference_data.attrs["label_names"] = list(self.label_names)
        inference_data.attrs["feature_names"] = list(self.feature_names)

        # add constant data along with the posterior predictive
        _append_data(inference_data, self.X, self.Y)
        inference_data.add_groups(
            posterior_predictive=self.sample_posterior_predictive)

                
        return inference_data

def _append_data(
        inference_data: az.InferenceData,
        X: np.ndarray,
        Y: np.ndarray,
) -> None:
    """Add training data (constant & observed) to an ArviZ InferenceData object.
    This modifies `inference_data` in-place and returns None.
    """

    # NB this function is only called by `sample`.  X and Y are already validated.

    label_names = inference_data.attrs.get("label_names")
    assert (label_names is not None) and (len(label_names) == Y.shape[1])

    feature_names = inference_data.attrs.get("feature_names")
    assert (feature_names is not None) and (len(feature_names) == X.shape[1])

    inference_data.add_groups(
        {
            'constant_data': xr.Dataset(
                data_vars={
                    "X":  (("N", "dim"), X),
                },
                coords={
                    "N": 1+np.arange(X.shape[0]),
                    "dim": feature_names,
                }),
            'observed_data': xr.Dataset(
                data_vars={
                    "Y":  (("N", "Q"), Y),
                },
                coords={
                    "N": 1+np.arange(Y.shape[0]),
                    "Q": label_names,
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
        X (np.ndarray): Input features, shape (N, dim).
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

    label_names = dataset.attrs.get("label_names")
    assert (label_names is not None) and (len(label_names) == preds.shape[3])

    # assemble predictions into a dataset with named coordinates
    ds = xr.Dataset(
        data_vars={"Yp": (("N", "chain", "draw", "Q"), preds)},
        coords={
            "N":     1+np.arange(preds.shape[0]),
            "chain": 1+np.arange(preds.shape[1]),
            "draw":  1+np.arange(preds.shape[2]),
            "Q":     label_names}
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
        X (np.ndarray): New input data of shape (N, dim), where dim matches the number of features.

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
        X (np.ndarray): New input data of shape (N, dim), where dim matches the number of features.

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
            X (np.ndarray): New input features, shape (N, dim).
            save (bool): If True, adds prediction results to self.inference_data under
                         'predictions' and 'predictions_constant_data'.

        Returns:
            xr.DataArray: Posterior mean prediction with shape (N, Q)
        """
        self._require_inference_data()
        ps = self.sample_posterior_predictive(X)

        if save:
            X = np.array(X)  # cast to np array; don't need to copy here

            feature_names = self.inference_data.attrs.get("feature_names")
            assert (feature_names is not None) and (len(feature_names) == X.shape[1])

            vars = {"X":  (("N", "dim"), X)}

            coords = {"N": 1+np.arange(X.shape[0]),
                      "dim": feature_names}

            if Y is not None:
                assert Y.shape[0] == X.shape[0]
                label_names = self.inference_data.attrs.get("label_names")
                assert (label_names is not None) and (len(label_names) == Y.shape[1])
                vars["Y"] = (("N", "Q"), Y)
                coords["Q"] = label_names

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

    def sample(self, X, Y,
               label_names: Optional[List] = None,
               feature_names: Optional[List] = None,
               additional_args: str = "",
               background=False):

        # validate input data
        assert len(X) == len(Y), (
            f"Mismatch in number of samples: X has {len(X)} rows but Y has {len(Y)} rows. "
            "Ensure that the feature matrix X and target array Y have the same number of rows."
        )

        X = np.array(X, copy=True) # force numpy array and
        Y = np.array(Y, copy=True) # copy to avoid bugs if X or Y is a view

        if label_names is not None:
            assert len(label_names) == Y.shape[1], (
                f"Mismatch in label dimensions: Y has {Y.shape[1]} columns (labels) but "
                f"{len(label_names)} label names were provided. "
                "Each column in Y should have a corresponding label name."
            )
        else:
            label_names = 1+np.arange(Y.shape[1])

        if feature_names is not None:
            assert len(feature_names) == X.shape[1], (
                f"Mismatch in feature dimensions: X has {X.shape[1]} columns (features) but "
                f"{len(feature_names)} feature names were provided. "
                "Each column in X should have a corresponding feature name."
            )
        else:
            feature_names = 1+np.arange(X.shape[1])

        data = dict(X=X, Y=Y, override_args=additional_args)

        # submit training job and make a job object
        job_id = self._post(f"/{self.model_type}", data).json()["job_id"]
        sample_posterior_predictive = self.sample_posterior_predictive(X)
        job = RegressionResult(API_key=self.API_key, msgpack=True, job_id=job_id, _base_url=self._job_base_url(),
                               label_names = label_names, feature_names = feature_names, X = X, Y = Y, sample_posterior_predictive = sample_posterior_predictive)

        # run in background: return job object
        if background:
            return job

        # wait for results: unpack into arviz dataset
        inference_data = job.getTrace()
        inference_data = inference_data.assign_coords({"Q": label_names, "dim": feature_names})
        inference_data.attrs["model_type"] = self.model_type
        inference_data.attrs["label_names"] = list(label_names)
        inference_data.attrs["feature_names"] = list(feature_names)
        self.inference_data = inference_data

        # add constant data along with the posterior predictive
        _append_data(self.inference_data, X, Y)
        self.inference_data.add_groups(
            posterior_predictive=sample_posterior_predictive)

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

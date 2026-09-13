from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
'''
# Anomaly Detection Toolkit (ADTK)

[![Build Status](https://travis-ci.com/arundo/adtk.svg?branch=master)](https://travis-ci.com/arundo/adtk)
[![Documentation Status](https://readthedocs.org/projects/adtk/badge/?version=stable)](https://adtk.readthedocs.io/en/stable)
[![Coverage Status](https://coveralls.io/repos/github/arundo/adtk/badge.svg?branch=master&service=github)](https://coveralls.io/github/arundo/adtk?branch=master)
[![PyPI](https://img.shields.io/pypi/v/adtk)](https://pypi.org/project/adtk/)
[![Downloads](https://pepy.tech/badge/adtk)](https://pepy.tech/project/adtk)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arundo/adtk/master?filepath=docs%2Fnotebooks%2Fdemo.ipynb)

Anomaly Detection Toolkit (ADTK) is a Python package for unsupervised /
rule-based time series anomaly detection.

As the nature of anomaly varies over different cases, a model may not work
universally for all anomaly detection problems. Choosing and combining
detection algorithms (detectors), feature engineering methods (transformers),
and ensemble methods (aggregators) properly is the key to build an effective
anomaly detection model.

This package offers a set of common detectors, transformers and aggregators
with unified APIs, as well as pipe classes that connect them together into
models. It also provides some functions to process and visualize time series
and anomaly events.

See https://adtk.readthedocs.io for complete documentation.

## Installation

Prerequisites: Python 3.5 or later.

It is recommended to install the most recent **stable** release of ADTK from PyPI.

```shell
pip install adtk
```

Alternatively, you could install from source code. This will give you the **latest**, but unstable, version of ADTK.

```shell
git clone https://github.com/arundo/adtk.git
cd adtk/
git checkout develop
pip install ./
```

## Examples

Please see [Quick Start](https://adtk.readthedocs.io/en/stable/quickstart.html) for a simple example.

For more detailed examples of each module of ADTK, please refer to
[Examples](https://adtk.readthedocs.io/en/stable/examples.html)
section in the documentation or [an interactive demo notebook](https://mybinder.org/v2/gh/arundo/adtk/master?filepath=docs%2Fnotebooks%2Fdemo.ipynb).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update unit tests as appropriate.

Please see [Contributing](https://adtk.readthedocs.io/en/stable/developer.html) for more details.


## License

ADTK is licensed under the Mozilla Public License 2.0 (MPL 2.0). See the
[LICENSE](LICENSE) file for details.

'''

class _Model(ABC):
    "Base class for all models (detectors, transformers, and aggregators)."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of this model.

        Returns
        -------
        dict
            Model parameters.

        """
        return {key: getattr(self, key) for key in self._param_names}

    def set_params(self, **params: Any) -> None:
        """Set the parameters of this model.

        Parameters
        ----------
        **params
            Model parameters to set.

        """
        for key in params.keys():
            if key not in self._param_names:
                raise KeyError(
                    "'{}' is not a valid parameter name.".format(key)
                )
        for key, value in params.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def _param_names(self) -> Tuple[str, ...]:
        return tuple()


class _NonTrainableModel(_Model):
    "Base class of models that do not need training."

    @abstractmethod
    def _predict(self, input: Any) -> Any:
        pass

    @abstractmethod
    def _predict_core(self, input: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, input: Any) -> Any:
        pass


class _TrainableModel(_Model):
    "Base class of models that need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # 0 for not fitted, 1 for fitted, 2 for univariate model fitted by DF
        self._fitted = 0  # type: int

    @abstractmethod
    def _fit(self, input: Any) -> None:
        pass

    @abstractmethod
    def _fit_core(self, input: Any) -> None:
        pass

    @abstractmethod
    def fit(self, input: Any) -> None:
        pass

    @abstractmethod
    def _predict(self, input: Any) -> Any:
        pass

    @abstractmethod
    def _predict_core(self, input: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, input: Any) -> Any:
        pass

    @abstractmethod
    def fit_predict(self, input: Any) -> Any:
        pass


class _NonTrainableUnivariateModel(_NonTrainableModel):
    "Base class of univariate detectors and transformers."

    def _predict(
        self, ts: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(ts, pd.Series):
            s = ts.copy()  # type: pd.Series
            if not isinstance(s.index, pd.DatetimeIndex):
                raise TypeError(
                    "Index of the input time series must be a pandas "
                    "DatetimeIndex object."
                )
            predicted = self._predict_core(s)
            # if a Series-to-Series operation, make sure Series name keeps
            if isinstance(predicted, pd.Series):
                predicted.name = ts.name
        elif isinstance(ts, pd.DataFrame):
            df = ts.copy()  # type: pd.DataFrame
            if df.columns.duplicated().any():
                raise ValueError(
                    "Input DataFrame must have unique column names."
                )
            # apply the model to each column
            predicted_all_cols = []
            for col in df.columns:
                predicted_this_col = self._predict(df[col])
                # if a Series-to-DF operation, update column name
                if isinstance(predicted_this_col, pd.DataFrame):
                    predicted_this_col = predicted_this_col.rename(
                        columns={
                            col1: "{}_{}".format(col, col1)
                            for col1 in predicted_this_col.columns
                        }
                    )
                predicted_all_cols.append(predicted_this_col)
            predicted = pd.concat(predicted_all_cols, axis=1)
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")
        # make sure index freq is the same (because pandas has a bug that some
        # operation, e.g. concat, may change freq)
        predicted.index.freq = ts.index.freq
        return predicted


class _TrainableUnivariateModel(_TrainableModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._models = dict()  # type: Dict[str, _TrainableUnivariateModel]

    def _fit(self, ts: Union[pd.Series, pd.DataFrame]) -> None:
        if isinstance(ts, pd.Series):
            s = ts.copy()  # type: pd.Series
            self._fit_core(s)
            self._fitted = 1
        elif isinstance(ts, pd.DataFrame):
            df = ts.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError(
                    "Index of the input time series must be a pandas "
                    "DatetimeIndex object."
                )
            if df.columns.duplicated().any():
                raise ValueError(
                    "Input DataFrame must have unique column names."
                )
            # create internal models
            self._models = {
                col: self.__class__(**deepcopy(self.get_params()))
                for col in df.columns
            }
            # fit model for each column
            for col in df.columns:
                self._models[col].fit(df[col])
            self._fitted = 2
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")

    def _predict(
        self, ts: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        if self._fitted == 0:
            raise RuntimeError("The model must be trained first.")

        if isinstance(ts, pd.Series):
            if self._fitted == 2:
                raise RuntimeError(
                    "The model was trained by a pandas DataFrame object, "
                    "it can only be applied to a pandas DataFrame object with "
                    "the same column names as the one used for training."
                )
            s = ts.copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                raise TypeError(
                    "Index of the input time series must be a pandas "
                    "DatetimeIndex object."
                )
            predicted = self._predict_core(s)
            # if a Series-to-Series operation, make sure Series name keeps
            if isinstance(predicted, pd.Series):
                predicted.name = ts.name
        elif isinstance(ts, pd.DataFrame):
            df = ts.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError(
                    "Index of the input time series must be a pandas "
                    "DatetimeIndex object."
                )
            if df.columns.duplicated().any():
                raise ValueError(
                    "Input DataFrame must have unique column names."
                )
            if self._fitted == 1:
                # apply the model to each column
                predicted_all_cols = []
                for col in df.columns:
                    predicted_this_col = self._predict(df[col])
                    if isinstance(predicted_this_col, pd.DataFrame):
                        predicted_this_col = predicted_this_col.rename(
                            columns={
                                col1: "{}_{}".format(col, col1)
                                for col1 in predicted_this_col.columns
                            }
                        )
                    predicted_all_cols.append(predicted_this_col)
                predicted = pd.concat(predicted_all_cols, axis=1)
            else:
                # predict for each column
                if not (set(self._models.keys()) >= set(df.columns)):
                    raise ValueError(
                        "The model was trained by a pandas DataFrame with "
                        "columns {}, but the input DataFrame contains columns "
                        "{} which are unknown to the model.".format(
                            list(set(self._models.keys())),
                            list(set(df.columns) - set(self._models.keys())),
                        )
                    )
                predicted = pd.concat(
                    [
                        self._models[col]._predict(df[col])
                        for col in df.columns
                    ],
                    axis=1,
                )
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")
        # make sure index freq is the same (because pandas has a bug that some
        # operation, e.g. concat, may change freq)
        predicted.index.freq = ts.index.freq
        return predicted


class _NonTrainableMultivariateModel(_NonTrainableModel):
    def _predict(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(df, pd.DataFrame):
            if df.columns.duplicated().any():
                raise ValueError(
                    "Input DataFrame must have unique column names."
                )
            df_copy = df.copy()
            predicted = self._predict_core(df_copy)
        else:
            raise TypeError("Input must be a pandas DataFrame.")
        # make sure index freq is the same (because pandas has a bug that some
        # operation, e.g. concat, may change freq)
        predicted.index.freq = df.index.freq
        return predicted


class _TrainableMultivariateModel(_TrainableModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cols = []  # type: List[str]

    def _fit(self, df: pd.DataFrame) -> None:
        if isinstance(df, pd.DataFrame):
            if df.columns.duplicated().any():
                raise ValueError(
                    "Input DataFrame must have unique column names."
                )
            df_copy = df.copy()
            self._fit_core(df_copy)
        else:
            raise TypeError("Input must be a pandas DataFrame.")
        self._cols = list(df.columns)
        self._fitted = 1

    def _predict(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        if self._fitted == 0:
            raise RuntimeError("The model must be trained first.")
        if isinstance(df, pd.DataFrame):
            if df.columns.duplicated().any():
                raise ValueError(
                    "Input DataFrame must have unique column names."
                )
            if not (set(df.columns) >= set(self._cols)):
                raise ValueError(
                    "The model was trained by a pandas DataFrame with columns "
                    "{}, but the input DataFrame does not contain columns {}.".format(
                        self._cols, list(set(self._cols) - set(df.columns))
                    )
                )
            df_copy = (
                df.loc[:, self._cols].copy() if self._cols else df.copy()
            )  # in a customized hd model that doesn't need fit, self._cols is empty
            predicted = self._predict_core(df_copy)
        else:
            raise TypeError("Input must be a pandas DataFrame.")
        # make sure index freq is the same (because pandas has a bug that some
        # operation, e.g. concat, may change freq)
        predicted.index.freq = df.index.freq
        return predicted

from pathlib import Path

import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series import DefaultFeatures, FeatureTransformer, TabPFNMode, TabPFNTimeSeriesPredictor
from tabpfn_time_series.data_preparation import generate_test_X

from tabpfn_custom import TabPFNWorkerCustom

data = pd.read_csv("/Users/user/Downloads/data.csv")
data = data.rename(columns={"value": "target"})
start_date = pd.Timestamp(day=1, month=1, year=2020)
data["timestamp"] = pd.period_range(start=start_date, end=start_date + pd.Timedelta(days=data.shape[0] - 1))
data["timestamp"] = data["timestamp"].apply(lambda x: x.to_timestamp())
data["item_id"] = 0


def load_tabfpn(model_path: Path) -> TabPFNTimeSeriesPredictor:
    predictor = TabPFNTimeSeriesPredictor(
        tabpfn_mode=TabPFNMode.LOCAL,
        config={
            "tabpfn_internal": {
                "model_path": model_path,
            }
        },
    )
    predictor.tabpfn_worker = TabPFNWorkerCustom(predictor.tabpfn_worker.config)
    return predictor


def get_ts_to_vec_tabpfn_dim() -> int:
    return 768


def ts_to_vec_tabfpn(ts_matrix: torch.Tensor, model: TabPFNTimeSeriesPredictor) -> torch.Tensor:
    verify_ts_matrix(ts_matrix)
    is_one_feat = ts_matrix.shape[1] == 1
    if not is_one_feat:
        raise ValueError("Current tabpfn implementation supports only time series with one feature")

    res = torch.empty((*ts_matrix.shape[:2], get_ts_to_vec_tabpfn_dim()))

    for ts_i, (cur_ts) in enumerate(ts_matrix):
        cur_ts_df = pd.DataFrame(cur_ts.reshape(-1), columns=["target"])
        random_start_date = pd.Timestamp(day=1, month=1, year=2020)
        cur_ts_df["timestamp"] = pd.date_range(
            start=random_start_date, end=random_start_date + pd.Timedelta(days=data.shape[0] - 1)
        )
        cur_ts_df["item_id"] = 0
        cur_ts_df = TimeSeriesDataFrame(cur_ts_df)
        cur_ts_df["month"] = cur_ts_df.dt.month

        for cur_step in range(cur_ts_df.shape[0]):
            cur_ts_df_slice = cur_ts_df.iloc[: cur_step + 1]
            cur_query_df = generate_test_X(cur_ts_df_slice, prediction_length=1)
            cur_pred = model.predict(cur_ts_df_slice, cur_query_df)
            res[ts_i, cur_step] = cur_pred

    return res


data, data_test = FeatureTransformer.add_features(data, data_test, [DefaultFeatures.add_calendar_features])
pred
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import split_time_series_to_X_y
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG
from tabpfn_time_series.tabpfn_worker import LocalTabPFN
from torch import nn


class LayerOutputHook:
    def __init__(self) -> None:
        self.outputs: list[torch.Tensor] = []

    def __call__(self, module: nn.Module, input_: torch.Tensor, output: torch.Tensor) -> None:
        self.outputs.append(output)


class TabPFNWorkerCustom(LocalTabPFN):
    def __init__(
        self,
        config: dict = {},
        num_workers: int = 1,
    ):
        self.config = config
        self.num_workers = num_workers

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        quantile_config: list[float],
    ) -> TimeSeriesDataFrame:
        if not set(quantile_config).issubset(set(TABPFN_TS_DEFAULT_QUANTILE_CONFIG)):
            raise NotImplementedError(
                f"We currently only supports {TABPFN_TS_DEFAULT_QUANTILE_CONFIG} for quantile prediction,"
                f" but got {quantile_config}."
            )

        predictions = [
            self._prediction_routine(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
                quantile_config,
            )
            for item_id in train_tsdf.item_ids
        ]

        predictions = pd.concat(predictions)

        # Sort predictions according to original item_ids order (important for MASE and WQL calculation)
        predictions = predictions.loc[train_tsdf.item_ids]

        return TimeSeriesDataFrame(predictions)

    def _prediction_routine(
        self,
        item_id: str,
        single_train_tsdf: TimeSeriesDataFrame,
        single_test_tsdf: TimeSeriesDataFrame,
        quantile_config: list[float],
    ) -> pd.DataFrame:
        test_index = single_test_tsdf.index
        train_X, train_y = split_time_series_to_X_y(single_train_tsdf.copy())
        test_X, _ = split_time_series_to_X_y(single_test_tsdf.copy())
        train_y = train_y.squeeze()

        train_y_has_constant_value = train_y.nunique() == 1
        if train_y_has_constant_value:
            result = self._predict_on_constant_train_target(single_train_tsdf, single_test_tsdf, quantile_config)
        else:
            tabpfn = self._get_tabpfn_engine()
            tabpfn.fit(train_X, train_y)

            embed_hook = LayerOutputHook()
            embed_layer: nn.Linear = tabpfn.model_.decoder_dict.standard[0]
            hook_handle = embed_layer.register_forward_hook(embed_hook)
            tabpfn.predict(test_X, output_type="full")
            embed: np.ndarray = torch.concat(embed_hook.outputs, dim=1).numpy()
            # (seq_len, feat_dim, embed_dim)
            assert embed.ndim == 3
            embed = embed.reshape(embed.shape[0], -1)
            result = {f"feat_{i}": feat for i, feat in enumerate(embed.T)}
            hook_handle.remove()

        result = pd.DataFrame(result, index=test_index)
        result["item_id"] = item_id
        result.set_index(["item_id", result.index], inplace=True)
        return result

    def _parse_model_path(self, model_name: str) -> str:
        if Path(model_name).exists():
            return model_name
        return f"tabpfn-v2-regressor-{model_name}.ckpt"

""""Contains the configured final Model as well as a baseline model"""

import polars as pl

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import HuberLoss

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape, rmse

class FinalModel():
    def __init__(self, h) -> None:
        ins = h*18
        ms = 1200

        nbeats_params = {
            "h": h,
            "input_size": ins,
            "loss": HuberLoss(),
            "max_steps": ms,
            "enable_model_summary": False,
            "enable_checkpointing": False,
            "logger": False,
            "accelerator": "gpu",
            "devices": 1,
            "batch_size": 128,
            "learning_rate": 1589e-7,
            "mlp_units": [[256, 256]]*2,
            'n_pool_kernel_size': (16, 8, 1),
            'n_freq_downsample': (168, 24, 1),
        }
        self.model  = NeuralForecast(
            models = [NHITS  (**nbeats_params)], #type: ignore
            freq = "1d",
            local_scaler_type="robust"
        )
    def fit(self, df: pl.DataFrame)-> None:
        print("impliment fit")

    def predict(self, h: int, future_features: pl.DataFrame|None) -> pl.DataFrame:
        print("impliment predict")

    def get_metrics(self, true: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
        print("impliment metrics")

class BaseLineModel():
    def __init__(self, season_length: int = 7) -> None:
        self.model = StatsForecast(models=[SeasonalNaive(season_length)], freq='1d')
    
    def fit(self, df: pl.DataFrame)-> None:
        self.model.fit(df) #type: ignore

    def predict(self, h: int, future_features: pl.DataFrame|None) -> pl.DataFrame:
        return self.model.predict(h, future_features) #type: ignore

    def get_metrics(self, true: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
        return evaluate(
            true.join(predictions, on=["unique_id", "ds"]),
            metrics=[mae, mape, rmse], # List the metrics you want
        )
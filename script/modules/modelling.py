""""Contains the configured final Model as well as a baseline model"""

import polars as pl

class Models():
    def __init__(self) -> None:
        pass

    def fit(self, df: pl.DataFrame)-> None:
        pass

    def predict(self, h: int, future_features: pl.DataFrame) -> pl.DataFrame:
        pass

    def get_metrics(self, true: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
        pass

class FinalModel(Models):
    def __init__(self) -> None:
        super().__init__()

class BaseLineModel(Models):
    def __init__(self) -> None:
        super().__init__()

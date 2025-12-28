"""The script for processing raw data into a prediction"""
import tomllib
import sys
import os
scripts_dir_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, scripts_dir_path)
base_dir = scripts_dir_path + "/../"

from modules import dataprocessing as data
from modules.modelling import FinalModel, BaseLineModel
from modules import visialising as vis


# Data ingestion
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

df = data.ingest( base_dir+config["data"]["path"])
df = data.preprocess(df, end_date=config["predictions"]["predict_from"])

# Train test splitt and model eval
train, test = data.train_test_split(df, config["predictions"]["predict_from"], 31)

train, future_features = data.add_features(train)

model = FinalModel()
baseline = BaseLineModel()
model.fit(train)
baseline.fit(train)
prediction = model.predict(31, future_features)
base_pred = baseline.predict(31, future_features)
metrics = model.get_metrics(test, train)
prediction.write_parquet(base_dir+config["predictions"]["path"]+"name.parquet")

"""The script for processing raw data into a prediction"""
import tomllib
import sys
import os
from datetime import date
from modules import dataprocessing as data
from modules.modelling import FinalModel, BaseLineModel
# from modules import visialising as vis
scripts_dir_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, scripts_dir_path)
base_dir = scripts_dir_path + "/../"

# Data ingestion
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# calculate the correct horizon, needed for us if we have data before the 31st
delta = date.fromisoformat(config["predictions"]["predict_until"]
                           ) - date.fromisoformat(config["predictions"]["predict_from"])
HORIZON = int(delta.days)

df = data.ingest( base_dir+config["data"]["path"])
df = data.preprocess(df, end_date=config["predictions"]["predict_from"])

# Train test splitt and model eval
train, test = data.train_test_split(df, config["predictions"]["predict_from"], HORIZON)

train, future_features = data.add_features(train, HORIZON) # turns out no features needed

model = FinalModel(HORIZON)
baseline = BaseLineModel()

model.fit(train)
baseline.fit(train)

prediction = model.predict( future_features)
base_pred = baseline.predict(HORIZON, future_features)

metrics = model.get_metrics(test, base_pred.join(prediction, on=["unique_id", "ds"]))
metrics.to_pandas().to_markdown("./metrics.md") # export the metrics for user to see

print("Metrics: \n", metrics)

base_pred.join(prediction, on=["unique_id", "ds"]
        ).join(test, on=["unique_id", "ds"]
        ).write_parquet("../"+config["predictions"]["path"]+"script_test_pred.parquet"
        ) # write the testing results

# Predict January

features_df, future_features = data.add_features(df, HORIZON) # turns out no features needed
model.fit(features_df)
jan_pred = model.predict(future_features)

jan_pred.write_parquet("../"+config["predictions"]["path"]+"script_january_pred.parquet")
jan_pred.write_csv("../"+config["predictions"]["path"]+"script_january_pred.csv")

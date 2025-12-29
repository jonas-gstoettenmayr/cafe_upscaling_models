""" Model for processing the Data"""

from datetime import timedelta
from typing import Tuple
import polars as pl


def ingest(path: str) -> pl.DataFrame:
    """
    Read pageviews CSV and convert it to long format with a Date column.
    """
    df = pl.read_csv(path)

    df_long = (
        df
        .unpivot(
            index=["Date"],
            variable_name="Drink",
            value_name="Views"
        )
        .with_columns(
            pl.col("Date").cast(pl.Date)
        )
    )
    df_long = df_long.rename({"Date": "ds", "Drink": "unique_id", "Views": "y"})

    return df_long

def preprocess(df: pl.DataFrame, end_date: None|str = None) -> pl.DataFrame:
    """takes long format and processes it, i.e. capping outliers, renaming for nixtla

    Args:
        df (pl.DataFrame): long format data
        end_date (None|str): None if all data is to be used,
        datestring if data should be limited to certain date, inclusive

    Returns:
        pl.DataFrame: processed df still long
    """
    df_without_pumpkin_spice, df_pumpkin_spice = _filter_out_drink(df, "Pumpkin spice latte")
    df_iqr = _cap_outliers_iqr(df_without_pumpkin_spice, factor=1.5)
    df_capped = _add_drink_back(df_iqr, df_pumpkin_spice)
    df_capped = df_capped.sort(['unique_id', 'ds'])

    if end_date:
        end = pl.lit(end_date).str.to_date()
        df_capped = df_capped.filter(pl.col("ds") <= end)

    return df_capped

def train_test_split(df: pl.DataFrame, test_end: str = "2025-12-01", test_length: int = 31
                    )-> Tuple[pl.DataFrame, pl.DataFrame]:
    """splits the data into train and test, it also limits to how far the data can go

    Args:
        df (pl.DataFrame): df to split
        test_start (str): where the test data starts.
        test_length (int) = how many days are in the test set
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: train, test
    """
    test_start = pl.lit(test_end).str.to_date() - timedelta(test_length-1)
    return df.filter(pl.col("ds") < test_start), df.filter(pl.col("ds") >= test_start)

def add_features(df: pl.DataFrame, h:int = 31 # pylint: disable=unused-argument
                 ) ->Tuple[pl.DataFrame, pl.DataFrame |None]:
    """adds features to a df and returns a future df for the horizon of the featues"""
    return df, None # the best model has no features ¯\(°_o)/¯


def _filter_out_drink(df: pl.DataFrame, drink_name: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filter out a specific drink from the dataframe
    
    Args:
        df: Input dataframe
        drink_name: Name of the drink to filter out (Drink)
        
    Returns:
        Tuple of (df_without_drink, df_drink_only)
    """
    df_without_drink = df.filter(pl.col("unique_id") != drink_name)
    df_drink_only = df.filter(pl.col("unique_id") == drink_name)

    return df_without_drink, df_drink_only

def _add_drink_back(df_processed: pl.DataFrame, df_drink: pl.DataFrame) -> pl.DataFrame:
    """
    Add a drink back into the processed dataframe
    
    Args:
        df_processed: Processed dataframe (after outlier capping, etc.)
        df_drink: Drink dataframe to add back
        
    Returns:
        Combined dataframe with drink added back
    """
    # Cast df_drink's 'Views' column to match df_processed's type
    df_drink = df_drink.with_columns(
        pl.col("y").cast(df_processed["y"].dtype)
    )
    df_combined = pl.concat([df_processed, df_drink])

    # Sort by Drink and Date to keep it organized
    df_combined = df_combined.sort(["unique_id", "ds"])

    return df_combined

def _cap_outliers_iqr(df, column='y', group_by='unique_id', factor=1.5):
    """Cap outliers using IQR method for each drink separately"""

    result = df.clone()

    for drink in df[group_by].unique():
        mask = df[group_by] == drink
        values = df.filter(mask)[column]

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Cap the values
        result = result.with_columns(
            pl.when((pl.col(group_by) == drink) & (pl.col(column) > upper_bound))
            .then(upper_bound)
            .when((pl.col(group_by) == drink) & (pl.col(column) < lower_bound))
            .then(lower_bound)
            .otherwise(pl.col(column))
            .alias(column)
        )

    return result

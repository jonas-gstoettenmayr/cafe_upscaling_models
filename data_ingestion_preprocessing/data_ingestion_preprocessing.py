import polars as pl
import matplotlib.pyplot as plt
from typing import Tuple
from typing import Dict
from utilsforecast.feature_engineering import fourier, trend
import holidays

import polars as pl


def load_and_unpivot_pageviews(csv_path: str, date_col: str = "Date") -> pl.DataFrame:
    """
    Read pageviews CSV and convert it to long format with a Date column.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    date_col : str, default="Date"
        Name of the date column in the CSV.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns:
        - Date (pl.Date)
        - Drink (str)
        - Views (numeric)
    """
    df = pl.read_csv(csv_path)

    df_long = (
        df
        .unpivot(
            index=[date_col],
            variable_name="Drink",
            value_name="Views"
        )
        .with_columns(
            pl.col(date_col).cast(pl.Date)
        )
    )

    return df_long


def filter_out_drink(df: pl.DataFrame, drink_name: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filter out a specific drink from the dataframe
    
    Args:
        df: Input dataframe
        drink_name: Name of the drink to filter out (Drink)
        
    Returns:
        Tuple of (df_without_drink, df_drink_only)
    """
    df_without_drink = df.filter(pl.col("Drink") != drink_name)
    df_drink_only = df.filter(pl.col("Drink") == drink_name)
    
    print(f"Filtered out '{drink_name}':")
    print(f"  Remaining drinks: {df_without_drink.shape[0]} rows")
    print(f"  Filtered drink: {df_drink_only.shape[0]} rows")
    
    return df_without_drink, df_drink_only

def add_drink_back(df_processed: pl.DataFrame, df_drink: pl.DataFrame) -> pl.DataFrame:
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
        pl.col("Views").cast(df_processed["Views"].dtype)
    )
    
    df_combined = pl.concat([df_processed, df_drink])
    
    # Sort by Drink and Date to keep it organized
    df_combined = df_combined.sort(["Drink", "Date"])
    
    print(f"Added drink back:")
    print(f"  Total rows: {df_combined.shape[0]}")
    print(f"  Total drinks: {df_combined['Drink'].n_unique()}")
    
    return df_combined


def cap_outliers_iqr(df, column='Views', group_by='Drink', factor=1.5):
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

        #rounding it so it is a whole number (as there cannot be half clicks)
        lower_bound = lower_bound.round(0)
        upper_bound = upper_bound.round(0)
        
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

def cap_outliers_percentile(df, column='Views', lower=1, upper=99):
    """Cap outliers using percentile method"""
    
    lower_bound = df[column].quantile(lower/100)
    upper_bound = df[column].quantile(upper/100)

    #rounding it so it is a whole number (as there cannot be half clicks)
    lower_bound = lower_bound.round(0)
    upper_bound = upper_bound.round(0)
    
    return df.with_columns(
        pl.col(column).clip(lower_bound, upper_bound).alias(column)
    )


def cap_outliers_zscore(df, column='Views', threshold=3):
    """Cap outliers using z-score method"""
    
    mean = df[column].mean()
    std = df[column].std()
    
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    #rounding it so it is a whole number (as there cannot be half clicks)
    lower_bound = lower_bound.round(0)
    upper_bound = upper_bound.round(0)
    
    return df.with_columns(
        pl.col(column).clip(lower_bound, upper_bound).alias(column)
    )


def prepare_forecasting_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sort and rename columns for forecasting models.

    Expected input columns
    - Date
    - Drink
    - Views

    Returns
    pl.DataFrame
        DataFrame with columns:
        - unique_id
        - ds
        - y
        Sorted by unique_id, ds
    """
    return (
        df
        .sort(["Drink", "Date"])
        .rename({
            "Date": "ds",
            "Drink": "unique_id",
            "Views": "y",
        })
    )


def create_holiday_features(start_date: str = "2020-07-01", end_date: str = "2026-01-31") -> pl.DataFrame:
    """
    Create holiday features for the US (adjust country as needed)
    
    Args:
        start_date: Start date for holiday range
        end_date: End date for holiday range
    
    Returns:
        DataFrame with holiday information
    """
    # Get US holidays (change to your country if needed)
    us_holidays = holidays.US(years=range(2020, 2027))
    
    # Create date range
    date_range = pl.date_range(
        start = pl.Series([start_date]).cast(pl.Date)[0],
        end = pl.Series([end_date]).cast(pl.Date)[0],
        interval="1d",
        eager=True
    )
    
    # Create holiday dataframe
    holiday_df = pl.DataFrame({
        "ds": date_range
    })
    
    # Add holiday indicators
    holiday_df = holiday_df.with_columns([
        # Is it a holiday?
        pl.col("ds").map_elements(lambda x: x in us_holidays, return_dtype=pl.Boolean).alias("is_holiday"),
        
        # Holiday name (empty string if not a holiday)
        pl.col("ds").map_elements(lambda x: us_holidays.get(x, ""), return_dtype=pl.String).alias("holiday_name"),
        
        # Days until next holiday
        pl.col("ds").map_elements(
            lambda x: min([abs((h - x).days) for h in us_holidays.keys() if (h - x).days > 0] + [365]),
            return_dtype=pl.Int64
        ).alias("days_to_holiday"),
        
        # Days since last holiday
        pl.col("ds").map_elements(
            lambda x: min([abs((x - h).days) for h in us_holidays.keys() if (x - h).days > 0] + [365]),
            return_dtype=pl.Int64
        ).alias("days_since_holiday"),
    ])
    
    # Count number of holidays in a rolling window (e.g., next 7 days)
    holiday_df = holiday_df.with_columns([
        pl.col("is_holiday").cast(pl.Int64).alias("num_holidays")
    ])
    
    return holiday_df

def split_train_test(df: pl.DataFrame, val_start: str = "2024-01-01", val_end: str = "2025-01-01") -> Dict[str, pl.DataFrame]:
    """
    Split data into train and validation sets
    
    Args:
        df: Input dataframe in utilsforecast format
        val_start: Start date for validation set
        val_end: End date for validation set start date for test
        
    Returns:
        Dictionary with 'train', 'val' and 'test' splits
    """
    val_start = pl.Series([val_start]).cast(pl.Date)[0]
    val_end = pl.Series([val_end]).cast(pl.Date)[0]

    return {
        "train": df.filter(pl.col("ds") < val_start),
        "val": df.filter((pl.col("ds") >= val_start)&( pl.col("ds") < val_end)),
        "test": df.filter(pl.col("ds") >= val_end)
    }


def make_feature_dicts(
    dfs: Dict[str, pl.DataFrame], 
    DATA_SPLITS: list[str] = ["train", "val", "test"],
    holiday_df: pl.DataFrame = None,
    h: int = H,
    freq: str = FREQ,
    season_length: int = SEASON_LENGTH
) -> Tuple[Dict[str, Dict[str, pl.DataFrame]], Dict[str, Dict[str, pl.DataFrame|None]]]:
    """
    Create feature engineering pipeline for cafe drinks prediction
    
    Features:
    - none: baseline (no additional features)
    - holidays: holiday indicators
    - fourier: Fourier terms for seasonality
    - trend: linear trend
    - fourier+trend+holidays: all features combined
    
    Args:
        dfs: Dictionary with train/val splits
        DATA_SPLITS: List of split names
        holiday_df: Holiday feature dataframe
        h: Forecast horizon (days)
        freq: Frequency string
        season_length: Length of seasonal period (7 for weekly)
        
    Returns:
        Tuple of (features_dict, future_features_dict)
    """
    if holiday_df is None:
        holiday_df = create_holiday_features()
    
    df: Dict[str, Dict[str, pl.DataFrame]] = {
        "none": {},
        "holidays": {}, 
        "fourier": {}, 
        "trend": {}, 
        "fourier+trend+holidays": {}
    }
    
    df_future: Dict[str, Dict[str, pl.DataFrame|None]] = {
        "none": {},
        "holidays": {}, 
        "fourier": {}, 
        "trend": {}, 
        "fourier+trend+holidays": {}
    }
    
    for split in DATA_SPLITS:
        print(f"Processing {split} split...")
        
        # Baseline (no features)
        df["none"][split] = dfs[split]
        df_future["none"][split] = None
        
        # Fourier features (captures seasonality)
        df["fourier"][split], df_future["fourier"][split] = fourier(
            df=df["none"][split], 
            freq=freq, 
            season_length=season_length,  # Weekly seasonality
            k=3,  # Number of Fourier terms (adjust as needed)
            h=h
        )
        
        # Trend features
        df["trend"][split], df_future["trend"][split] = trend(
            df=df["none"][split], 
            freq=freq, 
            h=h
        )
        
        # Holiday features
        df["holidays"][split] = dfs[split].join(holiday_df, "ds", "left")
        df_future["holidays"][split] = df_future["trend"][split].drop("trend").join(
            holiday_df, "ds", "left"
        )
        
        # Combined features
        df["fourier+trend+holidays"][split] = (
            df["holidays"][split]
            .join(df["fourier"][split].drop("y"), ["unique_id", "ds"])
            .join(df["trend"][split].drop("y"), ["unique_id", "ds"])
        )
        
        df_future["fourier+trend+holidays"][split] = (
            df_future["holidays"][split]
            .join(df_future["fourier"][split], ["unique_id", "ds"])
            .join(df_future["trend"][split], ["unique_id", "ds"])
        )
    
    return (df, df_future)

def save_features(
    df: Dict[str, Dict[str, pl.DataFrame]], 
    df_future: Dict[str, Dict[str, pl.DataFrame|None]]
) -> None:
    """
    Save all feature variants to disk
    
    Args:
        df: Features dictionary
        df_future: Future features dictionary
    """
    
    for feature_type in df.keys():
        for split in df[feature_type].keys():
            # Save current features
            filepath = f"../data/processed_data/{feature_type}_{split}.parquet"
            df[feature_type][split].write_parquet(filepath)
            print(f"Saved {filepath}")
            
            # Save future features (if they exist)
            if df_future[feature_type][split] is not None:
                future_filepath = f"../data/processed_data//{feature_type}_{split}_future.parquet"
                df_future[feature_type][split].write_parquet(future_filepath)
                print(f"Saved {future_filepath}")


def run_feature_engineering_pipeline(
    df_input: pl.DataFrame,
    val_start: str = "2024-01-01",
    val_end: str = "2025-01-01",
    save_output: bool = True
) -> Tuple[Dict[str, Dict[str, pl.DataFrame]], Dict[str, Dict[str, pl.DataFrame|None]]]:
    """
    Run the complete feature engineering pipeline
    
    Args:
        df_input: Input dataframe (df_capped)
        val_start: Date to split train/val
        val_end: Date to split val/test
        save_output: Whether to save features to disk
        
    Returns:
        Tuple of (features_dict, future_features_dict)
    """
  
    print("CAFE DRINKS FEATURE ENGINEERING PIPELINE")

    
    # Step 1: Quick information about our data
    print("\n1. Converting to time series format")
    print(f"   Shape: {df_input.shape}")
    print(f"   Drinks: {df_input['unique_id'].n_unique()}")
    print(f"   Date range: {df_input['ds'].min()} to {df_input['ds'].max()}")
    
    # Step 2: Split train/val/test
    print(f"\n2. Splitting data (test starts: {val_end})")
    dfs = split_train_test(df_input, val_start, val_end)
    print(f"   Train: {dfs['train'].shape[0]} rows")
    print(f"   Val: {dfs['val'].shape[0]} rows")
    print(f"   Test: {dfs['test'].shape[0]} rows")
    
    
    # Step 3: Engineer features
    print("\n4. Engineering features")
    df_features, df_future = make_feature_dicts(
        dfs=dfs,
        holiday_df=None,
        h=H
    )
    
    # Step 4: Save features
    if save_output:
        print("\n5. Saving features")
        save_features(df_features, df_future)
    
    
    return df_features, df_future
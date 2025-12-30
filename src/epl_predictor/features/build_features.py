"""
feature engineering module

transforms match data into ml-ready features using rolling averages
to capture team form. uses closed='left' to prevent data leakage.
"""

import pandas as pd
import numpy as np
import logging
from epl_predictor import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cleaned_data() -> pd.DataFrame | None:
    """load the cleaned dataset."""
    input_path = config.PROCESSED_DATA_DIR / "final_dataset.csv"
    
    if not input_path.exists():
        logger.error(f"File not found: {input_path}. Run make_dataset.py first.")
        return None
    
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['Date'])
    logger.info(f"Loaded {df.shape[0]} matches")
    return df


def create_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """transform match data into team-level rows (2 rows per match)."""
    logger.info("Reshaping data to team perspective...")
    
    # home team perspective
    home_df = df[[
        'Date', 'Season', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
        'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'FTR'
    ]].copy()
    
    home_df = home_df.rename(columns={
        'HomeTeam': 'Team', 'AwayTeam': 'Opponent',
        'FTHG': 'GoalsFor', 'FTAG': 'GoalsAgainst',
        'HS': 'ShotsFor', 'AS': 'ShotsAgainst',
        'HST': 'SoTFor', 'AST': 'SoTAgainst',
        'HC': 'CornersFor', 'AC': 'CornersAgainst',
        'HF': 'FoulsFor', 'AF': 'FoulsAgainst',
        'HY': 'YellowsFor', 'AY': 'YellowsAgainst',
        'HR': 'RedsFor', 'AR': 'RedsAgainst'
    })
    home_df['Venue'] = 'Home'
    
    # away team perspective
    away_df = df[[
        'Date', 'Season', 'AwayTeam', 'HomeTeam',
        'FTAG', 'FTHG', 'AS', 'HS', 'AST', 'HST',
        'AC', 'HC', 'AF', 'HF', 'AY', 'HY', 'AR', 'HR', 'FTR'
    ]].copy()
    
    away_df = away_df.rename(columns={
        'AwayTeam': 'Team', 'HomeTeam': 'Opponent',
        'FTAG': 'GoalsFor', 'FTHG': 'GoalsAgainst',
        'AS': 'ShotsFor', 'HS': 'ShotsAgainst',
        'AST': 'SoTFor', 'HST': 'SoTAgainst',
        'AC': 'CornersFor', 'HC': 'CornersAgainst',
        'AF': 'FoulsFor', 'HF': 'FoulsAgainst',
        'AY': 'YellowsFor', 'HY': 'YellowsAgainst',
        'AR': 'RedsFor', 'HR': 'RedsAgainst'
    })
    away_df['Venue'] = 'Away'
    
    # combine and sort
    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values(['Team', 'Date']).reset_index(drop=True)
    
    logger.info(f"Reshaped to {team_df.shape[0]} team-match rows")
    return team_df


def calculate_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """rolling averages for team form, excludes current match."""
    logger.info(f"Calculating rolling averages (window={window})...")
    
    stat_columns = [
        'GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst',
        'SoTFor', 'SoTAgainst', 'CornersFor', 'CornersAgainst',
        'FoulsFor', 'FoulsAgainst', 'YellowsFor', 'YellowsAgainst'
    ]
    
    for col in stat_columns:
        df[f'{col}_RollingAvg'] = (
            df.groupby('Team')[col]
            .rolling(window=window, min_periods=1, closed='left')
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    logger.info(f"Created {len(stat_columns)} rolling average features")
    return df


def merge_back_to_matches(team_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """convert team-level stats back to match-level format."""
    logger.info("Merging rolling stats back to match format...")
    
    rolling_cols = [col for col in team_df.columns if 'RollingAvg' in col]
    
    # home team stats
    home_rolling = team_df[team_df['Venue'] == 'Home'][
        ['Date', 'Team', 'Opponent'] + rolling_cols
    ].copy()
    home_rolling.columns = ['Date', 'HomeTeam', 'AwayTeam'] + [f'Home_{col}' for col in rolling_cols]
    
    # away team stats
    away_rolling = team_df[team_df['Venue'] == 'Away'][
        ['Date', 'Opponent', 'Team'] + rolling_cols
    ].copy()
    away_rolling.columns = ['Date', 'HomeTeam', 'AwayTeam'] + [f'Away_{col}' for col in rolling_cols]
    
    # merge
    features_df = original_df.merge(home_rolling, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
    features_df = features_df.merge(away_rolling, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
    
    logger.info(f"Merged features: {features_df.shape[1]} total columns")
    return features_df


def create_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """create home vs away differential features."""
    logger.info("Creating differential features...")
    
    base_stats = [
        'GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst',
        'SoTFor', 'SoTAgainst', 'CornersFor', 'CornersAgainst'
    ]
    
    diff_count = 0
    for stat in base_stats:
        home_col = f'Home_{stat}_RollingAvg'
        away_col = f'Away_{stat}_RollingAvg'
        
        if home_col in df.columns and away_col in df.columns:
            df[f'{stat}_Diff'] = df[home_col] - df[away_col]
            diff_count += 1
    
    logger.info(f"Created {diff_count} differential features")
    return df


def validate_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """check for unexpected nans in recent data."""
    logger.info("Validating feature engineering...")
    
    rolling_cols = [col for col in df.columns if 'RollingAvg' in col]
    recent_matches = df.tail(100)
    nan_count = recent_matches[rolling_cols].isna().sum().sum()
    
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in recent matches")
    else:
        logger.info("No unexpected NaN values in rolling features")
    
    if df['Date'].is_monotonic_increasing:
        logger.info("Data is properly sorted chronologically")
    else:
        logger.warning("Data is not sorted by date!")
    
    return df


def build_features(window: int = 5) -> pd.DataFrame | None:
    """main pipeline to build ml features from match data."""
    logger.info("Starting Feature Engineering Pipeline")
    logger.info(f"Rolling window: {window} games")
    
    # load data
    df = load_cleaned_data()
    if df is None:
        return None
    
    # transform to team perspective
    team_df = create_team_perspective(df)
    
    # calculate rolling stats
    team_df = calculate_rolling_stats(team_df, window=window)
    
    # merge back to match format
    features_df = merge_back_to_matches(team_df, df)
    
    # create differential features
    features_df = create_differential_features(features_df)
    
    # validate
    features_df = validate_no_leakage(features_df)
    
    # save
    output_path = config.PROCESSED_DATA_DIR / "features_dataset.csv"
    features_df.to_csv(output_path, index=False)
    
    logger.info(f"Feature engineering complete!")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Final shape: {features_df.shape[0]} rows x {features_df.shape[1]} columns")
    
    feature_cols = [col for col in features_df.columns if 'RollingAvg' in col or '_Diff' in col]
    logger.info(f"Created {len(feature_cols)} ML features")
    
    return features_df


if __name__ == "__main__":
    result = build_features(window=5)
    
    if result is not None:
        print("\nSample of engineered features:")
        sample_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 
                      'Home_GoalsFor_RollingAvg', 'Away_GoalsFor_RollingAvg', 'GoalsFor_Diff']
        existing_cols = [col for col in sample_cols if col in result.columns]
        print(result[existing_cols].tail(10))

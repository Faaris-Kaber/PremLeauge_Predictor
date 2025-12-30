"""
data cleaning module

compiles multiple seasons into a single clean dataset,
filters unplayed games and optimizes memory usage.
"""

import pandas as pd
import logging
from epl_predictor import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# columns to keep
KEEP_COLUMNS: list[str] = [
    'Date', 'HomeTeam', 'AwayTeam', 
    'FTHG', 'FTAG', 'FTR',
    'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HF', 'AF',
    'HY', 'AY', 'HR', 'AR'
]

# use smaller int types for memory
DTYPE_SPEC: dict[str, str] = {
    'FTHG': 'int8', 'FTAG': 'int8',
    'HS': 'int8', 'AS': 'int8',
    'HST': 'int8', 'AST': 'int8',
    'HC': 'int8', 'AC': 'int8',
    'HF': 'int8', 'AF': 'int8',
    'HY': 'int8', 'AY': 'int8',
    'HR': 'int8', 'AR': 'int8'
}


def compile_data() -> pd.DataFrame | None:
    """compile and clean multiple seasons into a single dataset."""
    all_seasons: list[pd.DataFrame] = []
    
    logger.info("Starting Data Cleaning")
    
    # load each season file
    for season in config.SEASONS:
        filename = f"{config.LEAGUE_CODE}_{season}.csv"
        file_path = config.RAW_DATA_DIR / filename

        if not file_path.exists():
            logger.warning(f"File not found: {filename}. Skipped.")
            continue
            
        try:
            logger.info(f"Processing: {filename}")
            df = pd.read_csv(file_path, dtype=DTYPE_SPEC)
            df['Season'] = season
            all_seasons.append(df)
        except pd.errors.EmptyDataError:
            logger.error(f"Empty file: {filename}. Skipped.")
        except pd.errors.ParserError as e:
            logger.error(f"Parse error in {filename}: {e}. Skipped.")
        except Exception as e:
            logger.error(f"Unexpected error reading {filename}: {e}. Skipped.")

    if not all_seasons:
        logger.error("No data found. Did you run fetch_data.py?")
        return None

    # combine into one table
    main_df = pd.concat(all_seasons, ignore_index=True)
    total_raw_rows = main_df.shape[0]
    logger.info(f"Loaded {total_raw_rows} total rows from {len(all_seasons)} seasons")

    # parse dates (european format)
    main_df['Date'] = pd.to_datetime(main_df['Date'], dayfirst=True, errors='coerce')
    
    invalid_dates = main_df['Date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} rows with invalid dates. Removing.")
        main_df = main_df.dropna(subset=['Date'])

    # keep only played games
    valid_results = ['H', 'D', 'A']
    main_df = main_df.dropna(subset=['FTR'])
    main_df['FTR'] = main_df['FTR'].astype(str).str.strip()
    main_df = main_df[main_df['FTR'].isin(valid_results)]
    
    rows_dropped = total_raw_rows - main_df.shape[0]
    logger.info(f"Removed {rows_dropped} rows (unplayed/invalid matches)")

    # remove duplicates
    duplicates = main_df.duplicated(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate matches. Keeping first.")
        main_df = main_df[~duplicates]

    # keep only needed columns
    cols_to_keep = KEEP_COLUMNS + ['Season']
    existing_cols = [c for c in cols_to_keep if c in main_df.columns]
    main_df = main_df[existing_cols]

    # use categorical types for memory
    main_df['HomeTeam'] = main_df['HomeTeam'].astype('category')
    main_df['AwayTeam'] = main_df['AwayTeam'].astype('category')
    main_df['FTR'] = main_df['FTR'].astype('category')
    main_df['Season'] = main_df['Season'].astype('category')

    # sort by date
    main_df = main_df.sort_values('Date').reset_index(drop=True)

    # save
    output_path = config.PROCESSED_DATA_DIR / "final_dataset.csv"
    main_df.to_csv(output_path, index=False)
    
    logger.info(f"Success! Data saved to: {output_path}")
    logger.info(f"Final: {main_df.shape[0]} rows x {main_df.shape[1]} columns")
    logger.info(f"Date range: {main_df['Date'].min().date()} to {main_df['Date'].max().date()}")
    
    return main_df


if __name__ == "__main__":
    result = compile_data()
    if result is not None:
        logger.info(f"Memory usage: {result.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

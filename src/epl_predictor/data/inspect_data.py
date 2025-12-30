import pandas as pd
from epl_predictor import config

def inspect():
    # grab the most recent season file
    latest_season = config.SEASONS[-1]
    filename = f"{config.LEAGUE_CODE}_{latest_season}.csv"
    file_path = config.RAW_DATA_DIR / filename
    
    print(f"Inspecting: {filename}")
    
    df = pd.read_csv(file_path)
    
    print(f"Dimensions: {df.shape[0]} matches, {df.shape[1]} columns")
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Column Names ---")
    print(list(df.columns))

if __name__ == "__main__":
    inspect()
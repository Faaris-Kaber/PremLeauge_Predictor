"""
prediction cli

make predictions for upcoming matches using the trained model.

usage:
    python -m epl_predictor.predict --home "Arsenal" --away "Chelsea"
"""

import argparse
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from epl_predictor import config


def load_model() -> tuple[torch.nn.Module, object, list[str]]:
    """load trained model, scaler, and feature columns."""
    model_path = config.PROJECT_ROOT / 'saved_models' / 'epl_predictor.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # rebuild model
    from epl_predictor.models.train_model import EPLNet
    
    feature_columns = checkpoint['feature_columns']
    model = EPLNet(input_dim=len(feature_columns))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    return model, scaler, feature_columns


def load_latest_team_stats(features_df: pd.DataFrame, team: str, feature_columns: list[str]) -> dict[str, float] | None:
    """get the most recent rolling stats for a team."""
    # find matches where this team played
    home_matches = features_df[features_df['HomeTeam'] == team].copy()
    away_matches = features_df[features_df['AwayTeam'] == team].copy()
    
    if len(home_matches) == 0 and len(away_matches) == 0:
        return None
    
    # most recent match for this team
    all_matches = pd.concat([home_matches, away_matches]).sort_values('Date')
    latest = all_matches.iloc[-1]
    
    # extract rolling stats
    stats = {}
    
    if latest['HomeTeam'] == team:
        # team played at home
        for col in feature_columns:
            if col.startswith('Home_'):
                stats[col] = latest[col]
            elif col.startswith('Away_'):
                # keep away stats for differential calc
                stats[col] = latest[col]
            elif '_Diff' in col:
                stats[col] = latest[col]
    else:
        # team played away
        for col in feature_columns:
            if col.startswith('Away_'):
                stats[col] = latest[col]
            elif col.startswith('Home_'):
                stats[col] = latest[col]
            elif '_Diff' in col:
                stats[col] = latest[col]
    
    return stats


def get_match_features(
    features_df: pd.DataFrame, 
    home_team: str, 
    away_team: str, 
    feature_columns: list[str]
) -> np.ndarray | None:
    """build feature vector for a match between two teams."""
    
    # latest stats for each team
    home_matches = features_df[features_df['HomeTeam'] == home_team].sort_values('Date')
    away_matches = features_df[features_df['AwayTeam'] == away_team].sort_values('Date')
    
    if len(home_matches) == 0:
        print(f"Error: No data found for {home_team} playing at home.")
        return None
    
    if len(away_matches) == 0:
        print(f"Error: No data found for {away_team} playing away.")
        return None
    
    # most recent home match for home team
    latest_home = home_matches.iloc[-1]
    
    # most recent away match for away team
    latest_away = away_matches.iloc[-1]
    
    # build feature vector
    features = []
    missing_cols: list[str] = []
    for col in feature_columns:
        if col.startswith('Home_'):
            if col in latest_home.index:
                features.append(latest_home[col])
            else:
                missing_cols.append(col)
                features.append(0.0)
        elif col.startswith('Away_'):
            if col in latest_away.index:
                features.append(latest_away[col])
            else:
                missing_cols.append(col)
                features.append(0.0)
        elif '_Diff' in col:
            # recalculate differential
            base_stat = col.replace('_Diff', '')
            home_col = f'Home_{base_stat}_RollingAvg'
            away_col = f'Away_{base_stat}_RollingAvg'
            if home_col in latest_home and away_col in latest_away:
                diff = latest_home[home_col] - latest_away[away_col]
                features.append(diff)
            else:
                if home_col not in latest_home.index:
                    missing_cols.append(home_col)
                if away_col not in latest_away.index:
                    missing_cols.append(away_col)
                features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def predict_match(home_team: str, away_team: str) -> None:
    """predict the outcome of a match."""
    
    # load model
    print("Loading model...")
    model, scaler, feature_columns = load_model()
    
    # load features dataset
    features_path = config.PROCESSED_DATA_DIR / "features_dataset.csv"
    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        print("Run: python -m epl_predictor.features.build_features")
        return
    
    features_df = pd.read_csv(features_path, parse_dates=['Date'])
    
    # available teams
    all_teams = set(features_df['HomeTeam'].unique()) | set(features_df['AwayTeam'].unique())
    
    # validate team names
    if home_team not in all_teams:
        print(f"Error: '{home_team}' not found in dataset.")
        print(f"\nAvailable teams:")
        for team in sorted(all_teams):
            print(f"  - {team}")
        return
    
    if away_team not in all_teams:
        print(f"Error: '{away_team}' not found in dataset.")
        print(f"\nAvailable teams:")
        for team in sorted(all_teams):
            print(f"  - {team}")
        return
    
    # get features for this match
    features = get_match_features(features_df, home_team, away_team, feature_columns)
    if features is None:
        return
    
    # handle nans
    features = np.nan_to_num(features, nan=0.0)
    
    # scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # predict
    with torch.no_grad():
        inputs = torch.FloatTensor(features_scaled)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
    
    # decode prediction
    outcomes = ['Home Win', 'Draw', 'Away Win']
    predicted_idx = np.argmax(probabilities)
    predicted_outcome = outcomes[predicted_idx]
    
    # print results
    print(f"\n{'='*50}")
    print(f"  {home_team} vs {away_team}")
    print(f"{'='*50}")
    print(f"\n  Prediction: {predicted_outcome}")
    print(f"\n  Probabilities:")
    print(f"    Home Win: {probabilities[0]*100:5.1f}%")
    print(f"    Draw:     {probabilities[1]*100:5.1f}%")
    print(f"    Away Win: {probabilities[2]*100:5.1f}%")
    print(f"{'='*50}\n")


def main() -> None:
    # warn if not using the repo venv interpreter
    venv_py = config.PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    try:
        if venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
            print(
                f"Warning: you're running with {sys.executable}, but this repo's venv is {venv_py}. "
                f"Prefer: {venv_py} -m epl_predictor.predict ..."
            )
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description='Predict EPL match outcomes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m epl_predictor.predict --home "Arsenal" --away "Chelsea"
  python -m epl_predictor.predict --home "Liverpool" --away "Man United"
        '''
    )
    parser.add_argument('--home', required=True, help='Home team name (e.g., "Arsenal")')
    parser.add_argument('--away', required=True, help='Away team name (e.g., "Chelsea")')
    
    args = parser.parse_args()
    predict_match(args.home, args.away)


if __name__ == "__main__":
    main()


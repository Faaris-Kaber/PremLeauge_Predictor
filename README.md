# Premier League Match Predictor

A machine learning pipeline that predicts English Premier League match outcomes (Home Win / Draw / Away Win) using historical match statistics and team form. The model uses rolling averages to capture team momentum while carefully preventing data leakage through chronological data splits.

## Tech Stack

- **Python 3.11+**
- **PyTorch** - Neural network training (with Intel XPU support)
- **Pandas** - Data manipulation and feature engineering
- **scikit-learn** - Preprocessing and evaluation metrics
- **Matplotlib/Seaborn** - Visualizations
- **Requests** - Data fetching from football-data.co.uk

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PL_Predictor.git
cd PL_Predictor

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .
```

## Usage

Run the pipeline in order:

```bash
# 1. Download match data (2020-2025 seasons)
python -m epl_predictor.data.fetch_data

# 2. (Optional) Inspect the raw data
python -m epl_predictor.data.inspect_data

# 3. Clean and merge the data
python -m epl_predictor.data.make_dataset

# 4. Engineer features (rolling averages, differentials)
python -m epl_predictor.features.build_features

# 5. Train the neural network
python -m epl_predictor.models.train_model
```

## Making Predictions

Once the model is trained, predict match outcomes:

```bash
python -m epl_predictor.predict --home "Arsenal" --away "Chelsea"
```

Or use the PowerShell wrapper script:

```powershell
.\scripts\run_predict.ps1 -HomeTeam "Arsenal" -AwayTeam "Chelsea"
```

Example output:

```
==================================================
  Arsenal vs Chelsea
==================================================

  Prediction: Home Win

  Probabilities:
    Home Win:  52.3%
    Draw:      24.1%
    Away Win:  23.6%
==================================================
```

## Visualizations

Generate analysis plots:

```bash
python -m epl_predictor.visualize
```

This creates three visualizations in the `figures/` directory:

- **class_distribution.png** - Match outcome distribution (H/D/A)
- **confusion_matrix.png** - Model prediction accuracy by class
- **feature_correlations.png** - Most predictive features
- 
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a26a5a54-05c3-4812-a2c5-23bde908ba19" width="350">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a5d22d8e-27d4-4577-8406-d5adbc7d2a22" width="350">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2bda5b1d-9e6d-478c-8e9a-e14a474b65ad" width="350">
    </td>
  </tr>
</table>


## ML Approach

1. **Data**: 5 seasons of EPL match statistics from football-data.co.uk
2. **Features**: 32 rolling average features (5-game window) capturing:
   - Goals scored/conceded
   - Shots and shots on target
   - Corners, fouls, cards
   - Home vs Away differentials
3. **Model**: 2-layer neural network (64 → 32 → 3) with:
   - BatchNorm and Dropout for regularization
   - Class-weighted loss to handle imbalance (Draws are rare)
   - Early stopping to prevent overfitting
4. **Evaluation**: Chronological split (Train: 2020-23, Val: 2023-24, Test: 2024-25)

## Project Structure

```
PL_Predictor/
├── data/
│   ├── raw/                 # Downloaded CSV files
│   └── processed/           # Cleaned and featured datasets
├── figures/                 # Generated visualizations
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/
│   └── run_predict.ps1      # PowerShell wrapper for predictions
├── saved_models/            # Trained model checkpoints
├── src/
│   └── epl_predictor/
│       ├── __init__.py
│       ├── config.py        # Paths and configuration
│       ├── predict.py       # CLI prediction tool
│       ├── visualize.py     # Generate plots
│       ├── data/
│       │   ├── fetch_data.py      # Download match data
│       │   ├── inspect_data.py    # Quick data inspection
│       │   └── make_dataset.py    # Clean and merge
│       ├── features/
│       │   └── build_features.py  # Feature engineering
│       └── models/
│           └── train_model.py     # Neural network training
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Results

| Metric        | Value |
| ------------- | ----- |
| Test Accuracy | ~47%  |
| Home Win F1   | ~0.59 |
| Draw F1       | ~0.17 |
| Away Win F1   | ~0.48 |

_Note: Football prediction is inherently difficult. Professional tipsters typically achieve 50-55% accuracy. The baseline (always predicting Home Win) is ~43%._

## Data Source

Match data from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) - free historical football statistics.

## License

MIT License

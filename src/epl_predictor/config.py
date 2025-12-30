import rootpath
from pathlib import Path


BASE_URL = "https://www.football-data.co.uk/mmz4281/"
LEAGUE_CODE = "E0"

# seasons to download
SEASONS = [
    '2021',
    '2122',
    '2223',
    '2324',
    '2425'
]

try:
    PROJECT_ROOT = Path(rootpath.detect())
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
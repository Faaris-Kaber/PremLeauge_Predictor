import requests
from epl_predictor import config

def init_directories():
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_csv(session, season, league_code):
    url = f"{config.BASE_URL}{season}/{league_code}.csv"
    output_filename = f"{league_code}_{season}.csv"
    output_path = config.RAW_DATA_DIR / output_filename

    print(f"Fetching: {season} from {url}...")

    try:
        response = session.get(url)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        print(f"Success, saved to {output_filename}")

    except requests.exceptions.RequestException as e:
        print(f"error downloading {season}: {e}")

def main():
    print ("---starting data download---")

    init_directories()

    with requests.Session() as session:
        for season in config.SEASONS:
            fetch_csv(session, season, config.LEAGUE_CODE)
    
    print("---download complete")

if __name__ == "__main__":
    main()



import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import os

# Setup logging
logging.basicConfig(filename="fetch_subtitles_parallel.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Function to fetch subtitles
def get_imdb_subtitles(imdb_id):
    base_url = "https://wizdom.xyz/api/search"
    params = {"action": "by_id", "imdb": imdb_id}
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        return imdb_id, response.json()  # Return IMDb ID and response
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch subtitles for {imdb_id}: {e}")
        return imdb_id, []

# Main script
def main():
    try:
        # Load DataFrame
        df_imdb = pd.read_csv("/sise/home/lielbin/The-Art-of-Analyzing-Big-Data/data/imdb_episodes_with_season.csv")

        # Step 1: Unique tconst_season values
        unique_tconst_seasons = df_imdb['tconst_season'].dropna().unique()

        # Step 2: Parallel fetching with ThreadPoolExecutor
        results = []
        output_file = "subtitles_data.csv"
        save_every = 1000  # Save every 1000 records

        # Check if output file exists to resume from it
        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file)
            existing_ids = set(existing_data['tconst_season'].unique())
            logging.info(f"Resuming from existing data. Found {len(existing_ids)} IMDb IDs already processed.")
        else:
            existing_data = pd.DataFrame(columns=["tconst_season", "id", "versioname"])
            existing_ids = set()

        with ThreadPoolExecutor(max_workers=40) as executor:  # Adjust workers based on system
            futures = [executor.submit(get_imdb_subtitles, tconst) 
                       for tconst in unique_tconst_seasons if tconst not in existing_ids]

            # Process results with a progress bar
            for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Fetching IMDb Subtitles")):
                imdb_id, response_data = future.result()
                for record in response_data:
                    results.append({
                        "tconst_season": imdb_id,
                        "id": record.get("id"),
                        "versioname": record.get("versioname")
                    })

                # Save results periodically
                if len(results) >= save_every:
                    df_chunk = pd.DataFrame(results)
                    df_chunk.drop_duplicates(inplace=True)
                    df_chunk.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
                    logging.info(f"Saved {len(results)} records to {output_file}")
                    results = []  # Clear the buffer

        # Step 3: Save any remaining results
        if results:
            df_remaining = pd.DataFrame(results)
            df_remaining.drop_duplicates(inplace=True)
            df_remaining.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
            logging.info(f"Saved remaining {len(results)} records to {output_file}")

        logging.info("Subtitle data fetched and saved successfully.")

    except Exception as e:
        logging.critical(f"Script failed: {e}")

if __name__ == "__main__":
    main()

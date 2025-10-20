import pandas as pd

def load_connections_game(csv_path, game_id=None):
    """
    Load a Connections-style dataset (optionally for one specific game),
    converting all text data to lowercase.

    Args:
        csv_path (str): Path to the CSV file.
        game_id (int or None): If provided, filter by Game ID.

    Returns:
        data (dict): {
            'game_id': int,
            'puzzle_date': str,
            'groups': {
                group_name: {
                    'level': int,
                    'words': [list of words]
                }, ...
            },
            'all_words': [all words in the game]
        }
    """
    df = pd.read_csv(csv_path)

    # Convert column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Filter by Game ID if given
    if game_id is not None:
        df = df[df["game id"] == game_id]
        if df.empty:
            raise ValueError(f"No puzzle found for Game ID {game_id}")

    game_id = int(df["game id"].iloc[0])
    puzzle_date = str(df["puzzle date"].iloc[0]).lower()

    # Build group mapping
    groups = {}
    for group_name, group_df in df.groupby("group name"):
        groups[group_name.lower()] = {
            "level": int(group_df["group level"].iloc[0]),
            "words": [w.lower() for w in group_df["word"].tolist()]
        }

    all_words = [w.lower() for w in df["word"].tolist()]

    return {
        "game_id": game_id,
        "puzzle_date": puzzle_date,
        "groups": groups,
        "all_words": all_words
    }

# Example usage
# if __name__ == "__main__":
#     data = load_connections_game("connections_data/Connections_Data.csv", game_id=870)
#     print(data["game_id"], data["puzzle_date"])
#     print(data["groups"].keys())
#     print(data["groups"])


def get_all_connections_words(csv_path):
    """Return a set of all unique words across all Connections games."""
    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Collect and lowercase all words
    all_words = df["word"].astype(str).str.lower().str.strip()
    
    unique_words = set(all_words)
    print(f"Total unique words: {len(unique_words)}")
    
    return unique_words

# Example usage
all_words = get_all_connections_words("connections_data/Connections_Data.csv")
print(list(all_words)[:20])  # show first 20
import pandas as pd

def load_connections_game(csv_path, game_id=None):
    """
    Load a Connections-style dataset (optionally for one specific game).

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

    # Filter by Game ID if given
    if game_id is not None:
        df = df[df["Game ID"] == game_id]
        if df.empty:
            raise ValueError(f"No puzzle found for Game ID {game_id}")

    
    game_id = int(df["Game ID"].iloc[0])
    puzzle_date = str(df["Puzzle Date"].iloc[0])

    # build group mapping
    groups = {}
    for group_name, group_df in df.groupby("Group Name"):
        groups[group_name] = {
            "level": int(group_df["Group Level"].iloc[0]),
            "words": group_df["Word"].tolist()
        }

    all_words = df["Word"].tolist()

    return {
        "game_id": game_id,
        "puzzle_date": puzzle_date,
        "groups": groups,
        "all_words": all_words
    }



# Example usage
if __name__ == "__main__":
    data = load_connections_game("connections_data/Connections_Data.csv", game_id=870)
    print(data["game_id"], data["puzzle_date"])
    print(data["groups"].keys())
    print(data["groups"]["INTO IT"])
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle


def get_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    csv_file = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
    # Load only needed columns, using efficient dtypes
    # df = pd.read_csv(csv_file, sep="\t", usecols=['cue', 'response'])
    min_strength = 0.05
    strength_col = "R123.Strength"
    df = pd.read_csv(csv_file, sep="\t", usecols=["cue", "response", strength_col])

    df = df[df[strength_col] >= min_strength]

    # Map words to integer IDs (vectorized)
    all_words = pd.Index(df["cue"]).append(pd.Index(df["response"])).unique()

    word2idx = pd.Series(range(len(all_words)), index=all_words)

    # Get unique words from the dataset
    just_words_actual = set(df["cue"].dropna()).union(set(df["response"].dropna()))
    just_words = just_words_actual

    print(f"Obtaining embeddings for {len(just_words)} words...")

    words_list = list(just_words)
    word_embeddings = model.encode(words_list, show_progress_bar=True)

    embeddings = {}
    for i, word in enumerate(words_list):
        if word in word2idx:
            embeddings[word2idx[word]] = word_embeddings[i]

    print(f"Generated embeddings for {len(embeddings)} words")

    with open("models/embeddings.pickle", "wb") as handle:
        pickle.dump(embeddings, handle)

    # print("saved it bruv")
    # print(all_embeddings)


if __name__ == "__main__":
    get_embeddings()

import fasttext.util
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
import pickle

def get_embeddings():

    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')

    csv_file = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
    # Load only needed columns, using efficient dtypes
    # df = pd.read_csv(csv_file, sep="\t", usecols=['cue', 'response'])
    min_strength=0.05
    strength_col='R123.Strength'
    df = pd.read_csv(csv_file, sep="\t", usecols=['cue', 'response', strength_col])

    df = df[df[strength_col] >= min_strength]


    # Map words to integer IDs (vectorized)
    all_words = pd.Index(df['cue']).append(pd.Index(df['response'])).unique()

    word2idx = pd.Series(range(len(all_words)), index=all_words)
    idx2word = dict(enumerate(all_words))

    # print(df.head())
    # just_words = set(df['cue']).union(set(df["response"]))
    just_words_actual = set(df['cue'].dropna()).union(set(df["response"].dropna()))
    # print("Length of just words:", len(just_words))
    # print("Length of just words actual:", len(just_words_actual))

    just_words = just_words_actual

    # exit()
    # print("Obtaining embeddings...")
    # get embeddings for all words. costly, but whatever.
    # all_embeddings = all_words.map(ft.get_word_vector)

    embeddings = {}

    for word in just_words:
        try:
            embeddings[word2idx[word]] = ft.get_word_vector(word)
        
        except:
            print(f"failed with `word` {word}")
            raise Exception
        
        # print(".", end="")
    print("")

    with open('embeddings.pickle', 'wb') as handle:
        pickle.dump(embeddings, handle)

    # print("saved it bruv")
    # print(all_embeddings)

if __name__ == "__main__":
    get_embeddings()
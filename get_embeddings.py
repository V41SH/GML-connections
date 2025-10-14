import fasttext.util
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

csv_file = "SWOW-EN18/strength.SWOW-EN.R123.20180827.csv"
# Load only needed columns, using efficient dtypes
df = pd.read_csv(csv_file, sep="\t", usecols=['cue', 'response'])

# Map words to integer IDs (vectorized)
all_words = pd.Index(df['cue']).append(pd.Index(df['response'])).unique()

# get embeddings for all words. costly, but whatever.
all_embeddings = all_words.map(ft.get_word_vector)

print(all_embeddings)
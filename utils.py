import requests
import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)

def get_proxy_words(word, word2idx, embedding, idx2word, top_k_textsim=3):
    """
    Given an OOV word, find related SWOW words to approximate its embedding.

    Fallback hierarchy:
        1. If word exists in SWOW → itself.
        2. If proper noun → use Wikipedia summary nouns that exist in SWOW.
        3. Else WordNet hypernyms that exist in SWOW.
        4. Else nearest SWOW nodes by cosine similarity of name (text-level).
    """
    word_l = word.lower().strip()

    # Already in SWOW
    if word_l in word2idx:
        print(f"'{word}' found in SWOW vocabulary.")
        return [word_l]

    proxies = []

    # Named entity or acronym → use Wikipedia summary nouns
    if word.isupper() or word.istitle():
        try:
            resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{word}",
                timeout=3
            )
            if resp.status_code == 200:
                text = resp.json().get("extract", "")
                tokens = [
                    t.lower()
                    for t, pos in pos_tag(word_tokenize(text))
                    if pos.startswith("NN")
                ]
                proxies = [t for t in tokens if t in word2idx][:3]
                print(f"Wikipedia nouns for '{word}': {proxies}")
                if proxies:
                    return proxies
        except Exception:
            pass  # silent fallback

    # WordNet hypernyms
    syns = wn.synsets(word_l)
    if syns:
        for hyp in syns[0].hypernyms():
            for name in hyp.lemma_names():
                n = name.lower().replace("_", " ")
                if n in word2idx:
                    proxies.append(n)
                    print(f"WordNet hypernym for '{word}': {n}")
        if proxies:
            return list(set(proxies))[:3]

    # Text similarity fallback (using string similarity between names)
    all_words = np.array(list(word2idx.keys()))
    sims = np.array([
        1 - abs(len(w) - len(word_l)) / max(len(w), len(word_l))
        for w in all_words
    ])
    print(f"Text similarity scores for '{word}': {sims}")
    top_idx = sims.argsort()[::-1][:top_k_textsim]
    return list(all_words[top_idx])

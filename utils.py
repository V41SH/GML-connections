import jellyfish


def phonetic_code(word: str) -> str:
    """
    Convert a word to its Metaphone phonetic code.
    """
    if not word:
        return ""
    return jellyfish.metaphone(word.lower())


def phonetic_similarity(w1: str, w2: str) -> float:
    """
    Compute a similarity score between 0 and 1 based on
    Levenshtein distance of Metaphone codes.
    """
    p1, p2 = phonetic_code(w1), phonetic_code(w2)
    max_len = max(len(p1), len(p2))
    if max_len == 0:
        return 0.0
    dist = jellyfish.levenshtein_distance(p1, p2)
    return 1 - dist / max_len


def phonetic_distance(w1: str, w2: str) -> float:
    """
    Return the raw normalized distance (1 - similarity).
    """
    return 1 - phonetic_similarity(w1, w2)

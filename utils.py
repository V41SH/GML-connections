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


def analyze_phonetic_pairs(word_pairs, threshold=0.7):
    """
    Analyze phonetic similarity for a list of word pairs.

    Args:
        word_pairs: List of (word1, word2) tuples
        threshold: Minimum similarity to consider as phonetically similar

    Returns:
        List of analysis results with similarity scores and codes
    """
    results = []

    for w1, w2 in word_pairs:
        similarity = phonetic_similarity(w1, w2)
        code1 = phonetic_code(w1)
        code2 = phonetic_code(w2)

        results.append(
            {
                "word1": w1,
                "word2": w2,
                "similarity": similarity,
                "code1": code1,
                "code2": code2,
                "above_threshold": similarity >= threshold,
            }
        )

    return results

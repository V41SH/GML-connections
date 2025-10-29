import pandas as pd

from wordninja import split as wordninja_split
from typing import Set, List, Tuple

def segment_compound_words(
    word: str, 
    existing_vocab: Set[str],
    min_subword_length: int = 3
) -> List[str]:
    """
    Segment a word into subwords, keeping only those in existing vocabulary.
    """
    # Try word segmentation
    segments = wordninja_split(word)
    
    # Filter: only keep segments that exist in ConceptNet and are long enough
    valid_segments = [
        seg.lower() for seg in segments 
        if seg.lower() in existing_vocab and len(seg) >= min_subword_length
    ]
    
    # Only return if we found multiple valid segments and it's not the word itself
    if len(valid_segments) > 1 and valid_segments != [word]:
        return valid_segments
    return []

def add_compound_edges(
    df: pd.DataFrame,
    word2idx: dict,
    existing_vocab: Set[str],
    compound_weight: float = 0.5,
    compound_relation: str = "compound_subword"
) -> pd.DataFrame:
    """
    Add edges for compound word decompositions.
    Returns a new dataframe with additional edges.
    """
    new_edges = []
    
    # Get all unique words (potential compounds)
    all_words = set(df['start_word'].unique()) | set(df['end_word'].unique())
    
    print(f"Analyzing {len(all_words):,} words for compound structure...")
    
    for word in all_words:
        subwords = segment_compound_words(word, existing_vocab)
        
        if subwords:
            # Create bidirectional edges between compound and its parts
            
            for subword in subwords:
                # Compound → subword
                new_edges.append({
                    'start_word': word,
                    'end_word': subword,
                    'relation': compound_relation,
                    'weight': compound_weight
                })
                # Subword → compound (for undirected signal)
                new_edges.append({
                    'start_word': subword,
                    'end_word': word,
                    'relation': compound_relation,
                    'weight': compound_weight
                })
    
    if new_edges:
        new_df = pd.DataFrame(new_edges)
        print(f"Added {len(new_edges):,} compound edges for {len(new_edges)//2:,} compound words")
        return pd.concat([df, new_df], ignore_index=True)
    
    return df
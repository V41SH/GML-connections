import networkx as nx

G = nx.read_gml('graphs/conceptnet_graph.gml')

# Check a few candidate compound words directly
test_words = ['bedroom', 'daybed', 'murphy', 'sunlight', 'football', 'riverbed']

for word in test_words:
    if G.has_node(word):
        # Get only outgoing edges from this word
        neighbors = G[word]
        compound_edges = [
            (v, data) for v, data in neighbors.items()
            if data.get('relation') == 'compound_subword'
        ]
        if compound_edges:
            print(f"\n{word}:")
            for v, data in compound_edges:
                print(f"  → {v}")
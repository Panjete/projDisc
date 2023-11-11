from annoy import AnnoyIndex

# Example word-vector pairs (you can replace these with your own data)
word_vector_pairs = {
    'apple': [0.3, 0.5, 0.8, -0.2, 0.6],
    'banana': [0.2, 0.4, 0.7, -0.1, 0.5],
    'cherry': [0.4, 0.6, 0.9, -0.3, 0.7],
    'date': [0.1, 0.3, 0.6, -0.05, 0.4],
    'elderberry': [0.5, 0.7, 1.0, -0.4, 0.8],
}

# Define the number of dimensions in your vectors
vector_dim = len(next(iter(word_vector_pairs.values())))
# Initialize the Annoy index
t = AnnoyIndex(vector_dim, 'angular')  # 'angular' is suitable for cosine similarity

# Add word-vector pairs to the index
for i, vector in enumerate(word_vector_pairs.values()):
    t.add_item(i, vector)  # Add items with unique IDs

# Build the Annoy index
t.build(n_trees=10)  # Adjust the number of trees as needed
random_vector = [0.1, 0.3, 0.6, -0.05, 0.4]
# Find the 5 nearest neighbors of the random vector
nearest_indices = t.get_nns_by_vector(random_vector, n=5)
print(nearest_indices)

# Retrieve the nearest words based on the indices
nearest_words = [list(word_vector_pairs.keys())[index] for index in nearest_indices]
print("Random Vector:", random_vector)
print("5 Nearest Neighbors:", nearest_words)

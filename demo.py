"""Demo for VPR tools"""
import numpy as np
from utils import cosine_similarity, batch_cosine_similarity, find_top_k

np.random.seed(42)

# Test 1: cosine similarity
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Cosine similarity: {cosine_similarity(a, b):.4f}")

# Test 2: batch similarity
features = np.random.randn(5, 128)
sim_matrix = batch_cosine_similarity(features)
print(f"Similarity matrix shape: {sim_matrix.shape}")

# Test 3: top-k search
query = np.random.randn(128)
db = np.random.randn(1000, 128)
indices, sims = find_top_k(query, db, k=5)
print(f"Top-5 indices: {indices}")
print(f"Top-5 similarities: {sims.round(4)}")

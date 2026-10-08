"""VPR utility functions"""
import numpy as np

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def batch_cosine_similarity(X):
    """Compute NxN similarity matrix"""
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X_norm @ X_norm.T

def find_top_k(query, database, k=5):
    """Find top-k most similar vectors in database"""
    q_norm = query / np.linalg.norm(query)
    db_norm = database / np.linalg.norm(database, axis=1, keepdims=True)
    sims = db_norm @ q_norm
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]
